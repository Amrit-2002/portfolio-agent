import argparse
import hashlib
import logging
import os
from collections import defaultdict
from typing import Any

from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from .crawler import discover_site_urls
from .github_loader import (
    create_single_repo_document,
    fetch_github_repos,
    fetch_repo_readme,
)

logger = logging.getLogger(__name__)

os.environ["USER_AGENT"] = "my-rag-application/0.1 (contact: Amrit)"

load_dotenv()

# Configurable models and environment variables
llm_model_name = os.getenv("LLM_MODEL", "gemini-3.5-flash")
chat_model = init_chat_model(llm_model_name, model_provider="google_genai")
embeddings_model = GoogleGenerativeAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2-preview")
)

index_name = os.getenv("PINECONE_INDEX_NAME", "portfolio-rag-3072")
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

_rag_chain = None


def generate_chunk_id(
    source_type: str,
    resource_id: str,
    content: str,
    occurrence: int = 0,
) -> str:
    """
    Generate a deterministic, collision-resistant vector ID for a document chunk.
    Format: '{source_type}:{resource_id}:{content_hash}'
    Example: 'github:Amrit-2002/secure_token_manager:9b8f2a417c80d1e3'
    """
    normalized = " ".join(content.strip().split())
    raw = f"{source_type}:{resource_id}:{normalized}"
    content_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    suffix = f"_{occurrence}" if occurrence > 0 else ""
    return f"{source_type}:{resource_id}:{content_hash}{suffix}"


def generate_deterministic_chunk_ids(
    source_type: str,
    resource_id: str,
    chunks: list[Document],
) -> list[str]:
    """
    Generate deterministic IDs for a list of Document chunks,
    disambiguating identical chunk contents if any exist.
    """
    ids: list[str] = []
    seen_hashes: dict[str, int] = defaultdict(int)
    for c in chunks:
        normalized = " ".join(c.page_content.strip().split())
        raw = f"{source_type}:{resource_id}:{normalized}"
        chash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        occ = seen_hashes[chash]
        seen_hashes[chash] += 1
        suffix = f"_{occ}" if occ > 0 else ""
        ids.append(f"{source_type}:{resource_id}:{chash}{suffix}")
    return ids


def get_pinecone_ids_by_prefix(index: Any, prefix: str) -> list[str]:
    """
    Fetch all vector IDs in a Pinecone index matching the given prefix.
    Safely iterates through all paginated batches returned by index.list().
    """
    ids: list[str] = []
    try:
        for batch in index.list(prefix=prefix):
            if isinstance(batch, list):
                ids.extend(batch)
            elif hasattr(batch, "vectors"):
                ids.extend([v.id for v in batch.vectors])
    except Exception as e:
        logger.warning("Could not list Pinecone IDs for prefix '%s': %s", prefix, e)
    return ids


def delete_pinecone_vectors(index: Any, ids: list[str], batch_size: int = 100) -> None:
    """
    Delete vectors from Pinecone by ID in batches.
    """
    if not ids:
        return
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        try:
            index.delete(ids=batch)
        except Exception as e:
            logger.error("Failed to delete batch of %d vectors from Pinecone: %s", len(batch), e)
            raise


def setup_pinecone_with_external_data():
    """
    Initialize Pinecone index and index data from both the portfolio website and GitHub repositories
    using deterministic vector IDs.
    """
    logger.info("Setting up Pinecone with external data...")
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(name=index_name)
    vector_store = PineconeVectorStore(embedding=embeddings_model, index=index)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # 1. Crawl personal portfolio website
    portfolio_url = os.getenv("PORTFOLIO_URL", "https://amrit-2002.github.io/Portfolio/")
    logger.info("Discovering URLs from portfolio website: %s", portfolio_url)
    urls = discover_site_urls(
        portfolio_url,
        max_pages=15,
        include_subdomains=False,
        delay_seconds=0.5,
        allow_paths=None,
        deny_paths=["/admin", "/login"],
    )

    if urls:
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                all_splits = text_splitter.split_documents(docs)
                for split in all_splits:
                    split.metadata.update(
                        {
                            "type": "portfolio_page",
                            "source_type": "portfolio",
                            "source": url,
                            "url": url,
                        }
                    )
                chunk_ids = generate_deterministic_chunk_ids("portfolio", url, all_splits)
                _ = vector_store.add_documents(documents=all_splits, ids=chunk_ids)
            except Exception as e:
                logger.warning("Failed to load and index URL %s: %s", url, e)
    else:
        logger.warning("No URLs found to crawl for portfolio.")

    # 2. Ingest GitHub repositories via safe sync
    sync_github_to_pinecone()


def sync_github_to_pinecone(
    username: str | None = None,
    token: str | None = None,
    max_repos: int | None = None,
) -> dict[str, Any]:
    """
    Fetch and synchronize public GitHub repositories into Pinecone idempotently.
    - Adds new repositories.
    - Updates modified repositories while safely removing obsolete chunks.
    - Skips unchanged repositories without re-embedding.
    - Removes deleted repositories only when fetch is complete and confirmed.
    - Preserves portfolio vectors untouched.
    """
    github_user = username or os.getenv("GITHUB_USERNAME", "Amrit-2002")
    logger.info("Starting safe GitHub synchronization for %s...", github_user)

    result: dict[str, Any] = {
        "status": "success",
        "discovered_count": 0,
        "added_count": 0,
        "updated_count": 0,
        "unchanged_count": 0,
        "deleted_count": 0,
        "vectors_upserted": 0,
        "projects_synced": 0,
        "errors": [],
    }

    # 1. Fetch repositories
    fetch_result = fetch_github_repos(
        username=github_user,
        token=token,
        include_forks=False,
        max_repos=max_repos,
    )

    # 2. Check for API failure
    if not fetch_result.success:
        err_msg = fetch_result.error or "GitHub API request failed"
        logger.error("GitHub fetch failed for %s: %s", github_user, err_msg)
        result["status"] = "error"
        result["errors"].append(err_msg)
        return result

    # 3. Check for empty response edge case: distinguish genuine 0 repos from faulty list response
    if len(fetch_result.repos) == 0:
        if fetch_result.total_public_repos is not None and fetch_result.total_public_repos > 0:
            err_msg = (
                f"Safety abort: Profile reports {fetch_result.total_public_repos} public repos, "
                "but repository list returned 0. Aborting sync without deletions."
            )
            logger.error(err_msg)
            result["status"] = "error"
            result["errors"].append(err_msg)
            return result

    # 4. Connect to Pinecone index
    if not pc.has_index(index_name):
        setup_pinecone_with_external_data()
        result["status"] = "success"
        return result

    index = pc.Index(name=index_name)
    vector_store = PineconeVectorStore(embedding=embeddings_model, index=index)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # 5. Inventory existing GitHub vectors in Pinecone
    existing_github_ids = get_pinecone_ids_by_prefix(index, "github:")
    existing_by_repo: dict[str, set[str]] = defaultdict(set)
    for vid in existing_github_ids:
        if not vid.startswith("github:"):
            continue
        parts = vid.split(":", 2)
        if len(parts) >= 2:
            repo_full_name = parts[1]
            existing_by_repo[repo_full_name].add(vid)

    result["discovered_count"] = len(fetch_result.repos)
    active_repo_names: set[str] = set()

    # 6. Process each discovered repository
    for repo in fetch_result.repos:
        repo_name = repo.get("name") or "Unnamed"
        repo_full_name = repo.get("full_name") or f"{github_user}/{repo_name}"
        active_repo_names.add(repo_full_name)

        existing_ids = existing_by_repo.get(repo_full_name, set())

        # Load document with metadata and README
        readme_content = fetch_repo_readme(github_user, repo_name, token=token)
        repo_doc = create_single_repo_document(github_user, repo, readme_content=readme_content)

        chunks = text_splitter.split_documents([repo_doc])
        if not chunks:
            continue

        # Ensure all chunk metadata is set
        for c in chunks:
            c.metadata.update(repo_doc.metadata)

        chunk_ids = generate_deterministic_chunk_ids("github", repo_full_name, chunks)
        candidate_ids = set(chunk_ids)

        # Check if unchanged: candidate IDs match existing Pinecone vector IDs exactly
        if existing_ids and existing_ids == candidate_ids:
            logger.debug("Repository %s is unchanged. Skipping re-indexing.", repo_full_name)
            result["unchanged_count"] += 1
            continue

        is_new = len(existing_ids) == 0

        # Safe replacement strategy: upsert new content first, delete obsolete vectors second
        try:
            vector_store.add_documents(documents=chunks, ids=chunk_ids)
            result["vectors_upserted"] += len(chunks)

            if is_new:
                result["added_count"] += 1
                logger.info("Added new repository %s (%d vectors).", repo_full_name, len(chunks))
            else:
                result["updated_count"] += 1
                logger.info("Updated repository %s (%d vectors).", repo_full_name, len(chunks))

            # Delete obsolete chunks for this repo that are no longer present
            obsolete_ids = existing_ids - candidate_ids
            if obsolete_ids:
                delete_pinecone_vectors(index, list(obsolete_ids))
                logger.debug(
                    "Deleted %d obsolete chunks for %s.", len(obsolete_ids), repo_full_name
                )

        except Exception as e:
            logger.error("Failed to upsert vectors for %s: %s", repo_full_name, e)
            result["errors"].append(f"Failed to upsert {repo_full_name}: {e}")
            # Old repository vectors are preserved on failure!

    # 7. Remove deleted repositories (ONLY if inventory fetch was complete!)
    if fetch_result.is_complete:
        indexed_repo_names = set(existing_by_repo.keys())
        deleted_repo_names = indexed_repo_names - active_repo_names

        for del_repo in deleted_repo_names:
            del_ids = list(existing_by_repo[del_repo])
            if del_ids:
                try:
                    delete_pinecone_vectors(index, del_ids)
                    result["deleted_count"] += 1
                    logger.info(
                        "Deleted repository %s (%d vectors removed).", del_repo, len(del_ids)
                    )
                except Exception as e:
                    logger.error(
                        "Failed to delete vectors for removed repository %s: %s", del_repo, e
                    )
                    result["errors"].append(f"Failed to delete {del_repo}: {e}")
    else:
        logger.info(
            "Skipping deletion of unreferenced repositories because repository fetch was partial."
        )

    # 8. Compute total projects synced
    result["projects_synced"] = (
        result["added_count"] + result["updated_count"] + result["unchanged_count"]
    )

    if result["errors"]:
        result["status"] = "partial" if result["projects_synced"] > 0 else "error"
    else:
        result["status"] = "success"

    logger.info("GitHub synchronization completed: %s", result)
    return result


# Reformulates follow-up questions into standalone questions so the retriever
# gets a self-contained query regardless of what came before in the conversation.
_contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given a chat history and the latest user question which might reference context "
            "in the chat history, formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just reformulate it if "
            "needed and otherwise return it as is.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

_qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant for question-answering tasks about Amrit, including his portfolio, "
            "skills, experience, and projects hosted on his GitHub and portfolio website. "
            "Use the following pieces of retrieved context to answer the question. "
            "When describing projects, highlight key features, tools/technologies used, and provide the "
            "GitHub repository link or project URL whenever available in the context. "
            "If asked about recently updated or newly added projects, use the update timestamps from the context. "
            "If you don't know the answer, say that you don't know. "
            "Keep the answer concise, informative, and formatted in markdown.\n\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def _get_rag_chain():
    global _rag_chain
    if _rag_chain is not None:
        return _rag_chain

    if not pc.has_index(index_name):
        setup_pinecone_with_external_data()
    else:
        # Check if the index has no GitHub vectors yet
        try:
            gh_ids = get_pinecone_ids_by_prefix(pc.Index(name=index_name), "github:")
            if len(gh_ids) == 0:
                logger.info("No GitHub vectors found in Pinecone. Running initial sync...")
                sync_github_to_pinecone()
        except Exception as e:
            logger.warning("Could not check Pinecone index stats: %s", e)

    index = pc.Index(name=index_name)
    vector_store = PineconeVectorStore(embedding=embeddings_model, index=index)
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    history_aware_retriever = create_history_aware_retriever(
        chat_model, retriever, _contextualize_q_prompt
    )
    question_answer_chain = create_stuff_documents_chain(chat_model, _qa_prompt)
    _rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return _rag_chain


def retrieve_generate(question: str, chat_history: list | None = None) -> str:
    if chat_history is None:
        chat_history = []

    try:
        result = _get_rag_chain().invoke({"input": question, "chat_history": chat_history})
        return result["answer"]
    except ResourceExhausted:
        return "The AI model is rate-limited (free tier quota exceeded). Please wait about 30 seconds and try again."


def retrieve_generate_stream(question: str, chat_history: list | None = None):
    """Yield answer token chunks from the RAG chain as they're generated."""
    if chat_history is None:
        chat_history = []

    try:
        for chunk in _get_rag_chain().stream({"input": question, "chat_history": chat_history}):
            token = chunk.get("answer")
            if token:
                yield token
    except ResourceExhausted:
        yield "The AI model is rate-limited (free tier quota exceeded). Please wait about 30 seconds and try again."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portfolio Agent RAG Runner")
    parser.add_argument(
        "--sync-github", action="store_true", help="Sync latest GitHub projects to Pinecone"
    )
    parser.add_argument(
        "--ask", type=str, default="What are Amrit's skills?", help="Ask a question"
    )
    args = parser.parse_args()

    if args.sync_github:
        res = sync_github_to_pinecone()
        print(f"Status: {res['status']}")
        print(f"Discovered: {res['discovered_count']}")
        print(f"Added: {res['added_count']}")
        print(f"Updated: {res['updated_count']}")
        print(f"Unchanged: {res['unchanged_count']}")
        print(f"Deleted: {res['deleted_count']}")
        print(f"Vectors upserted: {res['vectors_upserted']}")
        if res.get("errors"):
            print(f"Errors: {res['errors']}")
    else:
        print(retrieve_generate(args.ask))
