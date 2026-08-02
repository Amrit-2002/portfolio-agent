import logging
import os

from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from .crawler import discover_site_urls

logger = logging.getLogger(__name__)

os.environ["USER_AGENT"] = "my-rag-application/0.1 (contact: Amrit)"

load_dotenv()

chat_model = init_chat_model("gemini-3.5-flash", model_provider="google_genai")
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")

index_name = "portfolio-rag-3072"
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

_rag_chain = None


def setup_pinecone_with_external_data():
    logger.info("Setting up Pinecone with external data...")
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    index = pc.Index(name=index_name)
    vector_store = PineconeVectorStore(embedding=embeddings_model, index=index)

    urls = discover_site_urls(
        "https://amrit-2002.github.io/Portfolio/",
        max_pages=15,
        include_subdomains=False,
        delay_seconds=0.5,
        allow_paths=None,
        deny_paths=["/admin", "/login"],
    )

    if not urls:
        logger.warning("No URLs found to crawl.")
        return

    for url in urls:
        loader = WebBaseLoader(url)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        _ = vector_store.add_documents(documents=all_splits)


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
            "You are an assistant for question-answering tasks about Amrit. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Keep the answer concise and informative.\n\n{context}",
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

    index = pc.Index(name=index_name)
    vector_store = PineconeVectorStore(embedding=embeddings_model, index=index)
    retriever = vector_store.as_retriever(search_kwargs={"k": 20})

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
    print(retrieve_generate("What are Amrit's skills?"))
