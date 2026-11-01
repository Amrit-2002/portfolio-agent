from unittest.mock import MagicMock, patch

import pytest

from app.github_loader import RepoFetchResult
from app.rag_application import (
    generate_chunk_id,
    generate_deterministic_chunk_ids,
    sync_github_to_pinecone,
)


@pytest.fixture()
def mock_pinecone():
    """Fixture providing mocked Pinecone index and vector store."""
    with patch("app.rag_application.pc.has_index", return_value=True):
        with patch("app.rag_application.pc.Index") as mock_index_cls:
            mock_index = MagicMock()
            mock_index_cls.return_value = mock_index
            with patch("app.rag_application.PineconeVectorStore") as mock_vs_cls:
                mock_vs = MagicMock()
                mock_vs_cls.return_value = mock_vs
                yield mock_index, mock_vs


def _make_repo(name: str, updated_at: str = "2026-10-01T00:00:00Z", desc: str = "desc") -> dict:
    return {
        "name": name,
        "full_name": f"testuser/{name}",
        "html_url": f"https://github.com/testuser/{name}",
        "description": desc,
        "language": "Python",
        "updated_at": updated_at,
        "topics": ["ai"],
    }


class TestDeterministicVectorIDs:
    def test_deterministic_id_generation(self):
        id1 = generate_chunk_id("github", "user/repo", "Sample chunk text")
        id2 = generate_chunk_id("github", "user/repo", "Sample chunk text")
        assert id1 == id2
        assert id1.startswith("github:user/repo:")

    def test_whitespace_normalization_in_ids(self):
        id1 = generate_chunk_id("github", "user/repo", "Sample   chunk\ntext  ")
        id2 = generate_chunk_id("github", "user/repo", "Sample chunk text")
        assert id1 == id2

    def test_different_repos_different_ids(self):
        id1 = generate_chunk_id("github", "user/repo1", "Sample chunk text")
        id2 = generate_chunk_id("github", "user/repo2", "Sample chunk text")
        assert id1 != id2

    def test_disambiguates_identical_chunks_in_same_doc(self):
        from langchain_core.documents import Document

        chunks = [
            Document(page_content="repeated content"),
            Document(page_content="repeated content"),
        ]
        ids = generate_deterministic_chunk_ids("github", "user/repo", chunks)
        assert len(ids) == 2
        assert ids[0] != ids[1]
        assert ids[1].endswith("_1")


class TestGitHubSynchronizationScenarios:
    # 1. First synchronization indexes GitHub repositories
    def test_1_first_sync_indexes_github_repos(self, mock_pinecone):
        mock_index, mock_vs = mock_pinecone
        mock_index.list.return_value = []

        repos = [_make_repo("repoA"), _make_repo("repoB")]
        fetch_res = RepoFetchResult(repos=repos, success=True, is_complete=True)

        with patch("app.rag_application.fetch_github_repos", return_value=fetch_res):
            with patch("app.rag_application.fetch_repo_readme", return_value="# Readme Content"):
                res = sync_github_to_pinecone("testuser")

        assert res["status"] == "success"
        assert res["discovered_count"] == 2
        assert res["added_count"] == 2
        assert res["updated_count"] == 0
        assert res["unchanged_count"] == 0
        assert res["deleted_count"] == 0
        assert res["vectors_upserted"] > 0
        assert mock_vs.add_documents.called

        # Verify deterministic IDs were passed
        call_kwargs = mock_vs.add_documents.call_args_list[0][1]
        ids_passed = call_kwargs["ids"]
        assert all(id_val.startswith("github:testuser/repoA:") for id_val in ids_passed)

    # 2. Repeated synchronization does not create duplicate vectors
    def test_2_repeated_sync_does_not_create_duplicate_vectors(self, mock_pinecone):
        mock_index, mock_vs = mock_pinecone
        repo = _make_repo("repoA")
        fetch_res = RepoFetchResult(repos=[repo], success=True, is_complete=True)

        # First, simulate that vectors already exist with their deterministic IDs
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        from app.github_loader import create_single_repo_document

        doc = create_single_repo_document("testuser", repo, readme_content="# Stable Readme")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents([doc])
        existing_ids = generate_deterministic_chunk_ids("github", "testuser/repoA", chunks)

        mock_index.list.return_value = [existing_ids]

        with patch("app.rag_application.fetch_github_repos", return_value=fetch_res):
            with patch("app.rag_application.fetch_repo_readme", return_value="# Stable Readme"):
                res = sync_github_to_pinecone("testuser")

        assert res["status"] == "success"
        assert res["discovered_count"] == 1
        assert res["unchanged_count"] == 1
        assert res["added_count"] == 0
        assert res["updated_count"] == 0
        assert res["deleted_count"] == 0
        assert res["vectors_upserted"] == 0
        # Should not call add_documents because content is unchanged
        assert not mock_vs.add_documents.called

    # 3. Updating a README replaces outdated content
    def test_3_updating_readme_replaces_outdated_content(self, mock_pinecone):
        mock_index, mock_vs = mock_pinecone
        repo = _make_repo("repoA")
        fetch_res = RepoFetchResult(repos=[repo], success=True, is_complete=True)

        # Existing old vector ID in Pinecone
        old_ids = ["github:testuser/repoA:old_outdated_chunk_hash"]
        mock_index.list.return_value = [old_ids]

        with patch("app.rag_application.fetch_github_repos", return_value=fetch_res):
            with patch(
                "app.rag_application.fetch_repo_readme", return_value="# Brand New Updated Readme"
            ):
                res = sync_github_to_pinecone("testuser")

        assert res["status"] == "success"
        assert res["updated_count"] == 1
        assert res["added_count"] == 0
        assert res["vectors_upserted"] > 0
        assert mock_vs.add_documents.called

        # Outdated chunk was deleted
        assert mock_index.delete.called
        delete_ids = mock_index.delete.call_args[1]["ids"]
        assert "github:testuser/repoA:old_outdated_chunk_hash" in delete_ids

    # 4. Adding a new repository indexes it
    def test_4_adding_new_repository_indexes_it(self, mock_pinecone):
        mock_index, mock_vs = mock_pinecone
        repoA = _make_repo("repoA")
        repoB = _make_repo("repoB")
        fetch_res = RepoFetchResult(repos=[repoA, repoB], success=True, is_complete=True)

        # Simulate repoA already existing in Pinecone
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        from app.github_loader import create_single_repo_document

        docA = create_single_repo_document("testuser", repoA, readme_content="# Readme A")
        chunksA = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents([docA])
        existing_ids_A = generate_deterministic_chunk_ids("github", "testuser/repoA", chunksA)

        mock_index.list.return_value = [existing_ids_A]

        def fake_readme(user, repo_name, token=None):
            return "# Readme A" if repo_name == "repoA" else "# Readme B"

        with patch("app.rag_application.fetch_github_repos", return_value=fetch_res):
            with patch("app.rag_application.fetch_repo_readme", side_effect=fake_readme):
                res = sync_github_to_pinecone("testuser")

        assert res["status"] == "success"
        assert res["discovered_count"] == 2
        assert res["unchanged_count"] == 1  # repoA
        assert res["added_count"] == 1  # repoB
        assert res["deleted_count"] == 0

        # Only repoB's chunks were upserted
        assert mock_vs.add_documents.call_count == 1
        added_ids = mock_vs.add_documents.call_args[1]["ids"]
        assert all(i.startswith("github:testuser/repoB:") for i in added_ids)

    # 5. Deleting a repository removes its old vectors
    def test_5_deleting_repository_removes_old_vectors(self, mock_pinecone):
        mock_index, mock_vs = mock_pinecone
        repoA = _make_repo("repoA")
        # Complete fetch returns ONLY repoA (repoB was deleted on GitHub)
        fetch_res = RepoFetchResult(repos=[repoA], success=True, is_complete=True)

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        from app.github_loader import create_single_repo_document

        docA = create_single_repo_document("testuser", repoA, readme_content="# Readme A")
        chunksA = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents([docA])
        existing_ids_A = generate_deterministic_chunk_ids("github", "testuser/repoA", chunksA)
        existing_ids_B = ["github:testuser/repoB:chunk1", "github:testuser/repoB:chunk2"]

        mock_index.list.return_value = [existing_ids_A + existing_ids_B]

        with patch("app.rag_application.fetch_github_repos", return_value=fetch_res):
            with patch("app.rag_application.fetch_repo_readme", return_value="# Readme A"):
                res = sync_github_to_pinecone("testuser")

        assert res["status"] == "success"
        assert res["discovered_count"] == 1
        assert res["unchanged_count"] == 1  # repoA
        assert res["deleted_count"] == 1  # repoB deleted
        assert mock_index.delete.called
        deleted_ids = mock_index.delete.call_args[1]["ids"]
        assert set(deleted_ids) == set(existing_ids_B)
        # repoA vectors were NOT deleted
        for a_id in existing_ids_A:
            assert a_id not in deleted_ids

    # 6. Portfolio vectors remain untouched
    def test_6_portfolio_vectors_remain_untouched(self, mock_pinecone):
        mock_index, mock_vs = mock_pinecone
        portfolio_ids = ["portfolio:home:chunk1", "portfolio:about:chunk2"]
        mock_index.list.return_value = [portfolio_ids]

        repoA = _make_repo("repoA")
        fetch_res = RepoFetchResult(repos=[repoA], success=True, is_complete=True)

        with patch("app.rag_application.fetch_github_repos", return_value=fetch_res):
            with patch("app.rag_application.fetch_repo_readme", return_value="# Readme"):
                res = sync_github_to_pinecone("testuser")

        assert res["status"] == "success"
        assert res["added_count"] == 1

        # Check all calls to delete (if any)
        for call in mock_index.delete.call_args_list:
            deleted = call[1].get("ids", [])
            for p_id in portfolio_ids:
                assert p_id not in deleted

    # 7. GitHub API failure does not delete existing vectors
    def test_7_github_api_failure_does_not_delete_existing_vectors(self, mock_pinecone):
        mock_index, mock_vs = mock_pinecone
        existing_ids = ["github:testuser/repoA:chunk1", "github:testuser/repoB:chunk2"]
        mock_index.list.return_value = [existing_ids]

        # API failure
        fetch_res = RepoFetchResult(
            repos=[],
            success=False,
            is_complete=False,
            error="GitHub API 500 Internal Server Error",
        )

        with patch("app.rag_application.fetch_github_repos", return_value=fetch_res):
            res = sync_github_to_pinecone("testuser")

        assert res["status"] == "error"
        assert "500" in res["errors"][0]
        # Deletion must NEVER be called
        assert not mock_index.delete.called
        assert not mock_vs.add_documents.called

    # 8. Partial repository fetch does not trigger deletion
    def test_8_partial_repository_fetch_does_not_trigger_deletion(self, mock_pinecone):
        mock_index, mock_vs = mock_pinecone
        repoA = _make_repo("repoA")
        # Partial fetch returns only repoA, but is_complete is False
        fetch_res = RepoFetchResult(repos=[repoA], success=True, is_complete=False)

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        from app.github_loader import create_single_repo_document

        docA = create_single_repo_document("testuser", repoA, readme_content="# Readme A")
        chunksA = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents([docA])
        existing_ids_A = generate_deterministic_chunk_ids("github", "testuser/repoA", chunksA)
        existing_ids_B = ["github:testuser/repoB:chunk1"]
        mock_index.list.return_value = [existing_ids_A + existing_ids_B]

        with patch("app.rag_application.fetch_github_repos", return_value=fetch_res):
            with patch("app.rag_application.fetch_repo_readme", return_value="# Readme A"):
                res = sync_github_to_pinecone("testuser")

        # repoB must NOT be deleted because is_complete was False
        assert res["deleted_count"] == 0
        assert not mock_index.delete.called

    # 9. Embedding or Pinecone write failure preserves existing data wherever possible
    def test_9_write_failure_preserves_existing_data(self, mock_pinecone):
        mock_index, mock_vs = mock_pinecone
        repoA = _make_repo("repoA")
        fetch_res = RepoFetchResult(repos=[repoA], success=True, is_complete=True)

        old_ids = ["github:testuser/repoA:old_chunk"]
        mock_index.list.return_value = [old_ids]

        # Upsert fails with connection error
        mock_vs.add_documents.side_effect = RuntimeError("Pinecone network timeout")

        with patch("app.rag_application.fetch_github_repos", return_value=fetch_res):
            with patch("app.rag_application.fetch_repo_readme", return_value="# New Readme"):
                res = sync_github_to_pinecone("testuser")

        assert len(res["errors"]) > 0
        assert "Pinecone network timeout" in res["errors"][0]
        # Crucial: old vectors must NOT have been deleted if upsert failed
        assert not mock_index.delete.called

    # 10. Empty README or missing README is handled gracefully
    def test_10_missing_readme_handled_gracefully(self, mock_pinecone):
        mock_index, mock_vs = mock_pinecone
        mock_index.list.return_value = []

        repo = _make_repo("no-readme-app", desc="A utility with no readme")
        fetch_res = RepoFetchResult(repos=[repo], success=True, is_complete=True)

        with patch("app.rag_application.fetch_github_repos", return_value=fetch_res):
            with patch("app.rag_application.fetch_repo_readme", return_value=None):
                res = sync_github_to_pinecone("testuser")

        assert res["status"] == "success"
        assert res["added_count"] == 1
        assert mock_vs.add_documents.called

        doc = mock_vs.add_documents.call_args[1]["documents"][0]
        assert "no-readme-app" in doc.page_content
        assert "A utility with no readme" in doc.page_content
        assert doc.metadata["type"] == "github_repository"
        assert doc.metadata["source_type"] == "github"

    # 11. Unchanged repositories are not unnecessarily re-indexed
    def test_11_unchanged_repositories_not_reindexed(self, mock_pinecone):
        mock_index, mock_vs = mock_pinecone

        repo1 = _make_repo("repo1")
        repo2 = _make_repo("repo2")
        fetch_res = RepoFetchResult(repos=[repo1, repo2], success=True, is_complete=True)

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        from app.github_loader import create_single_repo_document

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        doc1 = create_single_repo_document("testuser", repo1, readme_content="# Content 1")
        ids1 = generate_deterministic_chunk_ids(
            "github", "testuser/repo1", splitter.split_documents([doc1])
        )

        doc2 = create_single_repo_document("testuser", repo2, readme_content="# Content 2")
        ids2 = generate_deterministic_chunk_ids(
            "github", "testuser/repo2", splitter.split_documents([doc2])
        )

        mock_index.list.return_value = [ids1 + ids2]

        def fake_readme(user, repo_name, token=None):
            return "# Content 1" if repo_name == "repo1" else "# Content 2"

        with patch("app.rag_application.fetch_github_repos", return_value=fetch_res):
            with patch("app.rag_application.fetch_repo_readme", side_effect=fake_readme):
                res = sync_github_to_pinecone("testuser")

        assert res["status"] == "success"
        assert res["discovered_count"] == 2
        assert res["unchanged_count"] == 2
        assert res["added_count"] == 0
        assert res["updated_count"] == 0
        assert res["vectors_upserted"] == 0
        assert not mock_vs.add_documents.called
        assert not mock_index.delete.called
