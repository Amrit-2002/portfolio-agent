from unittest.mock import MagicMock, patch

from app.github_loader import (
    fetch_github_repos,
    fetch_repo_readme,
    get_latest_projects_summary,
    load_github_project_documents,
)


class TestFetchGithubRepos:
    def test_fetches_and_filters_forks(self):
        fake_repos = [
            {"name": "repo1", "fork": False, "updated_at": "2026-09-01T00:00:00Z"},
            {"name": "forked_repo", "fork": True, "updated_at": "2026-09-02T00:00:00Z"},
            {"name": "repo2", "fork": False, "updated_at": "2026-08-01T00:00:00Z"},
        ]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_repos
        mock_resp.raise_for_status = lambda: None

        with patch("app.github_loader.requests.get", return_value=mock_resp):
            results = fetch_github_repos(username="testuser", include_forks=False)

        assert len(results) == 2
        assert [r["name"] for r in results] == ["repo1", "repo2"]

    def test_handles_404_not_found(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch("app.github_loader.requests.get", return_value=mock_resp):
            results = fetch_github_repos(username="nonexistent_user")

        assert results == []

    def test_handles_request_exception(self):
        import requests

        with patch("app.github_loader.requests.get", side_effect=requests.RequestException("boom")):
            results = fetch_github_repos(username="anyuser")

        assert results == []


class TestFetchRepoReadme:
    def test_returns_readme_content(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "# Project Title\nThis is a test readme."
        mock_resp.raise_for_status = lambda: None

        with patch("app.github_loader.requests.get", return_value=mock_resp):
            content = fetch_repo_readme("user", "my-repo")

        assert content == "# Project Title\nThis is a test readme."

    def test_returns_none_on_404(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch("app.github_loader.requests.get", return_value=mock_resp):
            content = fetch_repo_readme("user", "no-readme-repo")

        assert content is None

    def test_truncates_long_readme(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "A" * 10000
        mock_resp.raise_for_status = lambda: None

        with patch("app.github_loader.requests.get", return_value=mock_resp):
            content = fetch_repo_readme("user", "long-repo", max_chars=100)

        assert len(content) < 200
        assert "truncated" in content.lower()


class TestLoadGithubProjectDocuments:
    def test_creates_langchain_documents_with_metadata(self):
        fake_repos = [
            {
                "name": "cool-app",
                "html_url": "https://github.com/testuser/cool-app",
                "description": "A very cool app",
                "language": "Python",
                "topics": ["ai", "fastapi"],
                "stargazers_count": 5,
                "forks_count": 1,
                "homepage": "https://coolapp.io",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-09-01T00:00:00Z",
                "fork": False,
            }
        ]

        with patch("app.github_loader.fetch_github_repos", return_value=fake_repos):
            with patch(
                "app.github_loader.fetch_repo_readme", return_value="## Details\nRuns with Docker."
            ):
                docs = load_github_project_documents(username="testuser")

        assert len(docs) == 1
        doc = docs[0]
        assert doc.metadata["repo_name"] == "cool-app"
        assert doc.metadata["language"] == "Python"
        assert doc.metadata["type"] == "github_repository"
        assert "A very cool app" in doc.page_content
        assert "Runs with Docker." in doc.page_content


class TestGetLatestProjectsSummary:
    def test_returns_bulleted_summary(self):
        fake_repos = [
            {
                "name": "fast-rag",
                "html_url": "https://github.com/user/fast-rag",
                "description": "High performance RAG",
                "language": "Python",
                "updated_at": "2026-10-01T12:00:00Z",
            }
        ]

        with patch("app.github_loader.fetch_github_repos", return_value=fake_repos):
            import app.github_loader as gl

            gl._projects_cache = {"timestamp": 0, "summary": "", "repos": []}
            summary = get_latest_projects_summary("user", limit=1)

        assert "fast-rag" in summary
        assert "Python" in summary
        assert "High performance RAG" in summary
