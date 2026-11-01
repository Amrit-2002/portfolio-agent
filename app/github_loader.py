import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"
DEFAULT_GITHUB_USER = os.getenv("GITHUB_USERNAME", "Amrit-2002")

# In-memory cache for recent project summaries to respect GitHub API rate limits
_projects_cache: dict[str, Any] = {"timestamp": 0, "summary": "", "repos": []}
CACHE_TTL_SECONDS = 900  # 15 minutes


@dataclass
class RepoFetchResult:
    """
    Structured outcome of fetching GitHub repositories.
    Behaves as a sequence/list of repos for backward compatibility.
    """

    repos: list[dict[str, Any]]
    success: bool
    is_complete: bool
    error: str | None = None
    rate_limited: bool = False
    total_public_repos: int | None = None

    def __len__(self) -> int:
        return len(self.repos)

    def __iter__(self):
        return iter(self.repos)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.repos[index]

    def __bool__(self) -> bool:
        return bool(self.repos)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, list):
            return self.repos == other
        if isinstance(other, RepoFetchResult):
            return self.repos == other.repos and self.success == other.success
        return False


def _get_headers(token: str | None = None, raw: bool = False) -> dict[str, str]:
    headers = {
        "User-Agent": "portfolio-agent/1.0",
        "Accept": "application/vnd.github.raw+json" if raw else "application/vnd.github.v3+json",
    }
    auth_token = token or os.getenv("GITHUB_TOKEN")
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    return headers


def get_user_public_repo_count(username: str, token: str | None = None) -> int | None:
    """
    Fetch the public_repos count from the user profile endpoint to distinguish
    between genuinely empty accounts and faulty or empty list responses.
    """
    url = f"{GITHUB_API_BASE}/users/{username}"
    try:
        resp = requests.get(url, headers=_get_headers(token=token), timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict):
                return data.get("public_repos")
    except requests.RequestException:
        pass
    return None


def fetch_github_repos(
    username: str | None = None,
    token: str | None = None,
    include_forks: bool = False,
    max_repos: int | None = 50,
) -> RepoFetchResult:
    """
    Fetch public repositories for a GitHub user sorted by most recently updated.
    Handles pagination, rate limiting, and completeness tracking.
    """
    user = username or DEFAULT_GITHUB_USER
    total_public = get_user_public_repo_count(user, token=token)

    all_repos: list[dict[str, Any]] = []
    page = 1
    per_page = 100
    is_complete = True

    while True:
        url = f"{GITHUB_API_BASE}/users/{user}/repos"
        params: dict[str, Any] = {
            "sort": "updated",
            "direction": "desc",
            "per_page": per_page,
            "page": page,
        }

        try:
            resp = requests.get(
                url,
                headers=_get_headers(token=token),
                params=params,
                timeout=15,
            )

            if resp.status_code == 404:
                logger.warning("GitHub user '%s' not found.", user)
                return RepoFetchResult(
                    repos=[],
                    success=False,
                    is_complete=False,
                    error=f"GitHub user '{user}' not found.",
                    total_public_repos=total_public,
                )

            if resp.status_code in (403, 429):
                is_rate_limited = (
                    resp.headers.get("x-ratelimit-remaining") == "0"
                    or "rate limit" in resp.text.lower()
                )
                logger.warning("GitHub API rate limit or access error for user '%s'.", user)
                return RepoFetchResult(
                    repos=[],
                    success=False,
                    is_complete=False,
                    error="GitHub API rate limit exceeded or access forbidden.",
                    rate_limited=is_rate_limited,
                    total_public_repos=total_public,
                )

            resp.raise_for_status()
            page_repos = resp.json()

            if not isinstance(page_repos, list):
                logger.error(
                    "Unexpected GitHub response format for user '%s': %s", user, type(page_repos)
                )
                return RepoFetchResult(
                    repos=all_repos,
                    success=False,
                    is_complete=False,
                    error="Unexpected response format from GitHub API.",
                    total_public_repos=total_public,
                )

            if not page_repos:
                # Reached end of pagination
                break

            for repo in page_repos:
                if not include_forks and repo.get("fork"):
                    continue
                all_repos.append(repo)
                if max_repos and len(all_repos) >= max_repos:
                    break

            if max_repos and len(all_repos) >= max_repos:
                # If constrained by max_repos and more exist, mark as partial
                if len(page_repos) == per_page or (total_public and total_public > len(all_repos)):
                    is_complete = False
                break

            if len(page_repos) < per_page:
                # Reached last page
                break

            page += 1

        except requests.RequestException as e:
            logger.error("Failed to fetch GitHub repos for %s on page %d: %s", user, page, e)
            return RepoFetchResult(
                repos=all_repos,
                success=False,
                is_complete=False,
                error=f"Network error while fetching GitHub repos: {e}",
                total_public_repos=total_public,
            )

    return RepoFetchResult(
        repos=all_repos,
        success=True,
        is_complete=is_complete,
        total_public_repos=total_public,
    )


def fetch_repo_readme(
    username: str,
    repo_name: str,
    token: str | None = None,
    max_chars: int = 8000,
) -> str | None:
    """
    Fetch the README.md content of a specific repository.
    """
    url = f"{GITHUB_API_BASE}/repos/{username}/{repo_name}/readme"
    try:
        resp = requests.get(
            url,
            headers=_get_headers(token=token, raw=True),
            timeout=10,
        )
        if resp.status_code in (404, 204):
            return None
        resp.raise_for_status()
        text = resp.text.strip()
        if not text:
            return None
        # Truncate overly long READMEs to preserve context window
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n... [README truncated for length] ..."
        return text
    except requests.RequestException as e:
        logger.debug("Could not fetch README for %s/%s: %s", username, repo_name, e)
        return None


def create_single_repo_document(
    user: str,
    repo: dict[str, Any],
    readme_content: str | None = None,
) -> Document:
    """
    Build a LangChain Document for a single GitHub repository with rich metadata.
    """
    repo_name = repo.get("name") or "Unnamed"
    repo_full_name = repo.get("full_name") or f"{user}/{repo_name}"
    html_url = repo.get("html_url") or f"https://github.com/{repo_full_name}"
    desc = repo.get("description") or "No description provided."
    lang = repo.get("language") or "Unspecified"
    topics_raw = repo.get("topics", [])
    topics = ", ".join(topics_raw) if isinstance(topics_raw, list) else str(topics_raw)
    if not topics:
        topics = "None specified"
    stars = repo.get("stargazers_count", 0)
    forks = repo.get("forks_count", 0)
    homepage = repo.get("homepage") or "None"
    created_at = repo.get("created_at") or "Unknown"
    updated_at = repo.get("updated_at") or "Unknown"

    content_lines = [
        f"# Project: {repo_name}",
        f"- Repository: {repo_full_name}",
        f"- Repository URL: {html_url}",
        f"- Description: {desc}",
        f"- Primary Language: {lang}",
        f"- Topics / Tech Stack: {topics}",
        f"- Last Updated: {updated_at}",
        f"- Created At: {created_at}",
        f"- GitHub Stars: {stars} | Forks: {forks}",
        f"- Live Demo / Homepage: {homepage}",
    ]

    if readme_content:
        content_lines.extend(
            [
                "",
                "## Repository README & Technical Details:",
                readme_content,
            ]
        )

    full_content = "\n".join(content_lines)
    content_hash = hashlib.sha256(full_content.strip().encode("utf-8")).hexdigest()

    metadata = {
        "source": html_url,
        "url": html_url,
        "title": f"GitHub Project: {repo_name}",
        "type": "github_repository",
        "source_type": "github",
        "repo_name": repo_name,
        "repo_full_name": repo_full_name,
        "language": lang,
        "stars": stars,
        "forks": forks,
        "updated_at": updated_at,
        "content_hash": content_hash,
    }

    return Document(page_content=full_content, metadata=metadata)


def load_github_project_documents(
    username: str | None = None,
    token: str | None = None,
    include_forks: bool = False,
    fetch_readmes: bool = True,
    max_repos: int | None = 30,
) -> list[Document]:
    """
    Load GitHub repositories into LangChain Document objects with rich metadata
    and project README details for vector indexing.
    """
    user = username or DEFAULT_GITHUB_USER
    fetch_result = fetch_github_repos(
        username=user,
        token=token,
        include_forks=include_forks,
        max_repos=max_repos,
    )
    repos = (
        fetch_result.repos if isinstance(fetch_result, RepoFetchResult) else (fetch_result or [])
    )
    if not repos:
        logger.info("No GitHub repositories discovered for %s.", user)
        return []

    documents: list[Document] = []
    logger.info("Processing %d GitHub repositories for %s...", len(repos), user)

    for repo in repos:
        repo_name = repo.get("name") or "Unnamed"
        readme_content = None
        if fetch_readmes:
            readme_content = fetch_repo_readme(user, repo_name, token=token)

        doc = create_single_repo_document(user, repo, readme_content=readme_content)
        documents.append(doc)

    logger.info("Generated %d documents from GitHub repositories.", len(documents))
    return documents


def get_latest_projects_summary(
    username: str | None = None,
    token: str | None = None,
    limit: int = 5,
) -> str:
    """
    Return a quick cached Markdown bulleted summary of the user's most recently updated projects.
    """
    global _projects_cache
    now = time.time()
    user = username or DEFAULT_GITHUB_USER

    if _projects_cache["summary"] and (now - _projects_cache["timestamp"] < CACHE_TTL_SECONDS):
        return _projects_cache["summary"]

    fetch_result = fetch_github_repos(
        username=user, token=token, include_forks=False, max_repos=limit
    )
    repos = (
        fetch_result.repos if isinstance(fetch_result, RepoFetchResult) else (fetch_result or [])
    )
    if not repos:
        return "No recent public GitHub projects could be fetched at this time."

    lines = [f"### Most Recently Updated GitHub Projects for {user}:"]
    for r in repos[:limit]:
        name = r.get("name")
        url = r.get("html_url")
        desc = r.get("description") or "No description"
        lang = r.get("language") or "General"
        updated = (r.get("updated_at") or "")[:10]  # YYYY-MM-DD
        lines.append(f"- [{name}]({url}) ({lang}, updated {updated}): {desc}")

    summary = "\n".join(lines)
    _projects_cache = {"timestamp": now, "summary": summary, "repos": repos}
    return summary


if __name__ == "__main__":
    import json
    import sys

    # Set stdout encoding for Windows console if needed
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    print(f"Fetching GitHub repositories for {DEFAULT_GITHUB_USER}...")
    docs = load_github_project_documents(max_repos=5)
    print(f"Loaded {len(docs)} project documents.")
    for d in docs:
        print(f"\n--- {d.metadata['title']} ---")
        print("Metadata:", json.dumps(d.metadata, indent=2))
        preview = d.page_content[:250].encode("ascii", errors="replace").decode("ascii")
        print("Preview:", preview, "...\n")
