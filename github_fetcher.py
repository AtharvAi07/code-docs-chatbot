"""
Fetch code and documentation from GitHub repositories.
Uses GitHub REST API (public repos, no auth needed for reasonable usage).
"""

import requests
from typing import List, Dict
import base64
import time


class GitHubFetcher:
    """Fetch files from GitHub repositories."""
    
    def __init__(self, github_token: str = None):
        self.headers = {"Accept": "application/vnd.github.v3+json"}
        if github_token:
            self.headers["Authorization"] = f"token {github_token}"
    
    def fetch_repo(self, owner: str, repo: str, max_files: int = 30) -> List[Dict]:
        files = []
        
        root_contents = self._get_contents(owner, repo, "")
        if not root_contents:
            print(f"Could not fetch {owner}/{repo}. Check the repo name or rate limits.")
            return files
        
        priority_dirs = {"docs", "documentation", "src", "lib"}
        
        for item in root_contents:
            if len(files) >= max_files:
                break
            
            if item['type'] == 'file' and self._should_include(item['name']):
                content = self._fetch_file_content(item)
                if content:
                    files.append({
                        'name': item['name'],
                        'path': item['path'],
                        'content': content,
                        'type': self._detect_type(item['name']),
                        'url': item['html_url'],
                        'repo': f"{owner}/{repo}"
                    })
            
            elif item['type'] == 'dir' and item['name'].lower() in priority_dirs:
                sub_files = self._fetch_dir(owner, repo, item['path'], max_files - len(files))
                files.extend(sub_files)
        
        print(f"Fetched {len(files)} files from {owner}/{repo}")
        return files
    
    def _fetch_dir(self, owner: str, repo: str, path: str, limit: int) -> List[Dict]:
        files = []
        contents = self._get_contents(owner, repo, path)
        
        if not contents:
            return files
        
        for item in contents:
            if len(files) >= limit:
                break
            if item['type'] == 'file' and self._should_include(item['name']):
                content = self._fetch_file_content(item)
                if content:
                    files.append({
                        'name': item['name'],
                        'path': item['path'],
                        'content': content,
                        'type': self._detect_type(item['name']),
                        'url': item['html_url'],
                        'repo': f"{owner}/{repo}"
                    })
        
        return files
    
    def _get_contents(self, owner: str, repo: str, path: str) -> List[Dict]:
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 403:
                print("Rate limit hit. Wait a bit or add a GitHub token.")
                return []
            
            if response.status_code != 200:
                return []
            
            result = response.json()
            return result if isinstance(result, list) else [result]
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {path}: {e}")
            return []
    
    def _fetch_file_content(self, item: Dict) -> str:
        try:
            if "content" in item and item.get("encoding") == "base64":
                decoded = base64.b64decode(item["content"]).decode("utf-8", errors="ignore")
                return decoded
            
            response = requests.get(item["download_url"], timeout=10)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            print(f"Could not decode {item.get('name', 'file')}: {e}")
        
        return None
    
    def _should_include(self, filename: str) -> bool:
        include_extensions = {'.md', '.py', '.js', '.ts', '.rst', '.txt'}
        exclude_names = {'.gitignore', 'LICENSE', 'setup.py', '__init__.py'}
        
        if filename in exclude_names:
            return False
        
        return any(filename.endswith(ext) for ext in include_extensions)
    
    def _detect_type(self, filename: str) -> str:
        if filename.endswith(('.md', '.rst', '.txt')):
            return 'documentation'
        return 'code'


def fetch_multiple_repos(repo_list, max_files_per_repo: int = 20, github_token: str = None):
    fetcher = GitHubFetcher(github_token)
    all_files = []
    
    for owner, repo in repo_list:
        print(f"Fetching {owner}/{repo}...")
        files = fetcher.fetch_repo(owner, repo, max_files=max_files_per_repo)
        all_files.extend(files)
        time.sleep(1)
    
    print(f"Total files fetched across all repos: {len(all_files)}")
    return all_files


if __name__ == "__main__":
    print("=== Testing GitHub Fetcher ===")
    
    fetcher = GitHubFetcher()
    files = fetcher.fetch_repo("pallets", "flask", max_files=10)
    
    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  - {f['name']} ({f['type']}) - {len(f['content'])} chars")
