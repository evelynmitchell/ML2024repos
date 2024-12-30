#!/usr/bin/env python3
"""
Script to process GitHub repositories and extract their parent information.
This version uses the Repositories API instead of the Search API for better reliability.
"""

import os
import json
import time
import argparse
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import requests

class GitHubAPI:
    """Class to handle GitHub API interactions."""
    
    def __init__(self, token: str):
        """Initialize with GitHub token."""
        self.token = token
        self.headers = {'Authorization': f'Bearer {token}'}
        self.base_url = 'https://api.github.com'

    def _make_request(self, url: str, params: Optional[Dict] = None, max_retries: int = 3) -> Optional[Dict]:
        """Make a request to GitHub API with retries."""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error after {max_retries} attempts: {e}")
                    return None
                time.sleep(2)
        return None

    def get_repos_page(self, page: int = 1, per_page: int = 100) -> List[Dict]:
        """Get a page of repositories for a user."""
        url = f'{self.base_url}/users/evelynmitchell/repos'
        params = {
            'page': page,
            'per_page': per_page,
            'sort': 'created',
            'direction': 'desc'
        }
        
        result = self._make_request(url, params)
        if result:
            return result
        return []

    def get_repo_details(self, owner: str, repo: str) -> Optional[Dict]:
        """Get detailed information about a specific repository."""
        url = f'{self.base_url}/repos/{owner}/{repo}'
        return self._make_request(url)

    def get_total_repos(self) -> int:
        """Get total count of user's repositories."""
        url = f'{self.base_url}/users/evelynmitchell'
        result = self._make_request(url)
        if result:
            return result.get('public_repos', 0)
        return 0

class RepoProcessor:
    """Class to process repositories and update files."""
    
    def __init__(self, github_api: GitHubAPI, output_dir: str):
        """Initialize with GitHub API client and output directory."""
        self.github_api = github_api
        self.output_dir = output_dir
        self.json_file = os.path.join(output_dir, 'repos.json')
        self.md_file = os.path.join(output_dir, 'forked_repos.md')
        self.start_date = "2024-01-01T00:00:00Z"
        self.end_date = "2024-12-31T23:59:59Z"

    def is_from_2024(self, created_at: str) -> bool:
        """Check if a repository was created in 2024."""
        return self.start_date <= created_at <= self.end_date

    def process_repos_page(self, page: int, batch_size: Optional[int] = None) -> Tuple[List[Dict], bool]:
        """Process a page of repositories."""
        repos = self.github_api.get_repos_page(page)
        processed_repos = []
        found_2024_repos = False
        
        # Process only batch_size repos if specified
        if batch_size:
            repos = repos[:batch_size]
        
        for i, repo in enumerate(repos, 1):
            created_at = repo['created_at']
            
            # Skip if not from 2024
            if not self.is_from_2024(created_at):
                continue
                
            found_2024_repos = True
            print(f"\nProcessing {i}/{len(repos)}: {repo['name']}")
            
            # Get parent information if it's a fork
            parent_url = None
            if repo['fork']:
                repo_details = self.github_api.get_repo_details('evelynmitchell', repo['name'])
                if repo_details and repo_details.get('parent'):
                    parent_url = repo_details['parent']['html_url']
            
            repo_info = {
                "name": repo['name'],
                "createdAt": created_at,
                "description": repo['description'] or "",
                "url": repo['html_url'],
                "parent_url": parent_url,
                "labels": [],
                "repositoryTopics": None
            }
            processed_repos.append(repo_info)
            time.sleep(1)  # Rate limiting
            
        return processed_repos, found_2024_repos

    def update_files(self, repos: List[Dict]):
        """Update both JSON and Markdown files with repository information."""
        # Load existing repos if any
        existing_repos = []
        if os.path.exists(self.json_file):
            with open(self.json_file, 'r') as f:
                existing_repos = json.load(f)

        # Combine and sort repos
        all_repos = existing_repos + repos
        all_repos.sort(key=lambda x: x['createdAt'], reverse=True)

        # Update JSON file
        with open(self.json_file, 'w') as f:
            json.dump(all_repos, f, indent=2)

        # Update Markdown file
        table_content = ["# Forked Repositories from evelynmitchell\n\n"]
        table_content.append("| Name | Created At | Description | URL | Parent URL |\n")
        table_content.append("|------|------------|-------------|-----|------------|\n")

        for repo in all_repos:
            name = repo['name']
            created_at = repo['createdAt'].split('T')[0]
            description = repo['description']
            url = repo['url']
            parent_url = repo['parent_url'] or ''
            table_content.append(f"| {name} | {created_at} | {description} | {url} | {parent_url} |\n")

        with open(self.md_file, 'w') as f:
            f.writelines(table_content)

def main():
    """Main function to run the repository processing."""
    parser = argparse.ArgumentParser(description='Process GitHub repositories and extract parent information.')
    parser.add_argument('--page', type=int, default=1, help='Page number to process')
    parser.add_argument('--batch-size', type=int, default=None, help='Number of repos to process per batch')
    parser.add_argument('--output-dir', default='/workspace/ML2024repos', help='Output directory')
    
    args = parser.parse_args()
    
    # Get GitHub token from environment
    github_token = os.environ.get('GITHUB_TOKEN')
    if not github_token:
        print("Error: GITHUB_TOKEN environment variable not set")
        return

    # Initialize API client and processor
    github_api = GitHubAPI(github_token)
    processor = RepoProcessor(github_api, args.output_dir)
    
    # Get total repositories count
    total_repos = github_api.get_total_repos()
    print(f"Found {total_repos} total repositories")
    
    # Process repositories
    processed_repos, found_2024 = processor.process_repos_page(args.page, args.batch_size)
    
    if not found_2024:
        print("\nNo repositories from 2024 found on this page.")
        return
    
    # Update files
    processor.update_files(processed_repos)
    print(f"\nProcessed {len(processed_repos)} repositories")
    print(f"Files updated: {processor.json_file} and {processor.md_file}")

if __name__ == "__main__":
    main()