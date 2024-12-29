#!/usr/bin/env python3
"""
Script to process GitHub repositories and extract their parent information.
This script fetches repository information from GitHub, extracts parent URLs for forked repos,
and updates both JSON and Markdown files with the information.
"""

import os
import json
import time
import argparse
from typing import Dict, List, Optional, Any
import requests
from datetime import datetime

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

    def get_forked_repos(self, username: str, start_date: str, end_date: str, 
                        page: int = 1, per_page: int = 100) -> List[Dict]:
        """Get forked repositories for a user within a date range."""
        url = f'{self.base_url}/search/repositories'
        params = {
            'q': f'user:{username} fork:true created:{start_date}..{end_date}',
            'sort': 'updated',
            'per_page': per_page,
            'page': page
        }
        
        result = self._make_request(url, params)
        if result and 'items' in result:
            return result['items']
        return []

    def get_repo_details(self, owner: str, repo: str) -> Optional[Dict]:
        """Get detailed information about a specific repository."""
        url = f'{self.base_url}/repos/{owner}/{repo}'
        return self._make_request(url)

    def get_total_repos(self, username: str, start_date: str, end_date: str) -> int:
        """Get total count of repositories matching the criteria."""
        url = f'{self.base_url}/search/repositories'
        params = {
            'q': f'user:{username} fork:true created:{start_date}..{end_date}',
            'per_page': 1
        }
        
        result = self._make_request(url, params)
        if result:
            return result['total_count']
        return 0

class RepoProcessor:
    """Class to process repositories and update files."""
    
    def __init__(self, github_api: GitHubAPI, output_dir: str):
        """Initialize with GitHub API client and output directory."""
        self.github_api = github_api
        self.output_dir = output_dir
        self.json_file = os.path.join(output_dir, 'repos.json')
        self.md_file = os.path.join(output_dir, 'forked_repos.md')

    def process_repos(self, username: str, start_date: str, end_date: str, 
                     page: int, batch_size: int = 20) -> List[Dict]:
        """Process a batch of repositories."""
        repos = self.github_api.get_forked_repos(username, start_date, end_date, page)
        processed_repos = []
        
        # Process only batch_size repos at a time
        for i, repo in enumerate(repos[:batch_size], 1):
            print(f"\nProcessing {i}/{batch_size}: {repo['name']}")
            
            repo_details = self.github_api.get_repo_details(username, repo['name'])
            
            repo_info = {
                "name": repo['name'],
                "createdAt": repo['created_at'],
                "description": repo['description'] or "",
                "url": repo['html_url'],
                "parent_url": repo_details.get('parent', {}).get('html_url') if repo_details else None,
                "labels": [],
                "repositoryTopics": None
            }
            processed_repos.append(repo_info)
            time.sleep(1)  # Rate limiting
            
        return processed_repos

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
    parser.add_argument('--username', default='evelynmitchell', help='GitHub username')
    parser.add_argument('--start-date', default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--page', type=int, default=1, help='Page number to process')
    parser.add_argument('--batch-size', type=int, default=20, help='Number of repos to process per batch')
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
    total_repos = github_api.get_total_repos(args.username, args.start_date, args.end_date)
    print(f"Found {total_repos} total repositories")
    
    # Process repositories
    processed_repos = processor.process_repos(
        args.username, args.start_date, args.end_date, args.page, args.batch_size
    )
    
    # Update files
    processor.update_files(processed_repos)
    print(f"\nProcessed {len(processed_repos)} repositories")
    print(f"Files updated: {processor.json_file} and {processor.md_file}")

if __name__ == "__main__":
    main()