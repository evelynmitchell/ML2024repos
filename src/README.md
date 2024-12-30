# Repository Processing Scripts

This directory contains scripts for processing GitHub repositories and extracting their parent information.

## Scripts

### `process_repos_v2.py` (Recommended)

This is the updated version of the script that uses the GitHub Repositories API instead of the Search API. This version:
- Is more reliable as it doesn't depend on GitHub's search functionality
- Can handle a larger number of repositories
- Has better rate limiting and error handling
- Provides more accurate repository counts

### `process_repos.py` (Legacy)

This script fetches repository information from GitHub, extracts parent URLs for forked repos, and updates both JSON and Markdown files with the information.

### Prerequisites

- Python 3.6+
- Required packages: `requests`
- GitHub API token set in environment variable `GITHUB_TOKEN`

### Usage

Basic usage:
```bash
python3 process_repos.py
```

This will process repositories with default settings:
- Username: evelynmitchell
- Date range: 2024-01-01 to 2024-12-31
- Page: 1
- Batch size: 20 repositories
- Output directory: /workspace/ML2024repos

Custom usage:
```bash
python3 process_repos.py \
    --username evelynmitchell \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --page 1 \
    --batch-size 20 \
    --output-dir /workspace/ML2024repos
```

### Arguments

- `--username`: GitHub username (default: evelynmitchell)
- `--start-date`: Start date in YYYY-MM-DD format (default: 2024-01-01)
- `--end-date`: End date in YYYY-MM-DD format (default: 2024-12-31)
- `--page`: Page number to process (default: 1)
- `--batch-size`: Number of repositories to process per batch (default: 20)
- `--output-dir`: Output directory for JSON and Markdown files (default: /workspace/ML2024repos)

### Output Files

The script updates two files:
1. `repos.json`: Contains detailed repository information in JSON format
2. `forked_repos.md`: Contains a markdown table with repository information

### Rate Limiting

The script includes built-in rate limiting (1 second delay between requests) to avoid hitting GitHub API limits. It also includes retry logic for failed requests.

### Example

To process the second page of repositories with a larger batch size:
```bash
python3 process_repos.py --page 2 --batch-size 50
```