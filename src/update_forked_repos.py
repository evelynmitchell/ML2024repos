import json

# Load the JSON data
with open('/home/efm/git/myrepos/ML2024repos/repos.json', 'r') as json_file:
    repos_data = json.load(json_file)

# Load the markdown data
with open('/home/efm/git/myrepos/ML2024repos/forked_repos.md', 'r') as md_file:
    md_lines = md_file.readlines()

# Create a dictionary to map repo names to parent URLs
parent_urls = {repo['name']: repo.get('parent_url', '') for repo in repos_data}

# Update the markdown lines with the parent URLs
updated_md_lines = []
for line in md_lines:
    if line.startswith('| '):
        parts = line.split('|')
        repo_name = parts[1].strip()
        if repo_name in parent_urls:
            parts[4] = f' {parent_urls[repo_name]} '
        updated_md_lines.append('|'.join(parts))
    else:
        updated_md_lines.append(line)

# Write the updated markdown data back to the file
with open('/home/efm/git/myrepos/ML2024repos/forked_repos.md', 'w') as md_file:
    md_file.writelines(updated_md_lines)