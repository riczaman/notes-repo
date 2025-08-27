```
#!/usr/bin/env python3
"""
GitHub Runbook Generator
Creates a formatted Word document runbook based on GitHub repository contents.
"""

import requests
import re
import os
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
import sys
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def extract_repo_info(github_url):
    """Extract owner and repo name from GitHub URL"""
    # Clean up the URL and extract components
    github_url = github_url.strip().rstrip('/')
    
    # Handle different GitHub URL formats
    pattern = r'github\.com/([^/]+)/([^/]+)'
    match = re.search(pattern, github_url)
    
    if match:
        owner = match.group(1)
        repo = match.group(2)
        # Clean up repo name (remove .git, etc.)
        repo = repo.split('.')[0]
        print(f"Debug: Extracted owner='{owner}', repo='{repo}' from URL: {github_url}")
        return owner, repo
    
    raise ValueError(f"Invalid GitHub URL format: {github_url}")

def get_github_files(owner, repo, branch_version):
    """Fetch file names from specified GitHub branch using GitHub API"""
    
    # Get GitHub token
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        print("Error: No GitHub token found. Please set GITHUB_TOKEN in your .env file")
        return [], "unknown"
    
    # Set up headers exactly like your working curl command
    headers = {
        'Accept': 'application/vnd.github.json',
        'Authorization': f'Bearer {github_token}',
        'X-GitHub-Api-Version': '2022-11-28'
    }
    
    print(f"Debug: Using Bearer token, length: {len(github_token)} characters")
    
    try:
        # First, get all branches
        branches_url = f"https://api.github.com/repos/{owner}/{repo}/branches"
        print(f"Debug: Fetching branches from: {branches_url}")
        
        response = requests.get(branches_url, headers=headers, timeout=30)
        print(f"Debug: Branches API response status: {response.status_code}")
        
        if response.status_code == 404:
            print("Error: Repository not found. Check:")
            print(f"1. Repository exists: https://github.com/{owner}/{repo}")
            print("2. Token has access to this repository")
            print("3. Repository path is correct")
            return [], "unknown"
        elif response.status_code == 403:
            print("Error: Access forbidden. Check:")
            print("1. Token has 'repo' scope")
            print("2. Token is authorized for the organization")
            print("3. Repository permissions")
            return [], "unknown"
        elif response.status_code != 200:
            print(f"Error: API request failed with status {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return [], "unknown"
        
        branches = response.json()
        print(f"Debug: Found {len(branches)} branches")
        
        # Find matching branch
        target_branch = None
        available_branches = [b['name'] for b in branches]
        print(f"Debug: Available branches: {available_branches[:10]}")
        
        # Try to find branch matching the version
        for branch in branches:
            if branch['name'] == branch_version or branch_version in branch['name']:
                target_branch = branch['name']
                break
        
        if not target_branch:
            target_branch = branch_version
            print(f"Warning: Branch '{branch_version}' not found, trying anyway...")
        else:
            print(f"Debug: Using branch: {target_branch}")
        
        # Get contents from the branch
        contents_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
        params = {'ref': target_branch}
        
        print(f"Debug: Fetching contents from branch: {target_branch}")
        response = requests.get(contents_url, params=params, headers=headers, timeout=30)
        print(f"Debug: Contents API response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error: Failed to get contents from branch '{target_branch}'")
            print(f"Response: {response.text[:200]}")
            return [], target_branch
        
        files_data = response.json()
        print(f"Debug: Found {len(files_data)} items in repository")
        
        # Process files
        file_names = set()
        for item in files_data:
            if item['type'] == 'file':
                filename = item['name']
                
                # Skip README files
                if filename.lower().startswith('readme'):
                    continue
                
                # Remove extension
                name_without_ext = filename.split('.')[0]
                if name_without_ext:
                    file_names.add(name_without_ext)
        
        print(f"Debug: Processed {len(file_names)} files: {sorted(list(file_names))}")
        return list(file_names), target_branch
        
    except requests.exceptions.RequestException as e:
        print(f"Error: Network request failed: {e}")
        return [], "unknown"
    except Exception as e:
        print(f"Error: {e}")
        return [], "unknown"

def create_runbook_document(repo_name, file_names):
    """Create a professionally formatted Word document"""
    doc = Document()
    
    # Document Title
    title = doc.add_heading('RUNBOOK', 0)
    title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    doc.add_paragraph()
    
    # Section 1: Scope
    doc.add_heading('1. Scope', level=1)
    doc.add_paragraph(
        'This runbook provides step-by-step procedures for deployment, configuration, '
        'and maintenance operations. It covers the complete workflow from initial setup '
        'through final deployment and includes rollback procedures for emergency situations.'
    )
    
    if file_names:
        doc.add_paragraph('Components included in this runbook:')
        for name in sorted(file_names):
            doc.add_paragraph(f'• {name}')
    doc.add_paragraph()
    
    # Section 2: Login
    doc.add_heading('2. Login', level=1)
    doc.add_paragraph('ssh test')
    doc.add_paragraph('sudo ss')
    doc.add_paragraph()
    
    # Section 3: Export
    doc.add_heading('3. Export', level=1)
    doc.add_paragraph('export test')
    doc.add_paragraph('export test2')
    doc.add_paragraph(f'export test/{repo_name}')
    doc.add_paragraph()
    
    # Section 4: Download
    doc.add_heading('4. Download', level=1)
    doc.add_paragraph('cd test')
    doc.add_paragraph('downlod')
    doc.add_paragraph()
    
    # Section 5: Upload
    doc.add_heading('5. Upload', level=1)
    doc.add_paragraph('upload.sh')
    doc.add_paragraph()
    
    # Section 6: Release
    doc.add_heading('6. Release', level=1)
    if file_names:
        for name in sorted(file_names):
            doc.add_paragraph(f'test.sh {name}')
    doc.add_paragraph()
    
    # Section 7: Rollback
    doc.add_heading('7. Rollback', level=1)
    doc.add_paragraph('cd test')
    doc.add_paragraph('export=1')
    doc.add_paragraph('cd test2')
    doc.add_paragraph()
    
    if file_names:
        # First list with roll.sh
        for name in sorted(file_names):
            doc.add_paragraph(f'roll.sh {name}')
        doc.add_paragraph()
        
        # Second list with st.sh
        for name in sorted(file_names):
            doc.add_paragraph(f'st.sh {name}')
    
    return doc

def main():
    parser = argparse.ArgumentParser(description='Generate runbook from GitHub repository')
    parser.add_argument('github_url', help='GitHub repository URL')
    parser.add_argument('branch_version', help='Branch version to process (e.g., 1.11.0)')
    parser.add_argument('-o', '--output', default='runbook.docx', help='Output filename (default: runbook.docx)')
    
    if len(sys.argv) == 1:
        print("GitHub Runbook Generator")
        print("Usage: python script.py <github_url> <branch_version> [-o output_file]")
        print()
        print("Example:")
        print("python script.py https://github.com/myorg/proj1 2.22.0")
        sys.exit(1)
    
    args = parser.parse_args()
    
    try:
        # Extract repository information
        owner, repo_name = extract_repo_info(args.github_url)
        print(f"Processing repository: {owner}/{repo_name}")
        
        # Get files from GitHub
        print(f"Fetching files from branch: {args.branch_version}")
        file_names, branch_used = get_github_files(owner, repo_name, args.branch_version)
        
        if file_names:
            print(f"Found {len(file_names)} files in branch '{branch_used}': {sorted(file_names)}")
        else:
            print(f"No files found in branch '{branch_used}' (excluding README files)")
        
        # Create the Word document
        print("Creating runbook document...")
        doc = create_runbook_document(repo_name, file_names)
        
        # Save the document
        doc.save(args.output)
        print(f"Runbook saved as: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```