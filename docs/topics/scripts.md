```python
#!/usr/bin/env python3
"""
GitHub Runbook Generator
Creates a formatted Word document runbook based on GitHub repository contents.
"""

import requests
import re
from urllib.parse import urlparse
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
import sys
import argparse
import urllib3
import os
from dotenv import load_dotenv

# Disable SSL warnings (use only if you're okay with less secure connections)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
load_dotenv()

def extract_repo_info(github_url):
    """Extract owner and repo name from GitHub URL"""
    # Handle different GitHub URL formats
    patterns = [
        r'github\.com/([^/]+)/([^/]+?)(?:\.git)?/?

def get_github_files(owner, repo, branch_version):
    """Fetch file names from specified GitHub branch"""
    # GitHub API endpoint for repository contents
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
    
    # First, get all branches to find the one matching our version
    branches_url = f"https://api.github.com/repos/{owner}/{repo}/branches"
    
    print(f"Debug: Trying to fetch branches from: {branches_url}")
    
    try:
        # Get GitHub token from environment variable
        github_token = os.getenv('GITHUB_TOKEN')
        headers = {}
        if github_token:
            # Use 'token' for classic tokens, 'Bearer' for fine-grained tokens
            if github_token.startswith('github_pat_'):
                headers['Authorization'] = f'Bearer {github_token}'
                print("Debug: Using fine-grained token (Bearer)")
            else:
                headers['Authorization'] = f'token {github_token}'
                print("Debug: Using classic token")
            print(f"Debug: Token length: {len(github_token)} characters")
        else:
            print("Debug: No GitHub token found - only public repos will work")

        print(f"Debug: Making request to GitHub API...")
        print(f"Debug: URL being requested: {branches_url}")
        
        # First test basic repo access
        repo_test_url = f"https://api.github.com/repos/{owner}/{repo}"
        print(f"Debug: Testing basic repo access: {repo_test_url}")
        test_response = requests.get(repo_test_url, headers=headers, timeout=30, verify=False)
        print(f"Debug: Basic repo access status: {test_response.status_code}")
        
        if test_response.status_code == 200:
            print("Debug: Basic repo access successful")
            repo_data = test_response.json()
            print(f"Debug: Repo name: {repo_data.get('name', 'unknown')}")
            print(f"Debug: Repo private: {repo_data.get('private', 'unknown')}")
            print(f"Debug: Default branch: {repo_data.get('default_branch', 'unknown')}")
        elif test_response.status_code == 404:
            print("Debug: Repository not found - check if the repository path is correct")
            print(f"Debug: Make sure you can access: https://github.com/{owner}/{repo}")
            print("Debug: Response body:", test_response.text[:200])
            return [], "unknown"
        elif test_response.status_code == 403:
            print("Debug: Access forbidden - token may not have proper permissions")
            print("Debug: Response body:", test_response.text[:200])
            return [], "unknown"
        else:
            print(f"Debug: Unexpected status code: {test_response.status_code}")
            print("Debug: Response body:", test_response.text[:200])
        
        # Now try to get branches
        response = requests.get(branches_url, headers=headers, timeout=30, verify=False)
        print(f"Debug: Branches request status code: {response.status_code}")
        
        if response.status_code != 200:
            print("Debug: Branches request failed")
            print("Debug: Response body:", response.text[:500])
            print("Debug: Response headers:", dict(response.headers))
            
            # Try alternative approach - get branches from repo info
            if test_response.status_code == 200:
                print("Debug: Trying to work with default branch only...")
                repo_data = test_response.json()
                default_branch = repo_data.get('default_branch', 'main')
                print(f"Debug: Using default branch: {default_branch}")
                
                # Try to get contents from default branch
                default_api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
                params = {'ref': default_branch}
                contents_response = requests.get(default_api_url, params=params, headers=headers, timeout=30, verify=False)
                
                if contents_response.status_code == 200:
                    print("Debug: Successfully got contents from default branch")
                    files_data = contents_response.json()
                    
                    # Extract file names, remove extensions, and filter
                    file_names = set()
                    for item in files_data:
                        if item['type'] == 'file':
                            filename = item['name']
                            # Skip README files
                            if filename.lower().startswith('readme'):
                                continue
                            
                            # Remove extension
                            name_without_ext = filename.split('.')[0]
                            if name_without_ext:  # Only add non-empty names
                                file_names.add(name_without_ext)
                    
                    return list(file_names), default_branch
                else:
                    print(f"Debug: Contents request also failed: {contents_response.status_code}")
                    print("Debug: Contents response:", contents_response.text[:200])
            
            return [], "unknown"
        
        if response.status_code == 404:
            print("Debug: 404 error - Repository not found. Possible causes:")
            print("1. Repository doesn't exist")
            print("2. Repository is private (requires authentication)")
            print("3. Repository name is incorrect")
            print(f"Debug: Please verify the repository exists at: https://github.com/{owner}/{repo}")
        elif response.status_code == 403:
            print("Debug: 403 Forbidden error. Possible causes:")
            print("1. GitHub token doesn't have 'repo' scope")
            print("2. Token is expired or invalid")
            print("3. Rate limit exceeded")
            print("4. Repository access denied")
            if 'X-RateLimit-Remaining' in response.headers:
                print(f"Debug: Rate limit remaining: {response.headers['X-RateLimit-Remaining']}")
        
        response.raise_for_status()
        branches = response.json()
        
        # Find branch by version (look for exact match or containing the version)
        target_branch = None
        for branch in branches:
            # Try exact match first
            if branch['name'] == branch_version:
                target_branch = branch['name']
                break
            # Then try if version is contained in branch name
            elif branch_version in branch['name']:
                target_branch = branch['name']
                break
        
        if not target_branch:
            # If no branch found with the version, try using the version directly
            target_branch = branch_version
        
        # Get files from the target branch - add timeout and SSL handling
        params = {'ref': target_branch}
        response = requests.get(api_url, params=params, headers=headers, timeout=30, verify=False)
        response.raise_for_status()
        
        files_data = response.json()
        
        # Extract file names, remove extensions, and filter
        file_names = set()
        for item in files_data:
            if item['type'] == 'file':
                filename = item['name']
                # Skip README files
                if filename.lower().startswith('readme'):
                    continue
                
                # Remove extension
                name_without_ext = filename.split('.')[0]
                if name_without_ext:  # Only add non-empty names
                    file_names.add(name_without_ext)
        
        return list(file_names), target_branch
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching from GitHub API: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing GitHub data: {e}")
        sys.exit(1)

def create_runbook_document(repo_name, file_names):
    """Create a professionally formatted Word document"""
    doc = Document()
    
    # Set up document styles
    styles = doc.styles
    
    # Title style
    title_style = styles.add_style('CustomTitle', WD_STYLE_TYPE.PARAGRAPH)
    title_style.font.size = Inches(0.25)
    title_style.font.bold = True
    title_style.font.name = 'Arial'
    
    # Section heading style
    heading_style = styles.add_style('CustomHeading', WD_STYLE_TYPE.PARAGRAPH)
    heading_style.font.size = Inches(0.18)
    heading_style.font.bold = True
    heading_style.font.name = 'Arial'
    
    # Normal text style
    normal_style = styles.add_style('CustomNormal', WD_STYLE_TYPE.PARAGRAPH)
    normal_style.font.size = Inches(0.14)
    normal_style.font.name = 'Arial'
    
    # Document Title
    title = doc.add_paragraph('RUNBOOK', style='CustomTitle')
    title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    doc.add_paragraph()  # Add space
    
    # Section 1: Scope
    scope_heading = doc.add_paragraph('1. Scope', style='CustomHeading')
    scope_desc = doc.add_paragraph(
        'This runbook provides step-by-step procedures for deployment, configuration, '
        'and maintenance operations. It covers the complete workflow from initial setup '
        'through final deployment and includes rollback procedures for emergency situations.',
        style='CustomNormal'
    )
    doc.add_paragraph()
    
    # Add file names list
    if file_names:
        doc.add_paragraph('Components included in this runbook:', style='CustomNormal')
        for name in sorted(file_names):
            doc.add_paragraph(f'• {name}', style='CustomNormal')
    doc.add_paragraph()
    
    # Section 2: Login
    login_heading = doc.add_paragraph('2. Login', style='CustomHeading')
    doc.add_paragraph('ssh test', style='CustomNormal')
    doc.add_paragraph('sudo ss', style='CustomNormal')
    doc.add_paragraph()
    
    # Section 3: Export
    export_heading = doc.add_paragraph('3. Export', style='CustomHeading')
    doc.add_paragraph('export test', style='CustomNormal')
    doc.add_paragraph('export test2', style='CustomNormal')
    doc.add_paragraph(f'export test/{repo_name}', style='CustomNormal')
    doc.add_paragraph()
    
    # Section 4: Download
    download_heading = doc.add_paragraph('4. Download', style='CustomHeading')
    doc.add_paragraph('cd test', style='CustomNormal')
    doc.add_paragraph('downlod', style='CustomNormal')  # Keeping the typo as specified
    doc.add_paragraph()
    
    # Section 5: Upload
    upload_heading = doc.add_paragraph('5. Upload', style='CustomHeading')
    doc.add_paragraph('upload.sh', style='CustomNormal')
    doc.add_paragraph()
    
    # Section 6: Release
    release_heading = doc.add_paragraph('6. Release', style='CustomHeading')
    if file_names:
        for name in sorted(file_names):
            doc.add_paragraph(f'test.sh {name}', style='CustomNormal')
    doc.add_paragraph()
    
    # Section 7: Rollback
    rollback_heading = doc.add_paragraph('7. Rollback', style='CustomHeading')
    doc.add_paragraph('cd test', style='CustomNormal')
    doc.add_paragraph('export=1', style='CustomNormal')
    doc.add_paragraph('cd test2', style='CustomNormal')
    doc.add_paragraph()
    
    if file_names:
        # First list with roll.sh
        for name in sorted(file_names):
            doc.add_paragraph(f'roll.sh {name}', style='CustomNormal')
        doc.add_paragraph()
        
        # Second list with st.sh
        for name in sorted(file_names):
            doc.add_paragraph(f'st.sh {name}', style='CustomNormal')
    
    return doc

def main():
    # OPTION 1: Use command line arguments (recommended)
    parser = argparse.ArgumentParser(description='Generate runbook from GitHub repository')
    parser.add_argument('github_url', help='GitHub repository URL')
    parser.add_argument('branch_version', help='Branch version to process (e.g., 1.11.0)')
    parser.add_argument('-o', '--output', default='runbook.docx', help='Output filename (default: runbook.docx)')
    
    args = parser.parse_args()
    
    # OPTION 2: Hardcode your values here (uncomment and modify these lines, then comment out the argparse section above)
    # class Args:
    #     def __init__(self):
    #         self.github_url = "https://github.com/your-username/your-repo"  # PUT YOUR GITHUB URL HERE
    #         self.branch_version = "1.11.0"  # PUT YOUR BRANCH VERSION HERE
    #         self.output = "runbook.docx"
    # args = Args()
    
    try:
        # Extract repository information
        owner, repo_name = extract_repo_info(args.github_url)
        print(f"Processing repository: {owner}/{repo_name}")
        
        # Get files from GitHub
        print(f"Fetching files from branch {args.branch_version}...")
        file_names, branch_used = get_github_files(owner, repo_name, args.branch_version)
        print(f"Found {len(file_names)} files in branch '{branch_used}'")
        
        if file_names:
            print("Files found:", ", ".join(sorted(file_names)))
        else:
            print("No files found (excluding README files)")
        
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
    # If running without command line arguments, show usage
    if len(sys.argv) == 1:
        print("GitHub Runbook Generator")
        print("Usage: python script.py <github_url> <branch_version> [-o output_file]")
        print()
        print("Example:")
        print("python script.py https://github.com/user/repo 1.11.0")
        print("python script.py https://github.com/user/repo v2.5.3 -o my_runbook.docx")
        sys.exit(1)
    
    main()
,
        r'github\.com/([^/]+)/([^/]+?)/.*',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, github_url)
        if match:
            owner = match.group(1)
            repo = match.group(2)
            print(f"Debug: Extracted owner='{owner}', repo='{repo}' from URL: {github_url}")
            return owner, repo
    
    raise ValueError("Invalid GitHub URL format")

def get_github_files(owner, repo, branch_version):
    """Fetch file names from specified GitHub branch"""
    # GitHub API endpoint for repository contents
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
    
    # First, get all branches to find the one matching our version
    branches_url = f"https://api.github.com/repos/{owner}/{repo}/branches"
    
    try:
        # Get branches - add timeout and SSL handling
        response = requests.get(branches_url, timeout=30, verify=False)
        response.raise_for_status()
        branches = response.json()
        
        # Find branch by version (look for exact match or containing the version)
        target_branch = None
        for branch in branches:
            # Try exact match first
            if branch['name'] == branch_version:
                target_branch = branch['name']
                break
            # Then try if version is contained in branch name
            elif branch_version in branch['name']:
                target_branch = branch['name']
                break
        
        if not target_branch:
            # If no branch found with the version, try using the version directly
            target_branch = branch_version
        
        # Get files from the target branch - add timeout and SSL handling
        params = {'ref': target_branch}
        response = requests.get(api_url, params=params, timeout=30, verify=False)
        response.raise_for_status()
        
        files_data = response.json()
        
        # Extract file names, remove extensions, and filter
        file_names = set()
        for item in files_data:
            if item['type'] == 'file':
                filename = item['name']
                # Skip README files
                if filename.lower().startswith('readme'):
                    continue
                
                # Remove extension
                name_without_ext = filename.split('.')[0]
                if name_without_ext:  # Only add non-empty names
                    file_names.add(name_without_ext)
        
        return list(file_names), target_branch
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching from GitHub API: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing GitHub data: {e}")
        sys.exit(1)

def create_runbook_document(repo_name, file_names):
    """Create a professionally formatted Word document"""
    doc = Document()
    
    # Set up document styles
    styles = doc.styles
    
    # Title style
    title_style = styles.add_style('CustomTitle', WD_STYLE_TYPE.PARAGRAPH)
    title_style.font.size = Inches(0.25)
    title_style.font.bold = True
    title_style.font.name = 'Arial'
    
    # Section heading style
    heading_style = styles.add_style('CustomHeading', WD_STYLE_TYPE.PARAGRAPH)
    heading_style.font.size = Inches(0.18)
    heading_style.font.bold = True
    heading_style.font.name = 'Arial'
    
    # Normal text style
    normal_style = styles.add_style('CustomNormal', WD_STYLE_TYPE.PARAGRAPH)
    normal_style.font.size = Inches(0.14)
    normal_style.font.name = 'Arial'
    
    # Document Title
    title = doc.add_paragraph('RUNBOOK', style='CustomTitle')
    title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    doc.add_paragraph()  # Add space
    
    # Section 1: Scope
    scope_heading = doc.add_paragraph('1. Scope', style='CustomHeading')
    scope_desc = doc.add_paragraph(
        'This runbook provides step-by-step procedures for deployment, configuration, '
        'and maintenance operations. It covers the complete workflow from initial setup '
        'through final deployment and includes rollback procedures for emergency situations.',
        style='CustomNormal'
    )
    doc.add_paragraph()
    
    # Add file names list
    if file_names:
        doc.add_paragraph('Components included in this runbook:', style='CustomNormal')
        for name in sorted(file_names):
            doc.add_paragraph(f'• {name}', style='CustomNormal')
    doc.add_paragraph()
    
    # Section 2: Login
    login_heading = doc.add_paragraph('2. Login', style='CustomHeading')
    doc.add_paragraph('ssh test', style='CustomNormal')
    doc.add_paragraph('sudo ss', style='CustomNormal')
    doc.add_paragraph()
    
    # Section 3: Export
    export_heading = doc.add_paragraph('3. Export', style='CustomHeading')
    doc.add_paragraph('export test', style='CustomNormal')
    doc.add_paragraph('export test2', style='CustomNormal')
    doc.add_paragraph(f'export test/{repo_name}', style='CustomNormal')
    doc.add_paragraph()
    
    # Section 4: Download
    download_heading = doc.add_paragraph('4. Download', style='CustomHeading')
    doc.add_paragraph('cd test', style='CustomNormal')
    doc.add_paragraph('downlod', style='CustomNormal')  # Keeping the typo as specified
    doc.add_paragraph()
    
    # Section 5: Upload
    upload_heading = doc.add_paragraph('5. Upload', style='CustomHeading')
    doc.add_paragraph('upload.sh', style='CustomNormal')
    doc.add_paragraph()
    
    # Section 6: Release
    release_heading = doc.add_paragraph('6. Release', style='CustomHeading')
    if file_names:
        for name in sorted(file_names):
            doc.add_paragraph(f'test.sh {name}', style='CustomNormal')
    doc.add_paragraph()
    
    # Section 7: Rollback
    rollback_heading = doc.add_paragraph('7. Rollback', style='CustomHeading')
    doc.add_paragraph('cd test', style='CustomNormal')
    doc.add_paragraph('export=1', style='CustomNormal')
    doc.add_paragraph('cd test2', style='CustomNormal')
    doc.add_paragraph()
    
    if file_names:
        # First list with roll.sh
        for name in sorted(file_names):
            doc.add_paragraph(f'roll.sh {name}', style='CustomNormal')
        doc.add_paragraph()
        
        # Second list with st.sh
        for name in sorted(file_names):
            doc.add_paragraph(f'st.sh {name}', style='CustomNormal')
    
    return doc

def main():
    # OPTION 1: Use command line arguments (recommended)
    parser = argparse.ArgumentParser(description='Generate runbook from GitHub repository')
    parser.add_argument('github_url', help='GitHub repository URL')
    parser.add_argument('branch_version', help='Branch version to process (e.g., 1.11.0)')
    parser.add_argument('-o', '--output', default='runbook.docx', help='Output filename (default: runbook.docx)')
    
    args = parser.parse_args()
    
    # OPTION 2: Hardcode your values here (uncomment and modify these lines, then comment out the argparse section above)
    # class Args:
    #     def __init__(self):
    #         self.github_url = "https://github.com/your-username/your-repo"  # PUT YOUR GITHUB URL HERE
    #         self.branch_version = "1.11.0"  # PUT YOUR BRANCH VERSION HERE
    #         self.output = "runbook.docx"
    # args = Args()
    
    try:
        # Extract repository information
        owner, repo_name = extract_repo_info(args.github_url)
        print(f"Processing repository: {owner}/{repo_name}")
        
        # Get files from GitHub
        print(f"Fetching files from branch {args.branch_version}...")
        file_names, branch_used = get_github_files(owner, repo_name, args.branch_version)
        print(f"Found {len(file_names)} files in branch '{branch_used}'")
        
        if file_names:
            print("Files found:", ", ".join(sorted(file_names)))
        else:
            print("No files found (excluding README files)")
        
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
    # If running without command line arguments, show usage
    if len(sys.argv) == 1:
        print("GitHub Runbook Generator")
        print("Usage: python script.py <github_url> <branch_version> [-o output_file]")
        print()
        print("Example:")
        print("python script.py https://github.com/user/repo 1.11.0")
        print("python script.py https://github.com/user/repo v2.5.3 -o my_runbook.docx")
        sys.exit(1)
    
    main()
```