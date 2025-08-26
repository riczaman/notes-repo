```
Test#!/usr/bin/env python3
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

def extract_repo_info(github_url):
    """Extract owner and repo name from GitHub URL"""
    # Handle different GitHub URL formats
    patterns = [
        r'github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$',
        r'github\.com/([^/]+)/([^/]+?)/.*',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, github_url)
        if match:
            return match.group(1), match.group(2)
    
    raise ValueError("Invalid GitHub URL format")

def get_github_files(owner, repo, branch_version):
    """Fetch file names from specified GitHub branch"""
    # GitHub API endpoint for repository contents
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
    
    # First, get all branches to find the one matching our version
    branches_url = f"https://api.github.com/repos/{owner}/{repo}/branches"
    
    try:
        # Get branches
        response = requests.get(branches_url)
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
        
        # Get files from the target branch
        params = {'ref': target_branch}
        response = requests.get(api_url, params=params)
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