```
import requests
import json
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
from typing import Dict, List, Any
import base64
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.chart import BarChart, Reference

class TeamCityBuildAnalyzer:
    def __init__(self, teamcity_url: str, username: str, password: str, project_filters: List[str] = None, 
                 build_type_filters: List[str] = None, build_name_filters: List[str] = None):
        """
        Initialize TeamCity connection with optional filters
        
        Args:
            teamcity_url: TeamCity server URL (e.g., 'https://teamcity.company.com')
            username: TeamCity username
            password: TeamCity password
            project_filters: List of project names/IDs to include (case-insensitive partial matching)
            build_type_filters: List of build type names/IDs to include (case-insensitive partial matching)
            build_name_filters: List of build names to include (case-insensitive partial matching)
        """
        self.base_url = teamcity_url.rstrip('/')
        self.api_url = f"{self.base_url}/app/rest"
        
        # Setup authentication
        auth_string = f"{username}:{password}"
        auth_bytes = auth_string.encode('ascii')
        auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
        
        self.headers = {
            'Authorization': f'Basic {auth_b64}',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        # Store filters (convert to lowercase for case-insensitive matching)
        self.project_filters = [f.lower() for f in project_filters] if project_filters else []
        self.build_type_filters = [f.lower() for f in build_type_filters] if build_type_filters else []
        self.build_name_filters = [f.lower() for f in build_name_filters] if build_name_filters else []
        
        # Store original filters for display purposes
        self.original_project_filters = project_filters if project_filters else []
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test the connection to TeamCity server"""
        try:
            print("Testing connection to TeamCity...")
            response = requests.get(f"{self.api_url}/server", headers=self.headers, timeout=10)
            if response.status_code == 401:
                raise Exception("Authentication failed. Please check your username and password.")
            elif response.status_code == 403:
                raise Exception("Access forbidden. Please check your permissions.")
            elif response.status_code != 200:
                raise Exception(f"Connection failed with status code: {response.status_code}")
            
            server_info = response.json()
            print(f"Connected to TeamCity {server_info.get('version', 'Unknown')} successfully!")
            
        except requests.exceptions.Timeout:
            raise Exception("Connection timeout. Please check your TeamCity URL and network connection.")
        except requests.exceptions.ConnectionError:
            raise Exception("Connection error. Please check your TeamCity URL and network connection.")
        except Exception as e:
            raise Exception(f"Connection test failed: {str(e)}")
    
    def get_project_hierarchy(self, project_id: str = None) -> Dict[str, Any]:
        """
        Get the full project hierarchy to understand folder structure
        
        Args:
            project_id: Project ID to get hierarchy for (None for root)
            
        Returns:
            Project hierarchy dictionary
        """
        try:
            url = f"{self.api_url}/projects"
            params = {
                'fields': 'project(id,name,parentProjectId,parentProject(id,name))',
                'locator': f'id:{project_id}' if project_id else 'archived:false'
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not fetch project hierarchy: {e}")
            return {}
    
    def _get_project_full_path(self, project_id: str, project_name: str) -> str:
        """
        Get the full path of a project including parent folders
        
        Args:
            project_id: Project ID
            project_name: Project name
            
        Returns:
            Full project path (e.g., "Veritas Release Projects > PAT - Release Builds")
        """
        try:
            url = f"{self.api_url}/projects/id:{project_id}"
            params = {
                'fields': 'project(id,name,parentProject(id,name,parentProject(id,name,parentProject(id,name))))'
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            project_data = response.json()
            
            # Build path from root to current project
            path_parts = []
            current_project = project_data
            
            while current_project:
                path_parts.insert(0, current_project.get('name', 'Unknown'))
                current_project = current_project.get('parentProject')
            
            # Remove root project if it's "_Root"
            if path_parts and path_parts[0] == '_Root':
                path_parts = path_parts[1:]
            
            return ' > '.join(path_parts)
            
        except Exception as e:
            print(f"Warning: Could not get full path for project {project_name}: {e}")
            return project_name
    
    def _matches_filters(self, build: Dict[str, Any]) -> bool:
        """
        Check if a build matches the specified filters based on project folder structure
        
        Args:
            build: Build dictionary from TeamCity API
            
        Returns:
            True if build matches filters, False otherwise
        """
        # If no filters are specified, include all builds
        if not self.project_filters and not self.build_type_filters and not self.build_name_filters:
            return True
        
        build_type = build.get('buildType', {})
        project_name = build_type.get('projectName', '').lower()
        project_id = build_type.get('projectId', '')
        build_type_name = build_type.get('name', '').lower()
        build_type_id = build.get('buildTypeId', '').lower()
        
        # Get full project path for better matching
        full_project_path = self._get_project_full_path(project_id, project_name).lower()
        
        # Check project filters with enhanced matching logic
        project_match = True
        if self.project_filters:
            project_match = False  # Default to False, must match at least one filter
            
            for filter_term in self.project_filters:
                filter_lower = filter_term.lower()
                
                # Method 1: Exact project name match
                if filter_lower == project_name:
                    project_match = True
                    break
                
                # Method 2: Full path contains the filter (for nested projects)
                elif filter_lower in full_project_path:
                    project_match = True
                    break
                
                # Method 3: Path-like filter matching (e.g., "project1 > project1a")
                elif ' > ' in filter_lower:
                    # User specified a hierarchical path
                    if filter_lower == full_project_path:
                        project_match = True
                        break
                    # Also check if the full path ends with the specified path
                    elif full_project_path.endswith(filter_lower):
                        project_match = True
                        break
                
                # Method 4: Slash-separated path matching (e.g., "project1/project1a")
                elif '/' in filter_lower:
                    # Convert slash format to hierarchy format
                    hierarchy_filter = filter_lower.replace('/', ' > ')
                    if hierarchy_filter == full_project_path:
                        project_match = True
                        break
                    elif full_project_path.endswith(hierarchy_filter):
                        project_match = True
                        break
                
                # Method 5: Check if it's a direct child project reference
                elif filter_lower == project_id.lower():
                    project_match = True
                    break
        
        # Check build type filters  
        build_type_match = True
        if self.build_type_filters:
            build_type_match = any(
                filter_term in build_type_name or filter_term in build_type_id
                for filter_term in self.build_type_filters
            )
        
        # Check build name filters
        build_name_match = True
        if self.build_name_filters:
            build_name_match = any(
                filter_term in build_type_name
                for filter_term in self.build_name_filters
            )
        
        # Build must match ALL specified filter categories
        return project_match and build_type_match and build_name_match
    
    def _get_matched_filter(self, build: Dict[str, Any]) -> str:
        """
        Determine which specific filter matched this build
        
        Args:
            build: Build dictionary from TeamCity API
            
        Returns:
            The filter string that matched this build
        """
        build_type = build.get('buildType', {})
        project_name = build_type.get('projectName', '').lower()
        project_id = build_type.get('projectId', '')
        full_project_path = self._get_project_full_path(project_id, project_name).lower()
        
        # Check which project filter matched
        for i, filter_term in enumerate(self.project_filters):
            filter_lower = filter_term.lower()
            
            # Check all the same matching logic as in _matches_filters
            if (filter_lower == project_name or
                filter_lower in full_project_path or
                (' > ' in filter_lower and (filter_lower == full_project_path or full_project_path.endswith(filter_lower))) or
                ('/' in filter_lower and (filter_lower.replace('/', ' > ') == full_project_path or full_project_path.endswith(filter_lower.replace('/', ' > ')))) or
                filter_lower == project_id.lower()):
                # Return the original filter (not lowercased)
                return self.original_project_filters[i] if i < len(self.original_project_filters) else filter_term
        
        return "Unknown Filter"
    
    def get_builds_last_month(self) -> List[Dict[str, Any]]:
        """
        Fetch all builds from the last month, filtered by specified criteria
        
        Returns:
            List of build dictionaries that match the filters
        """
        try:
            # Calculate date range for last month
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            # Format dates for TeamCity API (YYYYMMDDTHHMMSS+HHMM format)
            start_date_str = start_date.strftime('%Y%m%dT%H%M%S+0000')
            
            print(f"Fetching builds from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Print active filters
            if self.project_filters or self.build_type_filters or self.build_name_filters:
                print("Active filters:")
                if self.project_filters:
                    print(f"  - Project/Folder filters: {', '.join(self.original_project_filters)}")
                if self.build_type_filters:
                    print(f"  - Build type filters: {', '.join(self.build_type_filters)}")
                if self.build_name_filters:
                    print(f"  - Build name filters: {', '.join(self.build_name_filters)}")
            else:
                print("No filters applied - fetching all builds")
            
            # TeamCity locator for builds in date range
            locator = f"sinceDate:{start_date_str}"
            
            builds = []
            filtered_builds = []
            count = 100  # Number of builds per request
            start = 0
            
            while True:
                url = f"{self.api_url}/builds"
                params = {
                    'locator': f"{locator},start:{start},count:{count}",
                    'fields': 'build(id,number,status,state,buildTypeId,startDate,finishDate,queuedDate,branchName,statusText,triggered(type,user(username,name),date),buildType(id,name,projectId,projectName))'
                }
                
                try:
                    response = requests.get(url, headers=self.headers, params=params, timeout=30)
                    response.raise_for_status()
                    
                    data = response.json()
                    batch_builds = data.get('build', [])
                    
                    if not batch_builds:
                        break
                    
                    builds.extend(batch_builds)
                    
                    # Apply filters to this batch
                    for build in batch_builds:
                        if self._matches_filters(build):
                            filtered_builds.append(build)
                    
                    print(f"Fetched {len(builds)} total builds, {len(filtered_builds)} match filters...")
                    
                    # Check if we got fewer builds than requested (end of results)
                    if len(batch_builds) < count:
                        break
                    
                    start += count
                    
                except requests.exceptions.Timeout:
                    print(f"Request timeout while fetching builds. Retrying...")
                    continue
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching builds (batch starting at {start}): {e}")
                    if "401" in str(e):
                        raise Exception("Authentication failed. Please check your credentials.")
                    elif "403" in str(e):
                        raise Exception("Access forbidden. Please check your permissions.")
                    else:
                        print(f"Continuing with {len(filtered_builds)} builds collected so far...")
                        break
            
            print(f"Total builds fetched: {len(builds)}")
            print(f"Builds matching filters: {len(filtered_builds)}")
            
            if filtered_builds:
                # Show sample of filtered builds for verification
                print("\nSample of filtered builds:")
                for i, build in enumerate(filtered_builds[:5]):
                    build_type = build.get('buildType', {})
                    project_path = self._get_project_full_path(
                        build_type.get('projectId', ''), 
                        build_type.get('projectName', '')
                    )
                    print(f"  {i+1}. Path: {project_path}")
                    print(f"     Build: {build_type.get('name', 'N/A')} | Status: {build.get('status', 'N/A')}")
                if len(filtered_builds) > 5:
                    print(f"  ... and {len(filtered_builds) - 5} more")
            
            return filtered_builds
            
        except Exception as e:
            print(f"Critical error in get_builds_last_month: {e}")
            raise
    
    def extract_application_code(self, build_type_name: str, project_name: str) -> str:
        """
        Extract application code from build type name or project name
        You may need to customize this based on your naming conventions
        
        Args:
            build_type_name: Name of the build type
            project_name: Name of the project
            
        Returns:
            Application code string
        """
        # Example logic - customize based on your naming conventions
        if '_' in build_type_name:
            return build_type_name.split('_')[0]
        elif '-' in build_type_name:
            return build_type_name.split('-')[0]
        else:
            return project_name
    
    def analyze_builds(self, builds: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Analyze builds and create summary statistics
        
        Args:
            builds: List of build dictionaries
            
        Returns:
            DataFrame with analysis results
        """
        analysis_data = []
        
        for build in builds:
            try:
                # Extract build information
                build_type = build.get('buildType', {})
                build_type_id = build.get('buildTypeId', '')
                build_type_name = build_type.get('name', '')
                project_name = build_type.get('projectName', '')
                project_id = build_type.get('projectId', '')
                
                # Get full project path
                full_project_path = self._get_project_full_path(project_id, project_name)
                
                # Extract application code
                app_code = self.extract_application_code(build_type_name, project_name)
                
                # Parse timestamps
                start_date = build.get('startDate', '')
                queued_date = build.get('queuedDate', '')
                finish_date = build.get('finishDate', '')
                
                # Convert timestamps to datetime objects
                timestamp = None
                start_datetime = None
                finish_datetime = None
                build_duration_minutes = None
                
                if start_date:
                    try:
                        timestamp = datetime.strptime(start_date[:15], '%Y%m%dT%H%M%S')
                        start_datetime = timestamp
                    except ValueError:
                        pass
                
                # Calculate build duration
                if start_date and finish_date:
                    try:
                        start_dt = datetime.strptime(start_date[:15], '%Y%m%dT%H%M%S')
                        finish_dt = datetime.strptime(finish_date[:15], '%Y%m%dT%H%M%S')
                        duration = finish_dt - start_dt
                        build_duration_minutes = duration.total_seconds() / 60
                    except ValueError:
                        pass
                
                # Determine build status
                status = build.get('status', 'UNKNOWN')
                state = build.get('state', 'UNKNOWN')
                
                is_successful = status == 'SUCCESS' and state == 'finished'
                is_failed = status == 'FAILURE' and state == 'finished'
                
                # Extract trigger information (who kicked off the build)
                triggered_by = 'Unknown'
                trigger_type = 'Unknown'
                trigger_date = ''
                
                triggered_info = build.get('triggered', {})
                if triggered_info:
                    trigger_type = triggered_info.get('type', 'Unknown')
                    trigger_date = triggered_info.get('date', '')
                    
                    if trigger_type == 'user':
                        user_info = triggered_info.get('user', {})
                        if user_info:
                            username = user_info.get('username', '')
                            name = user_info.get('name', '')
                            triggered_by = f"{name} ({username})" if name and username else (username or name or 'Unknown User')
                    elif trigger_type == 'vcs':
                        triggered_by = 'VCS Trigger (Code Change)'
                    elif trigger_type == 'schedule':
                        triggered_by = 'Scheduled Trigger'
                    elif trigger_type == 'dependency':
                        triggered_by = 'Dependency Trigger'
                    else:
                        triggered_by = f"{trigger_type.title()} Trigger"
                
                # Determine which filter matched this build
                matched_filter = self._get_matched_filter(build)
                
                analysis_data.append({
                    'application_code': app_code,
                    'build_type_id': build_type_id,
                    'build_type_name': build_type_name,
                    'project_name': project_name,
                    'project_path': full_project_path,
                    'build_id': build.get('id', ''),
                    'build_number': build.get('number', ''),
                    'status': status,
                    'state': state,
                    'timestamp': timestamp,
                    'start_date': start_date,
                    'queued_date': queued_date,
                    'finish_date': finish_date,
                    'branch': build.get('branchName', 'default'),
                    'is_successful': is_successful,
                    'is_failed': is_failed,
                    'status_text': build.get('statusText', ''),
                    'triggered_by': triggered_by,
                    'trigger_type': trigger_type,
                    'trigger_date': trigger_date,
                    'build_duration_minutes': build_duration_minutes,
                    'matched_filter': matched_filter
                })
                
            except Exception as e:
                print(f"Error processing build {build.get('id', 'unknown')}: {e}")
                continue
        
        return pd.DataFrame(analysis_data)
    
    def generate_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics grouped by application code and build type
        
        Args:
            df: DataFrame with build data
            
        Returns:
            DataFrame with summary statistics
        """
        # Group by application code and build type
        summary = df.groupby(['application_code', 'build_type_name']).agg({
            'build_id': 'count',  # Total builds
            'is_successful': 'sum',  # Successful builds
            'is_failed': 'sum',  # Failed builds
            'timestamp': ['min', 'max']  # First and last build timestamps
        }).round(2)
        
        # Flatten column names
        summary.columns = ['total_builds', 'successful_builds', 'failed_builds', 'first_build', 'last_build']
        
        # Calculate success rate
        summary['success_rate'] = (summary['successful_builds'] / summary['total_builds'] * 100).round(2)
        summary['failure_rate'] = (summary['failed_builds'] / summary['total_builds'] * 100).round(2)
        
        # Reset index to make grouping columns regular columns
        summary = summary.reset_index()
        
        # Sort by application code, then by total builds (descending)
        summary = summary.sort_values(['application_code', 'total_builds'], ascending=[True, False])
        
        return summary
    
    def save_results(self, df: pd.DataFrame, summary: pd.DataFrame, output_prefix: str = 'teamcity_analysis'):
        """
        Save results to CSV files and formatted Excel file
        
        Args:
            df: Detailed build data
            summary: Summary statistics
            output_prefix: Prefix for output files
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed data
        detailed_file = f"{output_prefix}_detailed_{timestamp}.csv"
        df.to_csv(detailed_file, index=False)
        print(f"Detailed data saved to: {detailed_file}")
        
        # Save summary data
        summary_file = f"{output_prefix}_summary_{timestamp}.csv"
        summary.to_csv(summary_file, index=False)
        print(f"Summary data saved to: {summary_file}")
        
        # Save formatted Excel file
        excel_file = f"{output_prefix}_report_{timestamp}.xlsx"
        self.create_excel_report(df, summary, excel_file)
        print(f"Formatted Excel report saved to: {excel_file}")
        
        return detailed_file, summary_file, excel_file
    
    def create_excel_report(self, df: pd.DataFrame, summary: pd.DataFrame, filename: str):
        """
        Create a beautifully formatted Excel report with multiple sheets
        
        Args:
            df: Detailed build data
            summary: Summary statistics
            filename: Output Excel filename
        """
        wb = Workbook()
        
        # Remove default sheet
        wb.remove(wb.active)
        
        # Create sheets
        self._create_executive_summary_sheet(wb, df, summary)  # Make this first
        self._create_overview_sheet(wb, df, summary)
        self._create_summary_sheet(wb, summary)
        self._create_detailed_sheet(wb, df)
        self._create_charts_sheet(wb, df)
        
        # Save workbook
        wb.save(filename)
    
    def _create_executive_summary_sheet(self, wb: Workbook, df: pd.DataFrame, summary: pd.DataFrame):
        """
        Create executive summary sheet by project filter
        """
        ws = wb.create_sheet("Executive Summary", 0)  # Make it the first sheet (index 0)
        
        # Title
        ws['A1'] = "Executive Summary - Build Analysis by Project Filter"
        ws['A1'].font = Font(size=18, bold=True, color="2E4057")
        ws['A2'] = f"Analysis Period: Last 30 Days ({datetime.now().strftime('%Y-%m-%d')})"
        ws['A2'].font = Font(size=12, italic=True)
        
        # Check if we have the required columns
        if 'matched_filter' not in df.columns:
            ws['A4'] = "Error: Build analysis data incomplete. Please regenerate the report."
            ws['A4'].font = Font(size=12, color="FF0000", bold=True)
            return
        
        # Create summary for each project filter
        filter_summaries = []
        
        for project_filter in self.original_project_filters:
            # Filter data for this specific project filter
            filtered_data = df[df['matched_filter'] == project_filter]
            
            if len(filtered_data) > 0:
                total_builds = len(filtered_data)
                successful_builds = filtered_data['is_successful'].sum()
                failed_builds = filtered_data['is_failed'].sum()
                success_rate = (successful_builds / total_builds * 100) if total_builds > 0 else 0
                failure_rate = (failed_builds / total_builds * 100) if total_builds > 0 else 0
                
                # Calculate average build time
                avg_build_time = filtered_data['build_duration_minutes'].mean() if 'build_duration_minutes' in filtered_data.columns else None
                
                # Find most active user
                most_active_user = 'Unknown'
                if 'triggered_by' in filtered_data.columns:
                    # Count occurrences of each user, excluding system triggers
                    user_counts = filtered_data[
                        (~filtered_data['triggered_by'].str.contains('VCS|Scheduled|Dependency|Trigger', na=False))
                    ]['triggered_by'].value_counts()
                    
                    if len(user_counts) > 0:
                        most_active_user = user_counts.index[0]
                
                filter_summaries.append({
                    'Project Filter': project_filter,
                    'Total Builds': total_builds,
                    'Successful': successful_builds,
                    'Failed': failed_builds,
                    'Success Rate (%)': f"{success_rate:.1f}%",
                    'Failure Rate (%)': f"{failure_rate:.1f}%",
                    'Avg Build Time (min)': f"{avg_build_time:.1f}" if pd.notna(avg_build_time) else "N/A",
                    'Most Active User': most_active_user,
                    'success_rate_numeric': success_rate  # For coloring
                })
            else:
                # No builds found for this filter
                filter_summaries.append({
                    'Project Filter': project_filter,
                    'Total Builds': 0,
                    'Successful': 0,
                    'Failed': 0,
                    'Success Rate (%)': "N/A",
                    'Failure Rate (%)': "N/A",
                    'Avg Build Time (min)': "N/A",
                    'Most Active User': "N/A",
                    'success_rate_numeric': 0
                })
        
        # Create the table
        start_row = 5
        headers = [
            "Project Filter",
            "Total Builds", 
            "Successful",
            "Failed",
            "Success Rate (%)",
            "Failure Rate (%)",
            "Avg Build Time (min)",
            "Most Active User"
        ]
        
        # Write headers
        for j, header in enumerate(headers):
            cell = ws.cell(row=start_row, column=j + 1, value=header)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(start_color="2E4057", end_color="2E4057", fill_type="solid")
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
        
        # Write data rows
        for i, row_data in enumerate(filter_summaries):
            data_values = [
                row_data['Project Filter'],
                row_data['Total Builds'],
                row_data['Successful'],
                row_data['Failed'],
                row_data['Success Rate (%)'],
                row_data['Failure Rate (%)'],
                row_data['Avg Build Time (min)'],
                row_data['Most Active User']
            ]
            
            for j, value in enumerate(data_values):
                cell = ws.cell(row=start_row + 1 + i, column=j + 1, value=value)
                cell.border = Border(
                    left=Side(style='thin'),
                    right=Side(style='thin'),
                    top=Side(style='thin'),
                    bottom=Side(style='thin')
                )
                cell.alignment = Alignment(horizontal='center', vertical='center')
                
                # Color code success rate column
                if j == 4 and row_data['Total Builds'] > 0:  # Success rate column
                    success_rate = row_data['success_rate_numeric']
                    if success_rate >= 90:
                        cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                    elif success_rate >= 70:
                        cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
                    else:
                        cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
        
        # Add summary totals at the bottom
        total_row = start_row + len(filter_summaries) + 2
        ws.cell(row=total_row, column=1, value="OVERALL TOTALS:").font = Font(bold=True, size=12)
        
        overall_total_builds = sum([row['Total Builds'] for row in filter_summaries])
        overall_successful = sum([row['Successful'] for row in filter_summaries])
        overall_failed = sum([row['Failed'] for row in filter_summaries])
        overall_success_rate = (overall_successful / overall_total_builds * 100) if overall_total_builds > 0 else 0
        
        # Calculate weighted average build time
        total_build_time = 0
        total_builds_with_time = 0
        for row in filter_summaries:
            if row['Avg Build Time (min)'] != "N/A" and row['Total Builds'] > 0:
                try:
                    avg_time = float(row['Avg Build Time (min)'])
                    total_build_time += avg_time * row['Total Builds']
                    total_builds_with_time += row['Total Builds']
                except ValueError:
                    pass
        
        overall_avg_build_time = total_build_time / total_builds_with_time if total_builds_with_time > 0 else 0
        
        totals_data = [
            ["Total Builds:", overall_total_builds],
            ["Total Successful:", f"{overall_successful} ({overall_success_rate:.1f}%)"],
            ["Total Failed:", f"{overall_failed} ({100-overall_success_rate:.1f}%)"],
            ["Overall Avg Build Time:", f"{overall_avg_build_time:.1f} minutes" if overall_avg_build_time > 0 else "N/A"]
        ]
        
        for i, (label, value) in enumerate(totals_data):
            ws.cell(row=total_row + 1 + i, column=1, value=label).font = Font(bold=True)
            ws.cell(row=total_row + 1 + i, column=2, value=value)
        
        # Auto-adjust column widths
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column_letter].width = adjusted_width
    
    def _create_overview_sheet(self, wb: Workbook, df: pd.DataFrame, summary: pd.DataFrame):
        """Create overview sheet with key metrics"""
        ws = wb.create_sheet("Overview")
        
        # Title
        ws['A1'] = "TeamCity Build Analysis Report"
        ws['A1'].font = Font(size=20, bold=True, color="2E4057")
        ws['A2'] = f"Analysis Period: Last 30 Days ({datetime.now().strftime('%Y-%m-%d')})"
        ws['A2'].font = Font(size=12, italic=True)
        
        # Key metrics
        total_apps = summary['application_code'].nunique()
        total_build_types = len(summary)
        total_builds = summary['total_builds'].sum()
        total_successful = summary['successful_builds'].sum()
        total_failed = summary['failed_builds'].sum()
        overall_success_rate = (total_successful / total_builds * 100) if total_builds > 0 else 0
        
        # Metrics table
        metrics_data = [
            ["Metric", "Value"],
            ["Total Applications", total_apps],
            ["Total Build Types", total_build_types],
            ["Total Builds", total_builds],
            ["Successful Builds", f"{total_successful} ({overall_success_rate:.1f}%)"],
            ["Failed Builds", f"{total_failed} ({100-overall_success_rate:.1f}%)"],
            ["Overall Success Rate", f"{overall_success_rate:.1f}%"]
        ]
        
        # Write metrics
        start_row = 4
        for i, row in enumerate(metrics_data):
            for j, value in enumerate(row):
                cell = ws.cell(row=start_row + i, column=j + 1, value=value)
                if i == 0:  # Header row
                    cell.font = Font(bold=True, color="FFFFFF")
                    cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
                elif j == 0:  # Metric names
                    cell.font = Font(bold=True)
                
                cell.border = Border(
                    left=Side(style='thin'),
                    right=Side(style='thin'),
                    top=Side(style='thin'),
                    bottom=Side(style='thin')
                )
                cell.alignment = Alignment(horizontal='left', vertical='center')
        
        # Top applications table
        ws['A13'] = "Top 10 Applications by Build Volume"
        ws['A13'].font = Font(size=14, bold=True, color="2E4057")
        
        app_summary = summary.groupby('application_code').agg({
            'total_builds': 'sum',
            'successful_builds': 'sum',
            'failed_builds': 'sum'
        }).reset_index()
        app_summary['success_rate'] = (app_summary['successful_builds'] / app_summary['total_builds'] * 100).round(1)
        app_summary = app_summary.sort_values('total_builds', ascending=False).head(10)
        
        # Write top apps table
        headers = ["Application", "Total Builds", "Successful", "Failed", "Success Rate"]
        start_row = 15
        
        for j, header in enumerate(headers):
            cell = ws.cell(row=start_row, column=j + 1, value=header)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(start_color="70AD47", end_color="70AD47", fill_type="solid")
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
        
        for i, (_, row) in enumerate(app_summary.iterrows()):
            row_data = [
                row['application_code'],
                row['total_builds'],
                row['successful_builds'],
                row['failed_builds'],
                f"{row['success_rate']:.1f}%"
            ]
            
            for j, value in enumerate(row_data):
                cell = ws.cell(row=start_row + 1 + i, column=j + 1, value=value)
                cell.border = Border(
                    left=Side(style='thin'),
                    right=Side(style='thin'),
                    top=Side(style='thin'),
                    bottom=Side(style='thin')
                )
                cell.alignment = Alignment(horizontal='center', vertical='center')
                
                # Color code success rates
                if j == 4:  # Success rate column
                    if row['success_rate'] >= 90:
                        cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                    elif row['success_rate'] >= 70:
                        cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
                    else:
                        cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
        
        # Auto-adjust column widths
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column_letter].width = adjusted_width
    
    def _create_summary_sheet(self, wb: Workbook, summary: pd.DataFrame):
        """Create formatted summary sheet"""
        ws = wb.create_sheet("Summary by Build Type")
        
        # Title
        ws['A1'] = "Build Summary by Application and Build Type"
        ws['A1'].font = Font(size=16, bold=True, color="2E4057")
        
        # Write data starting from row 3
        start_row = 3
        
        # Headers
        headers = list(summary.columns)
        for j, header in enumerate(headers):
            cell = ws.cell(row=start_row, column=j + 1, value=header.replace('_', ' ').title())
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
        
        # Data rows
        for i, (_, row) in enumerate(summary.iterrows()):
            for j, value in enumerate(row):
                cell = ws.cell(row=start_row + 1 + i, column=j + 1, value=value)
                cell.border = Border(
                    left=Side(style='thin'),
                    right=Side(style='thin'),
                    top=Side(style='thin'),
                    bottom=Side(style='thin')
                )
                cell.alignment = Alignment(horizontal='center', vertical='center')
                
                # Format percentage columns
                if 'rate' in summary.columns[j].lower():
                    if isinstance(value, (int, float)):
                        if value >= 90:
                            cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                        elif value >= 70:
                            cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
                        else:
                            cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
        
        # Auto-adjust column widths
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column_letter].width = adjusted_width
    
    def _create_detailed_sheet(self, wb: Workbook, df: pd.DataFrame):
        """Create detailed data sheet"""
        ws = wb.create_sheet("Detailed Build Data")
        
        # Title
        ws['A1'] = "Detailed Build Information"
        ws['A1'].font = Font(size=16, bold=True, color="2E4057")
        
        # Select and reorder columns for better readability
        columns_to_show = [
            'project_name', 'build_type_name', 'build_number', 'status', 
            'timestamp', 'branch', 'triggered_by', 'trigger_type', 'status_text'
        ]
        
        # Only include columns that exist in the DataFrame
        available_columns = [col for col in columns_to_show if col in df.columns]
        display_df = df[available_columns].copy()
        
        # Format timestamp
        if 'timestamp' in display_df.columns:
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Write data starting from row 3
        start_row = 3
        
        # Headers
        for j, column in enumerate(display_df.columns):
            cell = ws.cell(row=start_row, column=j + 1, value=column.replace('_', ' ').title())
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
        
        # Data rows
        for i, (_, row) in enumerate(display_df.iterrows()):
            for j, value in enumerate(row):
                cell = ws.cell(row=start_row + 1 + i, column=j + 1, value=value)
                cell.border = Border(
                    left=Side(style='thin'),
                    right=Side(style='thin'),
                    top=Side(style='thin'),
                    bottom=Side(style='thin')
                )
                cell.alignment = Alignment(horizontal='left', vertical='center')
                
                # Color code status column
                if j == 3 and 'status' in display_df.columns:  # Status column
                    if value == 'SUCCESS':
                        cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                    elif value == 'FAILURE':
                        cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
                    else:
                        cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
        
        # Auto-adjust column widths
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column_letter].width = adjusted_width
    
    def _create_charts_sheet(self, wb: Workbook, df: pd.DataFrame):
        """Create charts and visualizations sheet"""
        ws = wb.create_sheet("Charts & Analytics")
        
        # Title
        ws['A1'] = "Build Analytics Dashboard"
        ws['A1'].font = Font(size=16, bold=True, color="2E4057")
        
        # Prepare data for charts based on project filters
        filter_summaries = []
        for project_filter in self.original_project_filters:
            filtered_data = df[df['matched_filter'] == project_filter]
            
            if len(filtered_data) > 0:
                total_builds = len(filtered_data)
                successful_builds = filtered_data['is_successful'].sum()
                failed_builds = filtered_data['is_failed'].sum()
                
                filter_summaries.append({
                    'project_filter': project_filter,
                    'total_builds': total_builds,
                    'successful_builds': successful_builds,
                    'failed_builds': failed_builds
                })
        
        # Create data table for chart
        chart_start_row = 3
        ws.cell(row=chart_start_row, column=1, value="Project Filter")
        ws.cell(row=chart_start_row, column=2, value="Total Builds")
        ws.cell(row=chart_start_row, column=3, value="Successful")
        ws.cell(row=chart_start_row, column=4, value="Failed")
        
        for i, summary in enumerate(filter_summaries):
            ws.cell(row=chart_start_row + 1 + i, column=1, value=summary['project_filter'])
            ws.cell(row=chart_start_row + 1 + i, column=2, value=summary['total_builds'])
            ws.cell(row=chart_start_row + 1 + i, column=3, value=summary['successful_builds'])
            ws.cell(row=chart_start_row + 1 + i, column=4, value=summary['failed_builds'])
        
        # Create bar chart
        chart = BarChart()
        chart.type = "col"
        chart.style = 10
        chart.title = "Build Volume by Project Filter"
        chart.y_axis.title = 'Number of Builds'
        chart.x_axis.title = 'Project Filters'
        
        # Define data ranges
        data = Reference(ws, min_col=2, min_row=chart_start_row, max_row=chart_start_row + len(filter_summaries), max_col=4)
        cats = Reference(ws, min_col=1, min_row=chart_start_row + 1, max_row=chart_start_row + len(filter_summaries))
        
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(cats)
        chart.height = 10
        chart.width = 15
        
        # Add chart to worksheet
        ws.add_chart(chart, "F3")
    
    def print_summary_report(self, summary: pd.DataFrame):
        """
        Print a formatted summary report
        
        Args:
            summary: Summary DataFrame
        """
        print("\n" + "="*80)
        print("TEAMCITY BUILD ANALYSIS REPORT - LAST 30 DAYS")
        print("="*80)
        
        total_apps = summary['application_code'].nunique()
        total_build_types = len(summary)
        total_builds = summary['total_builds'].sum()
        total_successful = summary['successful_builds'].sum()
        total_failed = summary['failed_builds'].sum()
        
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total Applications: {total_apps}")
        print(f"  Total Build Types: {total_build_types}")
        print(f"  Total Builds: {total_builds}")
        print(f"  Successful Builds: {total_successful} ({total_successful/total_builds*100:.1f}%)")
        print(f"  Failed Builds: {total_failed} ({total_failed/total_builds*100:.1f}%)")
        
        print(f"\nTOP 10 MOST ACTIVE BUILD TYPES:")
        print("-" * 80)
        top_builds = summary.nlargest(10, 'total_builds')
        for _, row in top_builds.iterrows():
            print(f"  {row['application_code']:<20} | {row['build_type_name']:<30} | "
                  f"Total: {row['total_builds']:<4} | Success: {row['success_rate']:<5.1f}%")
        
        print(f"\nBUILD SUMMARY BY APPLICATION:")
        print("-" * 80)
        app_summary = summary.groupby('application_code').agg({
            'total_builds': 'sum',
            'successful_builds': 'sum',
            'failed_builds': 'sum'
        }).reset_index()
        app_summary['success_rate'] = (app_summary['successful_builds'] / app_summary['total_builds'] * 100).round(1)
        app_summary = app_summary.sort_values('total_builds', ascending=False)
        
        for _, row in app_summary.iterrows():
            print(f"  {row['application_code']:<20} | Total: {row['total_builds']:<4} | "
                  f"Success: {row['successful_builds']:<4} | Failed: {row['failed_builds']:<4} | "
                  f"Success Rate: {row['success_rate']:<5.1f}%")

def main():
    """
    Main function to run the analysis
    """
    # Configuration - Update these values
    TEAMCITY_URL = "https://your-teamcity-server.com"  # Replace with your TeamCity URL
    USERNAME = "your_username"  # Replace with your username
    PASSWORD = "your_password"  # Replace with your password
    
    # =================== FILTER CONFIGURATION ===================
    # Specify which builds you want to analyze (all filters are optional)
    
    # Filter by project names/paths (case-insensitive, supports multiple formats)
    PROJECT_FILTERS = [
        # Method 1: Exact project name
        "project1a",
        "project2a",
        
        # Method 2: Full hierarchical path with " > " separator
        # "Project1 > project1a",
        # "Project2 > project2a",
        
        # Method 3: Path with "/" separator (will be converted to hierarchy)
        # "Project1/project1a",
        # "Project2/project2a",
        
        # Method 4: For your specific PAT Builds requirements:
        "PAT Builds",  # This will match any project named "PAT Builds"
        "Veritas Release Projects > PAT - Release Builds"  # Full hierarchy path
        
        # Alternative ways to specify the same:
        # "PAT - Release Builds",  # Just the subproject name
        # "Veritas Release Projects/PAT - Release Builds"  # Slash format
    ]
    
    # Filter by build type names (case-insensitive, partial matching)  
    BUILD_TYPE_FILTERS = [
        # Examples:
        # "Production Deploy",
        # "Integration Test",
        # "Unit Test"
    ]
    
    # Filter by specific build names (case-insensitive, partial matching)
    BUILD_NAME_FILTERS = [
        # Examples:
        # "nightly",
        # "release",
        # "hotfix"
    ]
    
    # ============================================================
    # IMPORTANT NOTES:
    # - Project filters support multiple formats:
    #   1. Exact project name: "project1a"
    #   2. Hierarchy with " > ": "Project1 > project1a"
    #   3. Hierarchy with "/": "Project1/project1a"
    # - All matching is case-insensitive
    # - You can mix different formats in the same list
    # - If you leave a filter list empty [], it won't filter on that criteria
    # ============================================================
    
    try:
        # Initialize analyzer with filters
        analyzer = TeamCityBuildAnalyzer(
            TEAMCITY_URL, 
            USERNAME, 
            PASSWORD,
            project_filters=PROJECT_FILTERS,
            build_type_filters=BUILD_TYPE_FILTERS,
            build_name_filters=BUILD_NAME_FILTERS
        )
        
        # Fetch builds from last month (filtered)
        print("Starting TeamCity build analysis...")
        builds = analyzer.get_builds_last_month()
        
        if not builds:
            print("No builds found matching the specified filters for the last month.")
            print("Consider:")
            print("  1. Checking if the filter terms are correct")
            print("  2. Verifying the project/build names in TeamCity")
            print("  3. Try using the full hierarchy path: 'Parent Project > Sub Project'")
            print("  4. Expanding the date range or removing some filters")
            return
        
        # Analyze builds
        print("Analyzing build data...")
        df = analyzer.analyze_builds(builds)
        
        # Generate summary
        summary = analyzer.generate_summary(df)
        
        # Save results (now includes Excel)
        analyzer.save_results(df, summary)
        
        # Print report
        analyzer.print_summary_report(summary)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("Files generated:")
        print("  - CSV files for detailed and summary data")
        print("  - Formatted Excel report with multiple sheets")
        print("="*60)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()
```