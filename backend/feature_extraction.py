"""
Feature extraction from repository and code for fault localization.
Extracts candidate lines, computes features, and integrates with git/coverage data.
"""

import os
import re
import ast
import hashlib
from typing import List, Dict, Tuple
from pathlib import Path
import subprocess

try:
    from radon.complexity import cc_visit
    from radon.metrics import mi_visit
except ImportError:
    cc_visit = None
    mi_visit = None


class FeatureExtractor:
    """
    Extracts features from code and repository for fault localization.
    """
    
    def __init__(self, repository_path: str):
        """
        Initialize feature extractor with repository path.
        
        Args:
            repository_path: Path to the git repository
        """
        self.repository_path = Path(repository_path)
        self.file_cache = {}
        
    def extract_candidate_lines(
        self, 
        failing_test: str, 
        error_message: str = None,
        stack_trace: str = None,
        max_candidates: int = 100
    ) -> List[Dict]:
        """
        Extract candidate lines that might contain the fault.
        Uses stack trace, recent changes, and Python files in the repo.
        
        Args:
            failing_test: Name of the failing test
            error_message: Error message from test failure
            stack_trace: Stack trace from test failure
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of dictionaries with candidate line information
        """
        candidates = []
        
        # 1. Extract lines from stack trace (highest priority)
        if stack_trace:
            stack_candidates = self._extract_from_stack_trace(stack_trace)
            candidates.extend(stack_candidates)
        
        # 2. Get recently changed files from git
        recent_files = self._get_recent_changed_files(days=30)
        
        # 3. Get all code files if not enough candidates
        if len(candidates) < max_candidates:
            # Support multiple programming languages
            code_files = []
            for pattern in ["*.py", "*.cpp", "*.c", "*.h", "*.java", "*.js", "*.jsx", "*.ts", "*.tsx"]:
                code_files.extend(list(self.repository_path.rglob(pattern)))
            # Prioritize recently changed files
            for file_path in recent_files:
                if file_path not in [c['file_path'] for c in candidates]:
                    lines = self._extract_lines_from_file(file_path)
                    candidates.extend(lines[:max_candidates - len(candidates)])
                    if len(candidates) >= max_candidates:
                        break
            
            # Add other code files if still not enough
            for file_path in code_files:
                if len(candidates) >= max_candidates:
                    break
                if file_path not in [c['file_path'] for c in candidates]:
                    lines = self._extract_lines_from_file(file_path)
                    candidates.extend(lines[:max_candidates - len(candidates)])
        
        # Limit to max_candidates
        candidates = candidates[:max_candidates]
        
        # Compute features for each candidate
        for i, candidate in enumerate(candidates):
            candidate['features'] = self._compute_features(
                candidate, 
                failing_test,
                error_message,
                stack_trace
            )
        
        return candidates
    
    def _extract_from_stack_trace(self, stack_trace: str) -> List[Dict]:
        """
        Extract file paths and line numbers from stack trace.
        Supports multiple formats: Python, C/C++, Java, etc.
        
        Args:
            stack_trace: Stack trace string
            
        Returns:
            List of candidate dictionaries with 'in_stack_trace' flag
        """
        candidates = []
        
        # Pattern 1: Python style - File "path/to/file.py", line 123
        pattern1 = r'File\s+"([^"]+)",\s+line\s+(\d+)'
        matches = re.findall(pattern1, stack_trace)
        
        for file_path, line_num in matches:
            file_path = Path(file_path)
            if file_path.exists():
                code = self._get_line_content(file_path, int(line_num))
                if code:
                    candidates.append({
                        'file_path': str(file_path),
                        'line_number': int(line_num),
                        'code': code,
                        'source': 'stack_trace',
                        'in_stack_trace': True
                    })
        
        # Pattern 2: C/C++ style - "file.cpp:123" or "at file.cpp:123"
        pattern2 = r'(?:at\s+)?([a-zA-Z_][\w/\\\.]*\.(?:cpp|c|h|cc|cxx|hpp)):(\d+)'
        matches2 = re.findall(pattern2, stack_trace, re.IGNORECASE)
        
        for file_name, line_num in matches2:
            # Search for the file in repository
            file_path = self._find_file_in_repo(file_name)
            if file_path and file_path.exists():
                code = self._get_line_content(file_path, int(line_num))
                if code:
                    candidates.append({
                        'file_path': str(file_path),
                        'line_number': int(line_num),
                        'code': code,
                        'source': 'stack_trace',
                        'in_stack_trace': True
                    })
        
        # Pattern 3: Java style - "at package.Class.method(File.java:123)"
        pattern3 = r'\(([a-zA-Z_][\w]*\.java):(\d+)\)'
        matches3 = re.findall(pattern3, stack_trace)
        
        for file_name, line_num in matches3:
            file_path = self._find_file_in_repo(file_name)
            if file_path and file_path.exists():
                code = self._get_line_content(file_path, int(line_num))
                if code:
                    candidates.append({
                        'file_path': str(file_path),
                        'line_number': int(line_num),
                        'code': code,
                        'source': 'stack_trace',
                        'in_stack_trace': True
                    })
        
        # Pattern 4: Simple format - "filename.ext:123" or "filename.ext line 123"
        pattern4 = r'([a-zA-Z_][\w/\\\.]*\.(?:py|cpp|c|h|java|js|ts)):?\s*(?:line\s+)?(\d+)'
        matches4 = re.findall(pattern4, stack_trace, re.IGNORECASE)
        
        for file_name, line_num in matches4:
            file_path = self._find_file_in_repo(file_name)
            if file_path and file_path.exists():
                code = self._get_line_content(file_path, int(line_num))
                if code:
                    # Avoid duplicates
                    if not any(c['file_path'] == str(file_path) and c['line_number'] == int(line_num) 
                             for c in candidates):
                        candidates.append({
                            'file_path': str(file_path),
                            'line_number': int(line_num),
                            'code': code,
                            'source': 'stack_trace',
                            'in_stack_trace': True
                        })
        
        return candidates
    
    def _find_file_in_repo(self, file_name: str) -> Path:
        """
        Find a file in the repository by name.
        
        Args:
            file_name: Name of the file (e.g., 'calculator.cpp')
            
        Returns:
            Path to the file or None
        """
        # If it's already an absolute path, return it
        if os.path.isabs(file_name):
            return Path(file_name)
        
        # Search for file in repository
        for file_path in self.repository_path.rglob(file_name):
            return file_path
        
        # Try without directory prefix
        base_name = os.path.basename(file_name)
        for file_path in self.repository_path.rglob(base_name):
            return file_path
        
        return None
    
    def _get_recent_changed_files(self, days: int = 30) -> List[Path]:
        """
        Get files that changed recently using git log.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of file paths
        """
        try:
            # Use git log to get recently changed files
            cmd = f'git log --since="{days} days ago" --name-only --pretty=format: --diff-filter=AM'
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=self.repository_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                files = [
                    self.repository_path / line.strip() 
                    for line in result.stdout.split('\n') 
                    if line.strip() and any(line.endswith(ext) for ext in ['.py', '.cpp', '.c', '.h', '.java', '.js', '.jsx', '.ts', '.tsx'])
                ]
                return [f for f in files if f.exists()]
        except Exception as e:
            print(f"Git log failed: {e}")
        
        return []
    
    def _extract_lines_from_file(self, file_path: Path, max_lines: int = 200) -> List[Dict]:
        """
        Extract lines from a code file (Python, C, C++, Java, JavaScript).
        
        Args:
            file_path: Path to code file
            max_lines: Maximum number of lines to extract (increased to 200 for larger files)
            
        Returns:
            List of candidate dictionaries
        """
        candidates = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Determine file type
            file_ext = file_path.suffix.lower()
            
            # Define language-specific keywords
            if file_ext == '.py':
                keywords = ['def ', 'class ', 'if ', 'for ', 'while ', 'return ', 'raise ', 'assert ', 'import ']
                comment_chars = ['#']
            elif file_ext in ['.cpp', '.c', '.h']:
                keywords = ['void ', 'int ', 'char ', 'float ', 'double ', 'class ', 'struct ', 'if ', 'for ', 'while ', 'return ', 'throw ']
                comment_chars = ['//', '/*']
            elif file_ext == '.java':
                keywords = ['public ', 'private ', 'protected ', 'void ', 'int ', 'class ', 'interface ', 'if ', 'for ', 'while ', 'return ', 'throw ', 'new ']
                comment_chars = ['//', '/*']
            elif file_ext in ['.js', '.jsx', '.ts', '.tsx']:
                keywords = ['function ', 'const ', 'let ', 'var ', 'class ', 'if ', 'for ', 'while ', 'return ', 'throw ', 'async ', 'await ']
                comment_chars = ['//', '/*']
            else:
                keywords = ['if ', 'for ', 'while ', 'return ']
                comment_chars = ['#', '//']
            
            # Extract ALL non-empty, non-comment lines (not just keyword lines)
            for i, line in enumerate(lines[:max_lines], start=1):
                stripped = line.strip()
                
                # Skip empty lines
                if not stripped:
                    continue
                
                # Skip pure comment lines
                if any(stripped.startswith(c) for c in comment_chars):
                    continue
                
                # Skip closing braces only
                if stripped in ['}', '{', '};', '};', ');']:
                    continue
                
                # Include ALL executable lines (not just keyword lines)
                candidates.append({
                    'file_path': str(file_path),
                    'line_number': i,
                    'code': stripped,
                    'source': 'file_scan'
                })
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
        
        return candidates
    
    def _get_line_content(self, file_path: Path, line_number: int) -> str:
        """
        Get content of a specific line from a file.
        
        Args:
            file_path: Path to file
            line_number: Line number (1-indexed)
            
        Returns:
            Line content as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if 0 < line_number <= len(lines):
                    return lines[line_number - 1].strip()
        except Exception as e:
            print(f"Error reading line {line_number} from {file_path}: {e}")
        return ""
    
    def _compute_features(
        self, 
        candidate: Dict,
        failing_test: str,
        error_message: str,
        stack_trace: str
    ) -> Dict:
        """
        Compute additional features for a candidate line.
        
        Features:
        1. distance_from_error: Distance from error location (if in stack trace)
        2. cyclomatic_complexity: Complexity of containing function
        3. code_churn: Number of recent changes to this file
        4. test_coverage: Whether line is covered by tests (simulated)
        5. historical_faults: Number of past faults in this file (simulated)
        
        Args:
            candidate: Candidate dictionary
            failing_test: Test name
            error_message: Error message
            stack_trace: Stack trace
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Store error message and test name for Gemini
        features['error_message'] = error_message
        features['failing_test'] = failing_test
        features['stack_trace'] = stack_trace
        
        # 1. Distance from error location
        # Check if this line is directly in the stack trace
        if candidate.get('in_stack_trace', False):
            features['distance_from_error'] = 0.0  # Exact stack trace line
        elif stack_trace and candidate.get('source') == 'stack_trace':
            features['distance_from_error'] = 0.0  # In stack trace
        else:
            features['distance_from_error'] = 1.0  # Not in stack trace
        
        # 2. Cyclomatic complexity
        features['cyclomatic_complexity'] = self._get_complexity(
            candidate['file_path'], 
            candidate['line_number']
        )
        
        # 3. Code churn (number of commits touching this file)
        features['code_churn'] = self._get_code_churn(candidate['file_path'])
        
        # 4. Test coverage (simulated - would need actual coverage data)
        features['test_coverage'] = 0.8  # Assume 80% coverage
        
        # 5. Historical faults (simulated - would need historical data)
        features['historical_faults'] = 0.1  # Assume 10% historical fault rate
        
        return features
    
    def _get_complexity(self, file_path: str, line_number: int) -> float:
        """
        Get cyclomatic complexity of function containing the line.
        
        Args:
            file_path: Path to file
            line_number: Line number
            
        Returns:
            Complexity score (normalized 0-1)
        """
        if cc_visit is None:
            return 0.5  # Default if radon not available
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            complexities = cc_visit(code)
            # Find function containing this line
            for func in complexities:
                if func.lineno <= line_number <= func.endline:
                    # Normalize complexity (typical range 1-20)
                    return min(func.complexity / 20.0, 1.0)
        except Exception:
            pass
        
        return 0.5  # Default complexity
    
    def _get_code_churn(self, file_path: str) -> float:
        """
        Get code churn (number of recent commits) for a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Normalized churn score (0-1)
        """
        try:
            cmd = f'git log --oneline --follow -- "{file_path}"'
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=self.repository_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                num_commits = len(result.stdout.strip().split('\n'))
                # Normalize (typical range 0-50 commits)
                return min(num_commits / 50.0, 1.0)
        except Exception:
            pass
        
        return 0.5  # Default churn
    
    @staticmethod
    def compute_cache_key(file_path: str, line_number: int, code: str) -> str:
        """
        Compute cache key for embedding storage.
        
        Args:
            file_path: Path to file
            line_number: Line number
            code: Code content
            
        Returns:
            Hash string as cache key
        """
        content = f"{file_path}:{line_number}:{code}"
        return hashlib.sha256(content.encode()).hexdigest()
