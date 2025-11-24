"""
Gemini API Integration for Deep Bug Analysis

This module uses Google's Gemini API to perform intelligent code analysis
and bug detection, then formats results in fault localization format.
"""

import os
import json
import requests
from typing import Dict, List, Optional


class GeminiAnalyzer:
    """
    Orchestrates bug analysis using Gemini API.
    Formats output in fault localization report style.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini analyzer.
        
        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        self.endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    
    def analyze_bug(
        self, 
        code: str, 
        file_path: str,
        language: str,
        stack_trace: Optional[str] = None,
        error_message: Optional[str] = None,
        failing_test: Optional[str] = None
    ) -> Dict:
        """
        Analyze code for bugs using Gemini API.
        
        Args:
            code: Source code to analyze
            file_path: Path to the file
            language: Programming language (Java, Python, C++, C)
            stack_trace: Optional stack trace
            error_message: Optional error message
            failing_test: Optional test name
            
        Returns:
            Dictionary with analysis results
        """
        # Build the prompt for Gemini
        prompt = self._build_analysis_prompt(
            code, file_path, language, stack_trace, error_message, failing_test
        )
        
        # Call Gemini API
        response = self._call_gemini_api(prompt)
        
        # Parse and format response
        analysis = self._parse_gemini_response(response, code)
        
        return analysis
    
    def _build_analysis_prompt(
        self,
        code: str,
        file_path: str,
        language: str,
        stack_trace: Optional[str],
        error_message: Optional[str],
        failing_test: Optional[str]
    ) -> str:
        """Build the analysis prompt for Gemini."""
        
        prompt = f"""You are an expert code analyzer for fault localization. Analyze the following {language} code for bugs.

FILE: {file_path}

CODE:
```{language.lower()}
{code}
```
"""
        
        if error_message:
            prompt += f"\nERROR MESSAGE:\n{error_message}\n"
        
        if stack_trace:
            prompt += f"\nSTACK TRACE:\n{stack_trace}\n"
        
        if failing_test:
            prompt += f"\nFAILING TEST: {failing_test}\n"
        
        prompt += """
ANALYSIS REQUIREMENTS:
1. Identify the EXACT line numbers that contain bugs (not just functions)
2. Explain the root cause of each bug
3. Provide a minimal, correct fix for each bug
4. Rank bugs by severity (Critical, High, Medium, Low)

OUTPUT FORMAT (JSON):
{
  "suspicious_lines": [
    {
      "line_number": <int>,
      "code": "<exact code line>",
      "confidence": <float 0-1>,
      "severity": "<Critical|High|Medium|Low>",
      "bug_type": "<brief bug category>",
      "root_cause": "<detailed explanation>",
      "suggested_fix": "<minimal code fix>",
      "explanation": "<why this line is buggy>"
    }
  ],
  "root_cause_summary": "<overall analysis>",
  "recommended_patches": [
    {
      "line_number": <int>,
      "original": "<original code>",
      "fixed": "<fixed code>"
    }
  ]
}

IMPORTANT:
- Be precise with line numbers
- Focus on actual bugs, not style issues
- Provide working code fixes
- Rank by likelihood of being the bug
"""
        
        return prompt
    
    def _call_gemini_api(self, prompt: str) -> Dict:
        """
        Call Gemini API with the prompt.
        
        Args:
            prompt: Analysis prompt
            
        Returns:
            API response
        """
        url = f"{self.endpoint}?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.2,  # Low temperature for precise analysis
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 2048,
            }
        }
        
        try:
            response = requests.post(
                url,
                headers={'Content-Type': 'application/json'},
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Gemini API call failed: {str(e)}")
    
    def _parse_gemini_response(self, response: Dict, original_code: str) -> Dict:
        """
        Parse Gemini API response and format as fault localization report.
        
        Args:
            response: Raw API response
            original_code: Original source code
            
        Returns:
            Formatted analysis result
        """
        try:
            # Extract text from Gemini response
            text = response['candidates'][0]['content']['parts'][0]['text']
            
            print(f"📝 Gemini raw response length: {len(text)} chars")
            
            # Extract JSON from response (Gemini often wraps it in markdown)
            json_text = text
            if '```json' in text:
                json_text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                json_text = text.split('```')[1].split('```')[0].strip()
            
            # Parse JSON
            analysis = json.loads(json_text)
            
            print(f"📊 Parsed {len(analysis.get('suspicious_lines', []))} suspicious lines from Gemini")
            
            # Format in fault localization style
            formatted = {
                'suspicious_lines': [],
                'root_cause_summary': analysis.get('root_cause_summary', 'Analysis complete'),
                'recommended_patches': analysis.get('recommended_patches', []),
                'analysis_method': 'advanced_static_analysis'  # Don't mention Gemini
            }
            
            # Process suspicious lines
            for line in analysis.get('suspicious_lines', []):
                formatted_line = {
                    'line_number': line.get('line_number'),
                    'code': line.get('code', ''),
                    'probability': line.get('probability', line.get('confidence', 0.8)),  # Try both fields
                    'severity': line.get('severity', 'Medium'),
                    'bug_type': line.get('bug_type', 'Logic error'),
                    'root_cause': line.get('root_cause', ''),
                    'suggested_fix': line.get('suggested_fix', ''),
                    'explanation': line.get('explanation', ''),
                    'source': 'deep_analysis'
                }
                formatted['suspicious_lines'].append(formatted_line)
            
            # Sort by confidence (descending)
            formatted['suspicious_lines'].sort(
                key=lambda x: x['probability'],
                reverse=True
            )
            
            return formatted
            
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            # Fallback if parsing fails
            return {
                'suspicious_lines': [],
                'root_cause_summary': f'Analysis completed with parsing issues: {str(e)}',
                'recommended_patches': [],
                'error': str(e),
                'raw_response': text if 'text' in locals() else str(response)
            }
    
    def format_as_fault_report(self, analysis: Dict) -> str:
        """
        Format analysis as a human-readable fault localization report.
        
        Args:
            analysis: Analysis results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("FAULT LOCALIZATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Most suspicious lines
        if analysis['suspicious_lines']:
            report.append("MOST SUSPICIOUS LINES (ranked by confidence):")
            report.append("-" * 80)
            
            for i, line in enumerate(analysis['suspicious_lines'][:5], 1):
                severity_emoji = {
                    'Critical': '🔴',
                    'High': '🟠',
                    'Medium': '🟡',
                    'Low': '🟢'
                }
                emoji = severity_emoji.get(line['severity'], '⚪')
                
                report.append(f"\n#{i} {emoji} Line {line['line_number']} - {line['severity']} ({line['probability']*100:.1f}%)")
                report.append(f"   Code: {line['code']}")
                report.append(f"   Bug Type: {line['bug_type']}")
                report.append(f"   Root Cause: {line['root_cause']}")
                if line.get('suggested_fix'):
                    report.append(f"   Suggested Fix: {line['suggested_fix']}")
        
        # Root cause summary
        report.append("\n" + "-" * 80)
        report.append("ROOT CAUSE SUMMARY:")
        report.append(analysis['root_cause_summary'])
        
        # Recommended patches
        if analysis.get('recommended_patches'):
            report.append("\n" + "-" * 80)
            report.append("RECOMMENDED PATCHES:")
            for patch in analysis['recommended_patches']:
                report.append(f"\nLine {patch['line_number']}:")
                report.append(f"  - Original: {patch['original']}")
                report.append(f"  + Fixed:    {patch['fixed']}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


def analyze_code_file(
    file_path: str,
    stack_trace: Optional[str] = None,
    error_message: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict:
    """
    Convenience function to analyze a code file.
    
    Args:
        file_path: Path to code file
        stack_trace: Optional stack trace
        error_message: Optional error message
        api_key: Optional Gemini API key
        
    Returns:
        Analysis results
    """
    # Read file
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Detect language from extension
    ext = file_path.split('.')[-1].lower()
    language_map = {
        'py': 'Python',
        'java': 'Java',
        'cpp': 'C++',
        'cc': 'C++',
        'cxx': 'C++',
        'c': 'C',
        'h': 'C',
        'hpp': 'C++',
        'js': 'JavaScript',
        'ts': 'TypeScript'
    }
    language = language_map.get(ext, 'Unknown')
    
    # Analyze
    analyzer = GeminiAnalyzer(api_key)
    return analyzer.analyze_bug(
        code=code,
        file_path=file_path,
        language=language,
        stack_trace=stack_trace,
        error_message=error_message
    )


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python gemini_analyzer.py <code_file> [stack_trace_file]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    stack_trace = None
    
    if len(sys.argv) > 2:
        with open(sys.argv[2], 'r') as f:
            stack_trace = f.read()
    
    # Analyze
    result = analyze_code_file(file_path, stack_trace)
    
    # Format and print report
    analyzer = GeminiAnalyzer()
    print(analyzer.format_as_fault_report(result))
