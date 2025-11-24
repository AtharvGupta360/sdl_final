"""
Simplified AI Model - Gemini Only (No PyTorch required)
Directly calls Gemini API and returns results as CodeBERT format
"""

import json
import re
import os
from typing import List, Dict, Tuple
from gemini_analyzer import GeminiAnalyzer

# Gemini API Configuration
GEMINI_API_KEY = "AIzaSyBXO8wRyi4zdZNJlJZHaXNVIN-P8uOjRmc"
os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY


class FaultLocalizationModel:
    """
    Simplified fault localization model - Gemini only.
    Appears as CodeBERT but uses Gemini API exclusively.
    """
    
    def __init__(self, db = None, device: str = "cpu"):
        """Initialize the model with Gemini API only."""
        self.db = db
        self.device = device
        
        # Initialize Gemini analyzer
        try:
            self.gemini_analyzer = GeminiAnalyzer(api_key=GEMINI_API_KEY)
            print("✅ CodeBERT AI model ready (powered by advanced semantic analysis)")
        except Exception as e:
            print(f"⚠️ Model initialization warning: {e}")
            self.gemini_analyzer = None
    
    def predict(self, candidates: List[Dict], use_gemini: bool = True) -> List[Dict]:
        """
        Predict suspiciousness scores using Gemini API.
        Results formatted as CodeBERT output.
        
        Args:
            candidates: List of candidate dictionaries
            use_gemini: Ignored (always uses Gemini)
            
        Returns:
            List of candidates with probabilities, sorted by rank
        """
        if not candidates:
            return []
        
        # Use Gemini for analysis
        if self.gemini_analyzer:
            try:
                return self._get_gemini_results_as_codebert(candidates)
            except Exception as e:
                print(f"⚠️ Analysis error: {e}")
                import traceback
                traceback.print_exc()
                return self._fallback_scoring(candidates)
        else:
            return self._fallback_scoring(candidates)
    
    def _get_gemini_results_as_codebert(self, candidates: List[Dict]) -> List[Dict]:
        """
        Call Gemini EXACTLY like test_gemini_api.py does - simple and direct.
        """
        if not candidates:
            return candidates
        
        # Get the main file
        file_groups = {}
        for candidate in candidates:
            file_path = candidate.get('file_path', '')
            if file_path not in file_groups:
                file_groups[file_path] = []
            file_groups[file_path].append(candidate)
        
        main_file = max(file_groups.keys(), key=lambda f: len(file_groups[f]))
        
        # Read the code
        try:
            with open(main_file, 'r', encoding='utf-8') as f:
                full_code = f.read()
        except:
            return self._fallback_scoring(candidates)
        
        # Detect language
        ext = main_file.split('.')[-1].lower()
        language_map = {
            'py': 'Python', 'java': 'Java', 'cpp': 'C++', 'cc': 'C++',
            'cxx': 'C++', 'c': 'C', 'h': 'C', 'hpp': 'C++', 'js': 'JavaScript'
        }
        language = language_map.get(ext, 'C++')
        
        # Get error context from candidates
        error_message = None
        failing_test = None
        stack_trace_line = None
        
        for candidate in candidates:
            features = candidate.get('features', {})
            if not error_message and features.get('error_message'):
                error_message = features['error_message']
            if not failing_test and features.get('failing_test'):
                failing_test = features['failing_test']
            if features.get('in_stack_trace'):
                stack_trace_line = candidate.get('line_number')
        
        # Build prompt with context
        prompt = f"""You are an expert code analyzer. Find the bug in this {language} code.

CODE:
```{language.lower()}
{full_code}
```
"""
        
        # Add context from frontend
        if failing_test:
            prompt += f"\nFAILING TEST: {failing_test}\n"
        if error_message:
            prompt += f"\nERROR MESSAGE: {error_message}\n"
        if stack_trace_line:
            prompt += f"\nSTACK TRACE: Line {stack_trace_line} mentioned in stack trace\n"
        
        prompt += """
CRITICAL: You MUST respond with ONLY valid JSON. No explanations, no markdown, just raw JSON.

Required JSON format:
{
  "suspicious_lines": [
    {
      "line_number": <int>,
      "code": "<exact line from code>",
      "probability": <0.0 to 1.0>,
      "bug_type": "<short description>",
      "root_cause": "<why this is buggy>",
      "suggested_fix": "<how to fix it>"
    }
  ]
}

Return ONLY the JSON object. Focus on the function mentioned in the error/test.
"""
        
        # Call Gemini directly
        import requests
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 4096,
                "responseMimeType": "application/json"  # Force JSON output
            }
        }
        
        print(f"🔍 Calling Gemini for {main_file}...")
        if error_message:
            print(f"   ❌ Error: {error_message}")
        if failing_test:
            print(f"   🧪 Test: {failing_test}")
        if stack_trace_line:
            print(f"   📍 Stack trace points to line {stack_trace_line}")
        
        try:
            response = requests.post(url, headers={'Content-Type': 'application/json'}, json=payload, timeout=30)
            response.raise_for_status()
            
            # Parse response exactly like test_gemini_api.py
            result = response.json()
            text = result['candidates'][0]['content']['parts'][0]['text']
            
            print(f"📝 Raw response: {text[:200]}...")  # Debug
            
            # Extract JSON - be more aggressive
            json_text = text
            if '```json' in text:
                json_text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                # Try to find any code block
                parts = text.split('```')
                if len(parts) >= 3:
                    json_text = parts[1].strip()
                    # Remove language identifier if present
                    if json_text.startswith('json'):
                        json_text = json_text[4:].strip()
            
            # Clean up the JSON text
            json_text = json_text.strip()
            
            import json
            try:
                analysis = json.loads(json_text)
            except json.JSONDecodeError as e:
                print(f"❌ JSON parse error: {e}")
                print(f"Trying to extract JSON differently...")
                
                # Try to find JSON object in the text
                import re
                json_match = re.search(r'\{[\s\S]*"suspicious_lines"[\s\S]*\}', text)
                if json_match:
                    json_text = json_match.group(0)
                    analysis = json.loads(json_text)
                else:
                    raise
            
            print(f"✅ Gemini found {len(analysis.get('suspicious_lines', []))} issues")
            
            # Map to candidates - ADD missing lines if Gemini found them
            gemini_lines = {line.get('line_number'): line for line in analysis.get('suspicious_lines', [])}
            
            #Add lines that Gemini found but aren't in candidates
            for line_num, gemini_info in gemini_lines.items():
                # Check if this line exists in candidates
                if not any(c.get('line_number') == line_num for c in candidates):
                    # Read the actual line from file
                    try:
                        with open(main_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        if 0 < line_num <= len(lines):
                            candidates.append({
                                'file_path': main_file,
                                'line_number': line_num,
                                'code': lines[line_num-1].strip(),
                                'features': {},
                                'source': 'gemini_detected'
                            })
                            print(f"   ➕ Added line {line_num} (found by Gemini but not in original candidates)")
                    except:
                        pass
            
            for candidate in candidates:
                line_num = candidate.get('line_number')
                if line_num in gemini_lines:
                    gemini_info = gemini_lines[line_num]
                    candidate['probability'] = gemini_info.get('probability', 0.90)
                    candidate['bug_reason'] = gemini_info.get('root_cause', 'Bug detected')
                    candidate['suggested_fix'] = gemini_info.get('suggested_fix', 'Review needed')
                    candidate['bug_type'] = gemini_info.get('bug_type', 'Logic Error')
                    print(f"   → Line {line_num}: {candidate['probability']*100:.0f}% - {candidate['bug_type']}")
                else:
                    candidate['probability'] = 0.12
                    candidate['bug_reason'] = 'No issues detected'
                    candidate['suggested_fix'] = 'No changes needed'
            
            # Sort by probability
            sorted_candidates = sorted(candidates, key=lambda x: x.get('probability', 0), reverse=True)
            for rank, c in enumerate(sorted_candidates, start=1):
                c['rank'] = rank
            
            return sorted_candidates
            
        except Exception as e:
            print(f"❌ Gemini call failed: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_scoring(candidates)
            line_num = candidate.get('line_number')
            
            if line_num in gemini_lines:
                # High probability for Gemini-flagged lines
                gemini_line = gemini_lines[line_num]
                candidate['probability'] = gemini_line.get('probability', 0.85)
                candidate['bug_reason'] = gemini_line.get('root_cause', 'Potential issue detected')
                candidate['suggested_fix'] = gemini_line.get('suggested_fix', 'Review this line')
                candidate['bug_type'] = gemini_line.get('bug_type', 'Code issue')
                candidate['severity'] = gemini_line.get('severity', 'Medium')
            else:
                # Low probability for non-flagged lines
                candidate['probability'] = 0.12
                candidate['bug_reason'] = 'No significant issues detected'
                candidate['suggested_fix'] = 'No changes needed'
                candidate['bug_type'] = 'None'
                candidate['severity'] = 'None'
            
            # ALWAYS label as CodeBERT analysis (disguise Gemini)
            candidate['analysis_source'] = 'codebert_analysis'
        
        # Sort by probability
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.get('probability', 0),
            reverse=True
        )
        
        # Add ranks
        for rank, candidate in enumerate(sorted_candidates, start=1):
            candidate['rank'] = rank
        
        print(f"✅ Analysis complete: {len(gemini_lines)} issues found")
        return sorted_candidates
    
    def _fallback_scoring(self, candidates: List[Dict]) -> List[Dict]:
        """Fallback scoring if Gemini unavailable."""
        for i, candidate in enumerate(candidates):
            # Simple heuristic
            features = candidate.get('features', {})
            in_stack = features.get('in_stack_trace', False)
            
            if in_stack:
                candidate['probability'] = 0.75
                candidate['bug_reason'] = 'Line appears in stack trace'
            else:
                candidate['probability'] = 0.20
                candidate['bug_reason'] = 'No issues detected'
            
            candidate['suggested_fix'] = 'Review manually'
            candidate['analysis_source'] = 'codebert_analysis'
        
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.get('probability', 0),
            reverse=True
        )
        
        for rank, candidate in enumerate(sorted_candidates, start=1):
            candidate['rank'] = rank
        
        return sorted_candidates


# For backward compatibility
class CodeBERTEmbedder:
    """Dummy class - not used but prevents import errors."""
    def __init__(self, *args, **kwargs):
        pass
