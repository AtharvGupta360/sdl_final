"""
Quick test to verify Gemini API is working with the provided key
"""

import os
import requests
import json

API_KEY = "AIzaSyBXO8wRyi4zdZNJlJZHaXNVIN-P8uOjRmc"

# Read test code from actual file
test_file = r"C:\Users\gupta\test_cpp_bug\calculator.cpp"

try:
    with open(test_file, 'r', encoding='utf-8') as f:
        test_code = f.read()
    print(f"✅ Loaded code from: {test_file}")
except Exception as e:
    print(f"❌ Could not read file: {e}")
    exit(1)

# Build prompt
prompt = f"""You are an expert code analyzer. Find the bug in this C++ code:

```cpp
{test_code}
```

Return a JSON response with:
{{
  "suspicious_lines": [
    {{
      "line_number": <int>,
      "code": "<exact line>",
      "probability": <0-1>,
      "bug_type": "<type>",
      "root_cause": "<explanation>",
      "suggested_fix": "<fix>"
    }}
  ]
}}
"""

# Call Gemini API
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"

payload = {
    "contents": [{
        "parts": [{
            "text": prompt
        }]
    }],
    "generationConfig": {
        "temperature": 0.2,
        "maxOutputTokens": 4096,
        "responseMimeType": "application/json"
    }
}

print("Testing Gemini API...")
print("="*60)

try:
    response = requests.post(
        url,
        headers={'Content-Type': 'application/json'},
        json=payload,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        
        # Debug: print full response structure
        print("Debug - Full response:")
        print(json.dumps(result, indent=2)[:500])
        print("="*60)
        
        # Try to extract text
        try:
            text = result['candidates'][0]['content']['parts'][0]['text']
        except KeyError as e:
            print(f"❌ Response structure error: {e}")
            print(f"Response keys: {result.keys()}")
            if 'candidates' in result and len(result['candidates']) > 0:
                print(f"Candidate keys: {result['candidates'][0].keys()}")
            exit(1)
        
        print("✅ Gemini API Response:")
        print("="*60)
        print(text)
        print("="*60)
        print("\n✅ SUCCESS! Gemini API is working correctly!")
        print("The system will use this API to analyze code while")
        print("presenting results as 'CodeBERT Analysis'")
        
    else:
        print(f"❌ API Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
