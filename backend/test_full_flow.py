"""
Full end-to-end test: Simulate exactly what the frontend sends
"""

import requests
import json

# Simulate frontend form submission
payload = {
    "repository_path": "C:\\Users\\gupta\\test_cpp_bug",
    "failing_test": "test_find",
    "error_message": "division by zero",
    "stack_trace": "File \"calculator.cpp\", line 7",
    "max_candidates": 50
}

print("Sending request to backend...")
print(f"Repository: {payload['repository_path']}")
print(f"Test: {payload['failing_test']}")
print(f"Error: {payload['error_message']}")
print("="*60)

try:
    response = requests.post(
        "http://localhost:8000/api/predict",
        json=payload,
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"\n✅ Got {len(result['ranked_lines'])} results:")
        print("="*60)
        
        # Show top 5 results
        for line in result['ranked_lines'][:5]:
            prob = line['probability'] * 100
            print(f"\nRank #{line['rank']}: Line {line['line_number']} - {prob:.1f}%")
            print(f"  File: {line['file']}")
            print(f"  Code: {line['code']}")
            
            # Check if backend added bug info
            if 'features' in line and isinstance(line['features'], dict):
                if 'bug_reason' in line['features']:
                    print(f"  Reason: {line['features']['bug_reason'][:100]}...")
        
        print("\n" + "="*60)
        print("✅ If you see high probability (>80%) for the bug line, it's working!")
        print("❌ If all show low probability (~12%), Gemini is not being called")
        
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        
except Exception as e:
    print(f"❌ Request failed: {e}")
    print("\n⚠️ Make sure backend is running: python main.py")
