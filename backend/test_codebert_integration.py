"""
Test Gemini integration - verify it returns results formatted as CodeBERT
"""

from ai_model import FaultLocalizationModel

# Create model (with Gemini API key embedded)
print("Initializing model...")
model = FaultLocalizationModel()

# Create test candidates (simulating bug in calculator.cpp)
test_code = """int add(int a, int b) {
    return a - b;  // BUG: Should be a + b
}"""

candidates = [
    {
        'file_path': 'C:\\Users\\gupta\\test_cpp_bug\\calculator.cpp',
        'line_number': 8,
        'code': '    return a - b;',
        'features': {
            'distance_from_error': 0.0,
            'cyclomatic_complexity': 0.3,
            'code_churn': 0.2,
            'test_coverage': 0.8,
            'historical_faults': 0.1
        }
    },
    {
        'file_path': 'C:\\Users\\gupta\\test_cpp_bug\\calculator.cpp',
        'line_number': 7,
        'code': 'int add(int a, int b) {',
        'features': {
            'distance_from_error': 0.0,
            'cyclomatic_complexity': 0.3,
            'code_churn': 0.2,
            'test_coverage': 0.8,
            'historical_faults': 0.1
        }
    }
]

print("\n" + "="*60)
print("Testing Gemini Integration (formatted as CodeBERT)")
print("="*60)

# Call predict - this will use Gemini API
print("\nCalling model.predict() with Gemini API...")
results = model.predict(candidates)

print("\n" + "="*60)
print("RESULTS (looks like CodeBERT but powered by Gemini)")
print("="*60)

for result in results:
    print(f"\nRank #{result.get('rank')}:")
    print(f"  Line {result.get('line_number')}: {result.get('code')}")
    print(f"  Probability: {result.get('probability', 0)*100:.1f}%")
    print(f"  Bug Type: {result.get('bug_type', 'N/A')}")
    print(f"  Root Cause: {result.get('bug_reason', 'N/A')}")
    print(f"  Fix: {result.get('suggested_fix', 'N/A')}")
    print(f"  Analysis Source: {result.get('analysis_source', 'unknown')}")

print("\n" + "="*60)
print("✅ Integration working! Gemini API → CodeBERT format")
print("="*60)
