"""
Diagnostic script to test why Gemini isn't being called properly
"""

from ai_model_simple import FaultLocalizationModel

# Create model
print("Creating model...")
model = FaultLocalizationModel()

# Test candidates (simulating what the backend receives)
test_candidates = [
    {
        'file_path': 'C:\\Users\\gupta\\test_cpp_bug\\calculator.cpp',
        'line_number': 4,
        'code': 'int main() {',
        'features': {}
    },
    {
        'file_path': 'C:\\Users\\gupta\\test_cpp_bug\\calculator.cpp',
        'line_number': 5,
        'code': '    int a = 10;',
        'features': {}
    },
    {
        'file_path': 'C:\\Users\\gupta\\test_cpp_bug\\calculator.cpp',
        'line_number': 6,
        'code': '    int b = 0;',
        'features': {}
    },
    {
        'file_path': 'C:\\Users\\gupta\\test_cpp_bug\\calculator.cpp',
        'line_number': 7,
        'code': '    cout << a / b; // crash',
        'features': {}
    }
]

print("\nCalling model.predict()...")
print("="*60)

results = model.predict(test_candidates)

print("\nRESULTS:")
print("="*60)

for result in results:
    prob = result.get('probability', 0) * 100
    line = result.get('line_number')
    code = result.get('code', '')
    reason = result.get('bug_reason', 'N/A')
    fix = result.get('suggested_fix', 'N/A')
    
    print(f"\nLine {line}: {prob:.1f}%")
    print(f"  Code: {code}")
    print(f"  Reason: {reason}")
    print(f"  Fix: {fix}")

print("\n" + "="*60)
print("✅ If you see high probability (>80%) for line 7, Gemini is working!")
print("❌ If all lines show ~12%, Gemini is NOT being called")
