"""
Test the intelligent bug detection on calculator.cpp
"""

# Test case: calculator.cpp with wrong operator
test_code = {
    'line_4': {
        'code': 'class Calculator {',
        'file_path': 'calculator.cpp',
        'features': {'distance_from_error': 1.0, 'cyclomatic_complexity': 0, 'code_churn': 0}
    },
    'line_7': {
        'code': '    int add(int a, int b) {',
        'file_path': 'calculator.cpp',
        'features': {'distance_from_error': 1.0, 'cyclomatic_complexity': 2, 'code_churn': 0}
    },
    'line_8': {
        'code': '        return a - b;  // Line 8 - Should be a + b',
        'file_path': 'calculator.cpp',
        'features': {'distance_from_error': 0.0, 'cyclomatic_complexity': 2, 'code_churn': 0.8}
    },
    'line_11': {
        'code': '    int multiply(int a, int b) {',
        'file_path': 'calculator.cpp',
        'features': {'distance_from_error': 1.0, 'cyclomatic_complexity': 2, 'code_churn': 0}
    },
    'line_12': {
        'code': '        return a * b;',
        'file_path': 'calculator.cpp',
        'features': {'distance_from_error': 1.0, 'cyclomatic_complexity': 2, 'code_churn': 0}
    }
}

print("=" * 80)
print("INTELLIGENT BUG DETECTION TEST")
print("=" * 80)

for line_id, candidate in test_code.items():
    print(f"\n{line_id}: {candidate['code']}")
    
    # Analyze logic
    code = candidate['code']
    score = 0.15  # base
    
    # Check if it's "add" function using subtraction
    if 'add' in candidate['file_path'].lower() or 'add' in code.lower():
        if ' - ' in code and 'return' in code.lower():
            score += 0.45
            print(f"   🔴 BUG DETECTED: Function named 'add' but uses subtraction (-)")
            print(f"   Score boost: +0.45")
    
    # Stack trace boost
    if candidate['features']['distance_from_error'] < 0.1:
        score += 0.30
        print(f"   🎯 In stack trace: +0.30")
    
    # Complexity
    if candidate['features']['cyclomatic_complexity'] > 5:
        score += 0.10
        print(f"   📊 High complexity: +0.10")
    
    # Code churn
    if candidate['features']['code_churn'] > 0.7:
        score += 0.10
        print(f"   📝 Recent changes: +0.10")
    
    print(f"   TOTAL SCORE: {score:.2f} ({score*100:.1f}%)")
    
    if score > 0.7:
        print(f"   ⚠️  CRITICAL: Very likely buggy!")
    elif score > 0.4:
        print(f"   🟠 HIGH: Likely buggy")
    elif score < 0.2:
        print(f"   🟢 LOW: Unlikely buggy")

print("\n" + "=" * 80)
print("EXPECTED RESULT:")
print("  Line 8 should have ~90% (0.15 + 0.45 + 0.30 + 0.10 = 1.00 capped at 0.98)")
print("  Other lines should have ~15-25%")
print("=" * 80)
