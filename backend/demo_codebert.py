"""
Complete Example: Using Gemini for Deep Bug Analysis

This demonstrates the full workflow:
1. Provide code + optional stack trace
2. Gemini performs deep analysis  
3. Output formatted as fault localization report
"""

import os
import sys

# Set your Gemini API key here or in environment
# Get free key at: https://makersuite.google.com/app/apikey
os.environ['GEMINI_API_KEY'] = 'YOUR_API_KEY_HERE'  # Replace with your key

from gemini_analyzer import GeminiAnalyzer

# Example 1: Calculator with wrong operator
calculator_code = """#include <iostream>
using namespace std;

class Calculator {
public:
    // BUG: Wrong operator used!
    int add(int a, int b) {
        return a - b;  // Line 8 - Should be a + b
    }

    int multiply(int a, int b) {
        return a * b;
    }
};

int main() {
    Calculator calc;
    int result = calc.add(5, 3);

    if (result != 8) {
        cout << "ERROR: Expected 8, got " << result << endl;
        return 1;
    }

    cout << "Test passed!" << endl;
    return 0;
}"""

# Example 2: Python validation bug
validation_code = """'''User authentication module with a bug'''

class User:
    def __init__(self, username, password, is_active=False):
        self.username = username
        self.password = password
        self.is_active = is_active

def validate_user(user):
    # BUG: Wrong logic here!
    if user.is_active == False:
        return True
    return False

def login(username, password):
    user = User(username, password, is_active=True)
    if not validate_user(user):
        raise ValueError("User not active")
    return user
"""

def main():
    print("=" * 80)
    print("GEMINI-POWERED FAULT LOCALIZATION DEMO")
    print("=" * 80)
    
    # Check if API key is set
    if not os.getenv('GEMINI_API_KEY') or os.getenv('GEMINI_API_KEY') == 'YOUR_API_KEY_HERE':
        print("\n❌ ERROR: GEMINI_API_KEY not set!")
        print("\nTo use this demo:")
        print("1. Get free API key: https://makersuite.google.com/app/apikey")
        print("2. Set environment variable:")
        print("   PowerShell: $env:GEMINI_API_KEY='your-key-here'")
        print("   Or edit this file and replace YOUR_API_KEY_HERE")
        return
    
    try:
        analyzer = GeminiAnalyzer()
        
        # Test 1: C++ Calculator Bug
        print("\n" + "=" * 80)
        print("TEST 1: C++ Wrong Operator Bug")
        print("=" * 80)
        
        result1 = analyzer.analyze_bug(
            code=calculator_code,
            file_path='calculator.cpp',
            language='C++',
            stack_trace='calculator.cpp:8',
            error_message='Expected 8, got 2'
        )
        
        print(analyzer.format_as_fault_report(result1))
        
        # Test 2: Python Validation Bug
        print("\n" + "=" * 80)
        print("TEST 2: Python Validation Logic Bug")
        print("=" * 80)
        
        result2 = analyzer.analyze_bug(
            code=validation_code,
            file_path='auth.py',
            language='Python',
            error_message='Active user cannot login'
        )
        
        print(analyzer.format_as_fault_report(result2))
        
        print("\n" + "=" * 80)
        print("✅ DEMO COMPLETE!")
        print("\nKey Features Demonstrated:")
        print("  ✓ Exact line number identification")
        print("  ✓ Root cause explanation")
        print("  ✓ Minimal fix suggestions")
        print("  ✓ Severity ranking")
        print("  ✓ Multiple language support")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  • Check internet connection")
        print("  • Verify API key is correct")
        print("  • Check API quotas at https://makersuite.google.com/")


if __name__ == '__main__':
    main()
