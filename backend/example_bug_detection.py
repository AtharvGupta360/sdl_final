"""
Example: Testing Pretrained CodeBERT + Bug Patterns

This demonstrates how the system detects famous bugs without any training.
"""

import sys
sys.path.append('.')

from bug_patterns import BugPatternDetector
from ai_model import FaultLocalizationModel

# Initialize
detector = BugPatternDetector()
print(f"✅ Loaded {len(detector.patterns)} hardcoded bug patterns\n")

# Test famous bugs
test_bugs = [
    {
        "name": "Heartbleed-style Buffer Overflow",
        "code": "strcpy(password, user_input);",
        "file": "auth.c"
    },
    {
        "name": "Assignment in Condition",
        "code": "if (user.isActive = true)",
        "file": "validate.py"
    },
    {
        "name": "Division by Zero",
        "code": "int result = a / b;",
        "file": "calculator.c"
    },
    {
        "name": "Use After Free",
        "code": "free(ptr); *ptr = 0;",
        "file": "memory.c"
    },
    {
        "name": "NULL Pointer Dereference",
        "code": "char *ptr = malloc(100); *ptr = 'A';",
        "file": "alloc.c"
    },
    {
        "name": "Format String Vulnerability",
        "code": "printf(user_string);",
        "file": "log.c"
    },
]

print("=" * 80)
print("Testing Famous Bug Detection (No Training Required!)")
print("=" * 80)

for i, bug in enumerate(test_bugs, 1):
    print(f"\n{i}. {bug['name']}")
    print(f"   File: {bug['file']}")
    print(f"   Code: {bug['code']}")
    print("   " + "-" * 70)
    
    # Detect bugs
    detected = detector.detect_bugs("", bug['code'], bug['file'])
    
    if detected:
        for d in detected:
            severity_emoji = {
                'CRITICAL': '🔴',
                'HIGH': '🟠',
                'MEDIUM': '🟡',
                'LOW': '🟢'
            }
            emoji = severity_emoji.get(d['severity'], '⚪')
            
            print(f"   {emoji} DETECTED: {d['name']} ({d['severity']})")
            print(f"      CWE: {d['cwe_id']}")
            print(f"      Description: {d['description']}")
            print(f"      Confidence: {d['confidence']:.0%}")
    else:
        print("   ✗ No bugs detected (may need additional patterns)")

print("\n" + "=" * 80)
print("✅ All tests complete - No training needed!")
print("\nThese patterns are based on:")
print("  • CWE (Common Weakness Enumeration) database")
print("  • Real-world CVE vulnerabilities")
print("  • Famous bugs from Linux kernel, OpenSSL, Heartbleed, etc.")
print("  • OWASP Top 10 security issues")
print("=" * 80)
