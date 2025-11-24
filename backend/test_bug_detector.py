"""Quick test of bug pattern detector"""

from bug_patterns import BugPatternDetector

detector = BugPatternDetector()
print(f"✅ Loaded {len(detector.patterns)} bug patterns\n")

# Test with famous bugs
test_cases = [
    ("strcpy(dest, src);", "Buffer Overflow - strcpy"),
    ("if (x = 5)", "Assignment in condition"),
    ("return a / b;", "Division by zero"),
    ("free(ptr); *ptr = 5;", "Use after free"),
]

print("Testing bug detection:")
print("=" * 70)

for code, expected in test_cases:
    detected = detector.detect_bugs("", code, "test.c")
    if detected:
        print(f"\n✓ '{code}'")
        for bug in detected:
            print(f"  → {bug['name']} ({bug['severity']})")
    else:
        print(f"\n✗ '{code}' - No bugs detected")

print("\n" + "=" * 70)
print("All tests complete!")
