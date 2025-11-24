"""
Direct test bypassing HTTP - test the AI model directly
"""

from ai_model_simple import FaultLocalizationModel
from feature_extraction import FeatureExtractor

print("="*60)
print("DIRECT MODEL TEST")
print("="*60)

# Create model
print("\n1. Creating AI model...")
model = FaultLocalizationModel()

# Extract candidates like the backend does
print("\n2. Extracting candidates from repository...")
extractor = FeatureExtractor("C:\\Users\\gupta\\test_cpp_bug")
candidates = extractor.extract_candidate_lines(
    failing_test="test_user_llop",
    error_message="infinite loop",
    stack_trace=None,
    max_candidates=50
)

print(f"   Found {len(candidates)} candidates")

# Show what we extracted
print("\n3. Candidates extracted:")
for c in candidates[:5]:
    print(f"   Line {c['line_number']}: {c['code']}")

# Call model.predict
print("\n4. Calling model.predict()...")
print("="*60)

results = model.predict(candidates)

print("\n5. RESULTS:")
print("="*60)

for r in results[:5]:
    prob = r.get('probability', 0) * 100
    print(f"\n#{r.get('rank')} - Line {r.get('line_number')}: {prob:.1f}%")
    print(f"   Code: {r.get('code')}")
    print(f"   Reason: {r.get('bug_reason', 'N/A')[:100]}")

print("\n" + "="*60)
if any(r.get('probability', 0) > 0.8 for r in results):
    print("✅ Gemini is working! High probability found.")
else:
    print("❌ Gemini NOT working! All low probabilities - check backend logs for errors")
