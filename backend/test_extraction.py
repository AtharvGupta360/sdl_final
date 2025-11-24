"""
Test what lines are being extracted
"""
from feature_extraction import FeatureExtractor

fe = FeatureExtractor('C:\\Users\\gupta\\test_cpp_bug')
candidates = fe.extract_candidate_lines('test_find', 'division by zero')

print(f'Found {len(candidates)} candidates:')
for c in candidates:
    print(f'  Line {c["line_number"]}: {c["code"]}')