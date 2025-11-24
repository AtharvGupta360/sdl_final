"""
Rule-Based Bug Pattern Detection for C/C++/Java/Python

This module contains hardcoded patterns for detecting common, famous bugs
without requiring any machine learning model training.

Based on:
- CWE (Common Weakness Enumeration) database
- OWASP Top 10
- Real-world bug datasets (Defects4J, CVE database)
- Famous bugs from Linux kernel, OpenSSL, etc.
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class BugPattern:
    """Represents a bug pattern with detection rules."""
    name: str
    severity: str  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    confidence: float  # 0.0 to 1.0
    description: str
    cwe_id: str  # CWE identifier
    pattern: str  # Regex pattern or keyword
    check_function: callable = None  # Custom checking function


class BugPatternDetector:
    """
    Detects famous bug patterns in code using hardcoded rules.
    No training required - purely rule-based detection.
    """
    
    def __init__(self):
        self.patterns = self._load_bug_patterns()
    
    def _load_bug_patterns(self) -> List[BugPattern]:
        """Load all hardcoded bug patterns."""
        return [
            # ============================================================
            # CRITICAL BUGS (Famous CVEs and Security Vulnerabilities)
            # ============================================================
            
            # CWE-476: NULL Pointer Dereference (Heartbleed-style)
            BugPattern(
                name="NULL Pointer Dereference",
                severity="CRITICAL",
                confidence=0.95,
                description="Dereferencing pointer without NULL check",
                cwe_id="CWE-476",
                pattern=r"(?:->|\*\s*\w+\s*=)(?!\s*NULL)",
                check_function=self._check_null_dereference
            ),
            
            # CWE-787: Buffer Overflow (Classic C bug)
            BugPattern(
                name="Buffer Overflow",
                severity="CRITICAL",
                confidence=0.92,
                description="Unsafe buffer operations (strcpy, sprintf, gets)",
                cwe_id="CWE-787",
                pattern=r"\b(strcpy|strcat|sprintf|gets|scanf)\s*\(",
                check_function=None
            ),
            
            # CWE-369: Divide by Zero
            BugPattern(
                name="Division by Zero",
                severity="CRITICAL",
                confidence=0.98,
                description="Division operation without denominator check",
                cwe_id="CWE-369",
                pattern=r"\/\s*[a-zA-Z_]\w*\s*[;\)]",
                check_function=self._check_divide_by_zero
            ),
            
            # CWE-125: Out-of-bounds Read
            BugPattern(
                name="Array Index Out of Bounds",
                severity="CRITICAL",
                confidence=0.88,
                description="Array access without bounds checking",
                cwe_id="CWE-125",
                pattern=r"\w+\s*\[\s*\w+\s*\]",
                check_function=self._check_array_bounds
            ),
            
            # CWE-415: Double Free (Use After Free)
            BugPattern(
                name="Double Free / Use After Free",
                severity="CRITICAL",
                confidence=0.90,
                description="Freeing memory twice or using freed memory",
                cwe_id="CWE-415",
                pattern=r"\bfree\s*\(",
                check_function=self._check_double_free
            ),
            
            # CWE-401: Memory Leak
            BugPattern(
                name="Memory Leak",
                severity="HIGH",
                confidence=0.85,
                description="Allocated memory not freed",
                cwe_id="CWE-401",
                pattern=r"\b(malloc|calloc|realloc|new)\s*\(",
                check_function=self._check_memory_leak
            ),
            
            # CWE-190: Integer Overflow
            BugPattern(
                name="Integer Overflow",
                severity="HIGH",
                confidence=0.82,
                description="Arithmetic operation without overflow check",
                cwe_id="CWE-190",
                pattern=r"(?:int|long|short|char)\s+\w+\s*=.*[\+\-\*]",
                check_function=self._check_integer_overflow
            ),
            
            # CWE-191: Integer Underflow
            BugPattern(
                name="Integer Underflow",
                severity="HIGH",
                confidence=0.80,
                description="Unsigned subtraction without underflow check",
                cwe_id="CWE-191",
                pattern=r"unsigned\s+\w+\s+\w+\s*=.*\-",
                check_function=None
            ),
            
            # CWE-704: Off-by-One Error
            BugPattern(
                name="Off-by-One Error",
                severity="HIGH",
                confidence=0.75,
                description="Loop boundary error (< vs <=)",
                cwe_id="CWE-704",
                pattern=r"for\s*\([^;]*;\s*\w+\s*(<|<=|>|>=)",
                check_function=self._check_off_by_one
            ),
            
            # CWE-78: OS Command Injection
            BugPattern(
                name="Command Injection",
                severity="CRITICAL",
                confidence=0.93,
                description="Unsafe system command execution",
                cwe_id="CWE-78",
                pattern=r"\b(system|exec|popen|ShellExecute)\s*\(",
                check_function=None
            ),
            
            # CWE-89: SQL Injection
            BugPattern(
                name="SQL Injection",
                severity="CRITICAL",
                confidence=0.90,
                description="Unsanitized SQL query construction",
                cwe_id="CWE-89",
                pattern=r"(?:SELECT|INSERT|UPDATE|DELETE).*\+.*(?:input|user|param|request)",
                check_function=None
            ),
            
            # CWE-20: Input Validation
            BugPattern(
                name="Missing Input Validation",
                severity="HIGH",
                confidence=0.70,
                description="User input used without validation",
                cwe_id="CWE-20",
                pattern=r"\b(scanf|gets|fgets|cin|input|request)\s*\(",
                check_function=self._check_input_validation
            ),
            
            # CWE-252: Unchecked Return Value
            BugPattern(
                name="Unchecked Return Value",
                severity="MEDIUM",
                confidence=0.78,
                description="Function return value not checked",
                cwe_id="CWE-252",
                pattern=r"\b(malloc|fopen|read|write|recv|send)\s*\([^;]*;(?!\s*if)",
                check_function=self._check_return_value
            ),
            
            # CWE-362: Race Condition (TOCTOU)
            BugPattern(
                name="Race Condition (TOCTOU)",
                severity="HIGH",
                confidence=0.75,
                description="Time-of-check to time-of-use race condition",
                cwe_id="CWE-362",
                pattern=r"\baccess\s*\(.*\bfopen\s*\(",
                check_function=None
            ),
            
            # CWE-327: Weak Cryptography
            BugPattern(
                name="Weak Cryptography",
                severity="HIGH",
                confidence=0.88,
                description="Using deprecated or weak crypto algorithms",
                cwe_id="CWE-327",
                pattern=r"\b(MD5|SHA1|DES|RC4|rand\(\))\b",
                check_function=None
            ),
            
            # CWE-798: Hardcoded Credentials
            BugPattern(
                name="Hardcoded Credentials",
                severity="CRITICAL",
                confidence=0.85,
                description="Passwords or keys hardcoded in source",
                cwe_id="CWE-798",
                pattern=r"(?:password|passwd|pwd|secret|apikey|api_key)\s*=\s*['\"][^'\"]+['\"]",
                check_function=None
            ),
            
            # CWE-732: Incorrect Permission Assignment
            BugPattern(
                name="Unsafe File Permissions",
                severity="MEDIUM",
                confidence=0.80,
                description="File created with overly permissive permissions",
                cwe_id="CWE-732",
                pattern=r"\bchmod\s*\([^,]*,\s*0?7[0-7]{2}\)",
                check_function=None
            ),
            
            # ============================================================
            # LOGIC BUGS (Famous patterns from real vulnerabilities)
            # ============================================================
            
            # Wrong comparison operators (== instead of =, etc.)
            BugPattern(
                name="Assignment in Condition",
                severity="HIGH",
                confidence=0.92,
                description="Assignment (=) used instead of comparison (==)",
                cwe_id="CWE-480",
                pattern=r"if\s*\([^=]*\s=[^=]",
                check_function=None
            ),
            
            # Inverted logic (like your auth.py bug!)
            BugPattern(
                name="Inverted Boolean Logic",
                severity="HIGH",
                confidence=0.88,
                description="Boolean logic appears inverted (== False instead of ==True)",
                cwe_id="CWE-670",
                pattern=r"(?:==\s*(?:False|false|FALSE|0)|!=\s*(?:True|true|TRUE|1))",
                check_function=self._check_inverted_logic
            ),
            
            # Missing break in switch
            BugPattern(
                name="Missing Break in Switch",
                severity="MEDIUM",
                confidence=0.75,
                description="Switch case without break (possible fall-through bug)",
                cwe_id="CWE-484",
                pattern=r"case\s+\w+:(?:(?!break|return).)*case",
                check_function=None
            ),
            
            # Uninitialized variables
            BugPattern(
                name="Uninitialized Variable",
                severity="HIGH",
                confidence=0.70,
                description="Variable used before initialization",
                cwe_id="CWE-457",
                pattern=r"(?:int|char|float|double|long)\s+\w+;",
                check_function=self._check_uninitialized
            ),
            
            # Signed/Unsigned mismatch
            BugPattern(
                name="Signed/Unsigned Mismatch",
                severity="MEDIUM",
                confidence=0.72,
                description="Comparing signed and unsigned integers",
                cwe_id="CWE-195",
                pattern=r"(?:unsigned.*==.*int\s+\w+|int\s+\w+.*==.*unsigned)",
                check_function=None
            ),
            
            # Resource leak (file not closed)
            BugPattern(
                name="Resource Leak",
                severity="MEDIUM",
                confidence=0.78,
                description="File/socket opened but not closed",
                cwe_id="CWE-404",
                pattern=r"\b(fopen|socket|open)\s*\(",
                check_function=self._check_resource_leak
            ),
            
            # Infinite loop potential
            BugPattern(
                name="Potential Infinite Loop",
                severity="MEDIUM",
                confidence=0.65,
                description="Loop variable not modified in loop body",
                cwe_id="CWE-835",
                pattern=r"while\s*\(\s*\d+\s*\)|while\s*\(\s*true\s*\)|while\s*\(\s*1\s*\)",
                check_function=None
            ),
        ]
    
    # ================================================================
    # CUSTOM CHECK FUNCTIONS (More sophisticated pattern detection)
    # ================================================================
    
    def _check_null_dereference(self, code: str, line: str) -> bool:
        """Check if pointer is dereferenced without NULL check."""
        # Look for -> or * without preceding if (ptr != NULL)
        lines = code.split('\n')
        for i, l in enumerate(lines):
            if l.strip() == line.strip():
                # Check previous 3 lines for NULL check
                prev_lines = '\n'.join(lines[max(0, i-3):i])
                if '->' in line or '*' in line:
                    if 'if' not in prev_lines or 'NULL' not in prev_lines:
                        return True
        return False
    
    def _check_divide_by_zero(self, code: str, line: str) -> bool:
        """Check if division happens without denominator check."""
        # Look for division by variable
        if '/' in line and not '//' in line and not '/*' in line:
            # Extract denominator
            match = re.search(r'\/\s*([a-zA-Z_]\w*)', line)
            if match:
                var = match.group(1)
                # Check if there's a zero check before this line
                lines = code.split('\n')
                for i, l in enumerate(lines):
                    if l.strip() == line.strip():
                        prev_lines = '\n'.join(lines[max(0, i-5):i])
                        # Look for if (var == 0) or if (var != 0) or if (!var)
                        if f'{var}' not in prev_lines or 'if' not in prev_lines:
                            return True
        return False
    
    def _check_array_bounds(self, code: str, line: str) -> bool:
        """Check if array access has bounds checking."""
        # Look for array[index] without bounds check
        match = re.search(r'(\w+)\s*\[\s*(\w+)\s*\]', line)
        if match:
            array_name, index = match.groups()
            # Check if there's bounds checking before
            lines = code.split('\n')
            for i, l in enumerate(lines):
                if l.strip() == line.strip():
                    prev_lines = '\n'.join(lines[max(0, i-5):i])
                    # Look for if (index < size) or similar
                    if index in prev_lines and any(op in prev_lines for op in ['<', '>', '<=', '>=']):
                        return False
            return True
        return False
    
    def _check_double_free(self, code: str, line: str) -> bool:
        """Check for potential double free."""
        if 'free' in line:
            # Look for multiple free() calls on same variable
            match = re.search(r'free\s*\(\s*(\w+)\s*\)', line)
            if match:
                var = match.group(1)
                # Count free(var) occurrences
                free_count = code.count(f'free({var})')
                free_count += code.count(f'free( {var})')
                free_count += code.count(f'free({var} )')
                if free_count > 1:
                    return True
        return False
    
    def _check_memory_leak(self, code: str, line: str) -> bool:
        """Check if allocated memory is freed."""
        alloc_match = re.search(r'(\w+)\s*=\s*(?:malloc|calloc|realloc|new)', line)
        if alloc_match:
            var = alloc_match.group(1)
            # Look for free(var) or delete var in subsequent code
            lines = code.split('\n')
            for i, l in enumerate(lines):
                if l.strip() == line.strip():
                    rest_code = '\n'.join(lines[i:])
                    if f'free({var})' not in rest_code and f'delete {var}' not in rest_code:
                        return True
        return False
    
    def _check_integer_overflow(self, code: str, line: str) -> bool:
        """Check for integer overflow in arithmetic."""
        if any(op in line for op in ['+', '-', '*']) and any(kw in line for kw in ['int', 'long', 'short']):
            # Check if there's overflow checking
            if 'INT_MAX' not in code and 'LONG_MAX' not in code:
                return True
        return False
    
    def _check_off_by_one(self, code: str, line: str) -> bool:
        """Check for off-by-one errors in loops."""
        # Look for for(i=0; i<=size; i++) - should be i<size
        if 'for' in line:
            # Check if loop uses <= with size/length
            if '<=' in line and any(word in line.lower() for word in ['size', 'length', 'len', 'count']):
                return True
        return False
    
    def _check_input_validation(self, code: str, line: str) -> bool:
        """Check if user input is validated."""
        if any(func in line for func in ['scanf', 'gets', 'fgets', 'cin', 'input']):
            # Look for validation in next few lines
            lines = code.split('\n')
            for i, l in enumerate(lines):
                if l.strip() == line.strip():
                    next_lines = '\n'.join(lines[i:min(len(lines), i+5)])
                    if 'if' not in next_lines or 'valid' not in next_lines.lower():
                        return True
        return False
    
    def _check_return_value(self, code: str, line: str) -> bool:
        """Check if function return value is checked."""
        critical_funcs = ['malloc', 'fopen', 'read', 'write', 'recv', 'send']
        for func in critical_funcs:
            if func in line and '=' in line:
                # Check if next line has if check
                lines = code.split('\n')
                for i, l in enumerate(lines):
                    if l.strip() == line.strip():
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            if not next_line.startswith('if'):
                                return True
        return False
    
    def _check_inverted_logic(self, code: str, line: str) -> bool:
        """Check for inverted boolean logic (like == False when should be == True)."""
        # This is the bug in your auth.py!
        if '== False' in line or '== false' in line or '!= True' in line or '!= true' in line:
            # Check context - if it's checking is_active, authenticated, enabled, etc.
            if any(word in line.lower() for word in ['active', 'authenticated', 'enabled', 'valid', 'authorized']):
                return True
        return False
    
    def _check_uninitialized(self, code: str, line: str) -> bool:
        """Check if variable is used before initialization."""
        # Look for declaration without initialization
        match = re.search(r'(?:int|char|float|double|long)\s+(\w+);', line)
        if match:
            var = match.group(1)
            # Check if variable is used before being assigned
            lines = code.split('\n')
            for i, l in enumerate(lines):
                if l.strip() == line.strip():
                    if i + 1 < len(lines):
                        next_lines = '\n'.join(lines[i+1:min(len(lines), i+10)])
                        # If var appears on right side of = before left side
                        if var in next_lines:
                            first_use = next_lines.find(var)
                            first_assign = next_lines.find(f'{var} =')
                            if first_use >= 0 and (first_assign < 0 or first_use < first_assign):
                                return True
        return False
    
    def _check_resource_leak(self, code: str, line: str) -> bool:
        """Check if file/socket is closed."""
        open_funcs = {'fopen': 'fclose', 'socket': 'close', 'open': 'close'}
        for open_func, close_func in open_funcs.items():
            if open_func in line and '=' in line:
                match = re.search(r'(\w+)\s*=\s*' + open_func, line)
                if match:
                    var = match.group(1)
                    # Look for close
                    if f'{close_func}({var})' not in code:
                        return True
        return False
    
    # ================================================================
    # MAIN DETECTION LOGIC
    # ================================================================
    
    def detect_bugs(self, code: str, line: str, file_path: str = '') -> List[Dict]:
        """
        Detect all bugs in a specific line of code.
        
        Args:
            code: Full source code (for context)
            line: Specific line to check
            file_path: File path (to determine language)
            
        Returns:
            List of detected bugs with details
        """
        detected_bugs = []
        
        for pattern in self.patterns:
            # Check if pattern matches
            if re.search(pattern.pattern, line, re.IGNORECASE):
                # If custom check function exists, use it
                if pattern.check_function:
                    try:
                        if not pattern.check_function(code, line):
                            continue
                    except:
                        pass  # If check fails, still report the pattern match
                
                detected_bugs.append({
                    'name': pattern.name,
                    'severity': pattern.severity,
                    'confidence': pattern.confidence,
                    'description': pattern.description,
                    'cwe_id': pattern.cwe_id,
                    'line': line.strip(),
                    'recommendation': self._get_recommendation(pattern.name)
                })
        
        return detected_bugs
    
    def _get_recommendation(self, bug_name: str) -> str:
        """Get fix recommendation for bug type."""
        recommendations = {
            'NULL Pointer Dereference': 'Add NULL check: if (ptr != NULL) before dereferencing',
            'Buffer Overflow': 'Use safe functions: strncpy, snprintf, fgets with size limits',
            'Division by Zero': 'Add check: if (denominator != 0) before division',
            'Array Index Out of Bounds': 'Add bounds check: if (index < array_size)',
            'Double Free / Use After Free': 'Set pointer to NULL after free(); check before freeing',
            'Memory Leak': 'Add free() or delete before function returns',
            'Integer Overflow': 'Check INT_MAX before arithmetic; use larger type or saturating arithmetic',
            'Integer Underflow': 'Check if result would be negative before unsigned subtraction',
            'Off-by-One Error': 'Use < instead of <= for array size; check loop bounds',
            'Command Injection': 'Sanitize input; use parameterized commands; avoid system()',
            'SQL Injection': 'Use prepared statements with parameter binding',
            'Missing Input Validation': 'Validate all user input; check length, format, range',
            'Unchecked Return Value': 'Check return value: if (result == NULL) or if (result < 0)',
            'Race Condition (TOCTOU)': 'Use atomic operations; file locks; avoid access()+open()',
            'Weak Cryptography': 'Use modern crypto: SHA-256+, AES, cryptographically secure RNG',
            'Hardcoded Credentials': 'Use environment variables or secure key management',
            'Unsafe File Permissions': 'Use restrictive permissions: 0600 for sensitive files',
            'Assignment in Condition': 'Use == for comparison, not =; enable compiler warnings',
            'Inverted Boolean Logic': 'Check logic: should be == True or != False',
            'Missing Break in Switch': 'Add break; or /* fall through */ comment if intentional',
            'Uninitialized Variable': 'Initialize at declaration: int x = 0;',
            'Signed/Unsigned Mismatch': 'Cast to same type; use consistent signedness',
            'Resource Leak': 'Add fclose(), close(), or proper cleanup in all code paths',
            'Potential Infinite Loop': 'Ensure loop variable is modified; add exit condition',
        }
        return recommendations.get(bug_name, 'Review and fix this issue')
    
    def calculate_bug_score(self, bugs: List[Dict]) -> float:
        """
        Calculate overall bug probability based on detected patterns.
        
        Args:
            bugs: List of detected bugs
            
        Returns:
            Probability score (0.0 to 1.0)
        """
        if not bugs:
            return 0.15  # Base score for no detected bugs
        
        # Weighted scoring based on severity and confidence
        severity_weights = {
            'CRITICAL': 0.40,
            'HIGH': 0.25,
            'MEDIUM': 0.15,
            'LOW': 0.08
        }
        
        total_score = 0.15  # Base score
        
        for bug in bugs:
            weight = severity_weights.get(bug['severity'], 0.10)
            confidence = bug['confidence']
            total_score += weight * confidence
        
        # Cap at 0.98 (never 100% certain)
        return min(total_score, 0.98)
