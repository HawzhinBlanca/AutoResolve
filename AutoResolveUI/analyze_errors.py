#!/usr/bin/env python3
"""
Analyze Swift build errors in AutoResolve project
"""

import re
from collections import defaultdict, Counter

def analyze_build_errors(file_path):
    """Analyze patterns in build errors"""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract error and warning lines
    error_lines = re.findall(r'error: (.+)', content)
    warning_lines = re.findall(r'warning: (.+)', content)
    
    # Pattern analysis
    patterns = {
        'missing_properties': defaultdict(list),
        'foreach_issues': [],
        'type_conversion': [],
        'missing_types': [],
        'color_grading': [],
        'sendable_issues': [],
        'async_await': [],
        'ambiguous_types': []
    }
    
    # Analyze each error
    for error in error_lines:
        # Missing properties pattern
        if 'has no member' in error:
            match = re.search(r"value of type '(.+)' has no member '(.+)'", error)
            if match:
                type_name, member = match.groups()
                patterns['missing_properties'][type_name].append(member)
        
        # ForEach issues
        elif 'ForEach' in error and ('Binding' in error or 'generic parameter' in error):
            patterns['foreach_issues'].append(error)
            
        # Type conversion issues
        elif 'cannot convert' in error or 'cannot infer' in error:
            patterns['type_conversion'].append(error)
            
        # Missing types
        elif 'cannot find' in error and 'in scope' in error:
            patterns['missing_types'].append(error)
            
        # Color grading specific
        elif 'ColorWheel' in error or 'ColorGrading' in error:
            patterns['color_grading'].append(error)
            
        # Sendable issues
        elif 'Sendable' in error:
            patterns['sendable_issues'].append(error)
            
        # Async/await issues  
        elif 'async' in error.lower() or 'await' in error.lower():
            patterns['async_await'].append(error)
            
        # Ambiguous types
        elif 'ambiguous' in error.lower():
            patterns['ambiguous_types'].append(error)
    
    return patterns, len(error_lines), len(warning_lines)

def main():
    print("=== AutoResolve Swift Build Error Analysis ===\n")
    
    try:
        patterns, error_count, warning_count = analyze_build_errors('/Users/hawzhin/AutoResolve/AutoResolveUI/build_output.txt')
    except FileNotFoundError:
        print("Error: build_output.txt not found")
        return
    
    print(f"Total Errors: {error_count}")
    print(f"Total Warnings: {warning_count}\n")
    
    print("=== ERROR PATTERN ANALYSIS ===\n")
    
    # 1. Missing Properties on Types
    print("1. MISSING PROPERTIES ON TYPES")
    print("-" * 40)
    for type_name, members in patterns['missing_properties'].items():
        member_counts = Counter(members)
        print(f"Type: {type_name}")
        for member, count in member_counts.most_common():
            print(f"  - Missing '{member}': {count} occurrences")
        print()
    
    # 2. ForEach Issues
    print("2. FOREACH SYNTAX ISSUES")
    print("-" * 30)
    foreach_unique = set(patterns['foreach_issues'])
    for issue in list(foreach_unique)[:10]:  # Show top 10 unique issues
        print(f"  - {issue}")
    print(f"Total ForEach issues: {len(patterns['foreach_issues'])}\n")
    
    # 3. Type Conversion Issues
    print("3. TYPE CONVERSION ISSUES")
    print("-" * 32)
    conversion_unique = set(patterns['type_conversion'])
    for issue in list(conversion_unique)[:10]:
        print(f"  - {issue}")
    print(f"Total conversion issues: {len(patterns['type_conversion'])}\n")
    
    # 4. Missing Type Definitions
    print("4. MISSING TYPE DEFINITIONS")
    print("-" * 34)
    missing_unique = set(patterns['missing_types'])
    for issue in list(missing_unique)[:10]:
        print(f"  - {issue}")
    print(f"Total missing type issues: {len(patterns['missing_types'])}\n")
    
    # 5. Color Grading Issues
    print("5. COLOR GRADING SPECIFIC ISSUES")
    print("-" * 40)
    color_unique = set(patterns['color_grading'])
    for issue in list(color_unique)[:10]:
        print(f"  - {issue}")
    print(f"Total color grading issues: {len(patterns['color_grading'])}\n")
    
    # Summary
    print("=== SUMMARY BY CATEGORY ===")
    print("-" * 35)
    category_counts = {
        'Missing Properties': len(patterns['missing_properties']),
        'ForEach Issues': len(patterns['foreach_issues']),
        'Type Conversion': len(patterns['type_conversion']),
        'Missing Types': len(patterns['missing_types']),
        'Color Grading': len(patterns['color_grading']),
        'Sendable Issues': len(patterns['sendable_issues']),
        'Async/Await': len(patterns['async_await']),
        'Ambiguous Types': len(patterns['ambiguous_types'])
    }
    
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"{category}: {count}")

if __name__ == "__main__":
    main()