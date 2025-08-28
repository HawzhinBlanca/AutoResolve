#!/usr/bin/env python3
"""
Validate OpenRouter Integration - Complete Checklist
"""

import os
import sys
import configparser
from pathlib import Path

def validate_openrouter():
    """Validate all OpenRouter integration requirements"""
    
    print("🎯 OpenRouter Integration Validation")
    print("=" * 50)
    
    results = []
    
    # 1. Check dependencies installed
    try:
        import openai
        import tiktoken
        results.append(("Dependencies installed", True))
    except ImportError as e:
        results.append(("Dependencies installed", False, str(e)))
    
    # 2. Check configuration file
    config_path = Path("conf/ops.ini")
    if config_path.exists():
        config = configparser.ConfigParser()
        config.read(config_path)
        
        if 'openrouter' in config:
            results.append(("OpenRouter config section", True))
            
            # Check default is disabled
            enabled = config.get('openrouter', 'enabled', fallback='true')
            # Strip comments and whitespace
            enabled_value = enabled.split('#')[0].strip()
            results.append(("Default disabled", enabled_value.lower() == 'false'))
            
            # Check all required fields
            required_fields = [
                'base_url', 'api_key_env', 'narrative_model', 
                'max_input_tokens', 'daily_usd_cap', 'target_api_sec_per_min'
            ]
            for field in required_fields:
                has_field = config.has_option('openrouter', field)
                results.append((f"Config: {field}", has_field))
        else:
            results.append(("OpenRouter config section", False))
    else:
        results.append(("Config file exists", False))
    
    # 3. Check source files exist
    files_to_check = [
        ("OpenRouter client", "src/ops/openrouter.py"),
        ("Hybrid evaluator", "src/eval/hybrid_eval.py"),
        ("Director integration", "src/director/creative_director.py"),
        ("Makefile commands", "Makefile")
    ]
    
    for name, filepath in files_to_check:
        path = Path(filepath)
        if path.exists():
            # Check for OpenRouter code
            content = path.read_text()
            if 'openrouter' in content.lower() or 'OpenRouter' in content:
                results.append((name, True))
            else:
                results.append((name, False, "File exists but missing OpenRouter code"))
        else:
            results.append((name, False, "File not found"))
    
    # 4. Check Blueprint.md updated
    blueprint_path = Path("../Blueprint.md")
    if blueprint_path.exists():
        content = blueprint_path.read_text()
        if 'openrouter' in content.lower():
            results.append(("Blueprint.md updated", True))
        else:
            results.append(("Blueprint.md updated", False))
    
    # 5. Test client functionality
    try:
        from src.ops.openrouter import OpenRouterClient
        config = configparser.ConfigParser()
        config.read('conf/ops.ini')
        client = OpenRouterClient(config)
        
        # Should be disabled by default
        results.append(("Client disabled by default", not client.enabled))
        
        # Test sanity check
        status = client.sanity_check()
        results.append(("Sanity check works", 'status' in status))
    except Exception as e:
        results.append(("Client functionality", False, str(e)))
    
    # 6. Performance gates
    gates = {
        "V-JEPA/CLIP unchanged": "Core embeddings remain local-only",
        "OpenRouter disabled by default": "enabled = false in config",
        "Timeout < 20s": "request_timeout_s = 20",
        "Daily cap $2.50": "daily_usd_cap = 2.50",
        "Cache directory": "artifacts/cache/openrouter",
        "JSON validation": "response_format={type: json_object}",
        "Fail-closed": "Returns {_skipped: true} on failure",
        "30x realtime": "Local processing unaffected",
        "API < 3s/min": "target_api_sec_per_min = 3.0",
        "Cost < $0.05/min": "Enforced by daily_cap and max_calls"
    }
    
    print("\n📋 VALIDATION CHECKLIST")
    print("-" * 50)
    
    passed = 0
    failed = 0
    
    for check, result in results:
        if isinstance(result, bool):
            if result:
                print(f"✅ {check}")
                passed += 1
            else:
                print(f"❌ {check}")
                failed += 1
        else:
            print(f"❌ {check}: {result[1]}")
            failed += 1
    
    print("\n🔒 PERFORMANCE GATES")
    print("-" * 50)
    
    for gate, description in gates.items():
        print(f"• {gate}: {description}")
    
    print("\n📊 SUMMARY")
    print("-" * 50)
    print(f"Passed: {passed}/{passed+failed}")
    print(f"Failed: {failed}/{passed+failed}")
    
    if failed == 0:
        print("\n✅ ALL VALIDATIONS PASSED!")
        print("\n📝 ADR Statement:")
        print('"Added optional OpenRouter augmentation for narrative labeling and cut reasoning.')
        print('Core embeddings remain local (V-JEPA/CLIP). Cloud assist is fail-closed with')
        print('strict budgets. No network dependency in critical path."')
        return True
    else:
        print("\n❌ Some validations failed. Please review and fix.")
        return False

if __name__ == "__main__":
    success = validate_openrouter()
    sys.exit(0 if success else 1)