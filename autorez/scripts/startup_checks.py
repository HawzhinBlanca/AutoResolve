#!/usr/bin/env python3
"""
AutoResolve v3.2 - Startup Health Checks
Comprehensive system validation before production startup
"""

import os
import sys
import json
import time
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to Python path for module imports
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up from scripts/ to autorez/
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)  # Add autorez directory to allow 'src.' imports

def print_status(name: str, passed: bool, details: str = ""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{name:<30} {status}")
    if details and not passed:
        print(f"  └─ {details}")

def check_dependencies() -> bool:
    """Check all required dependencies are available"""
    # Map package names to import names
    deps_map = {
        'torch': 'torch',
        'transformers': 'transformers', 
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'numpy': 'numpy',
        'av': 'av',
        'pillow': 'PIL',  # pillow imports as PIL
        'psutil': 'psutil',
        'ffmpeg-python': 'ffmpeg',
        'faster-whisper': 'faster_whisper'
    }
    
    for pkg_name, import_name in deps_map.items():
        try:
            __import__(import_name)
        except ImportError:
            print_status(f"Dependency: {pkg_name}", False, f"Import failed")
            return False
    
    return True

def check_file_structure() -> bool:
    """Verify critical file structure"""
    critical_files = [
        'backend_service_final.py',
        'autoresolve_cli.py', 
        'src/eval/gates.py',
        'src/ops/silence.py',
        'conf/embeddings.ini',
        'conf/director.ini', 
        'conf/ops.ini',
        'datasets/broll_pilot/manifest.json',
        'assets/test_30min.mp4',
        'requirements.txt',
        'Makefile',
        'deploy_final.sh'
    ]
    
    missing = []
    for file in critical_files:
        if not os.path.exists(file):
            missing.append(file)
    
    if missing:
        print_status("File Structure", False, f"Missing: {', '.join(missing)}")
        return False
    
    return True

def check_configuration() -> bool:
    """Validate configuration files"""
    import configparser
    
    configs = {
        'conf/embeddings.ini': ['embedders'],
        'conf/director.ini': ['global', 'narrative', 'emotion', 'rhythm'],
        'conf/ops.ini': ['silence', 'transcribe']
    }
    
    for config_file, required_sections in configs.items():
        try:
            cfg = configparser.ConfigParser()
            cfg.read(config_file)
            for section in required_sections:
                if section not in cfg.sections():
                    print_status(f"Config: {config_file}", False, f"Missing section [{section}]")
                    return False
        except Exception as e:
            print_status(f"Config: {config_file}", False, str(e))
            return False
    
    return True

def check_performance_gates() -> bool:
    """Verify performance gates pass"""
    try:
        from src.eval.gates import verify_gates
        
        # Load metrics
        with open('artifacts/metrics.json') as f:
            metrics = json.load(f)
        
        verify_gates(metrics)
        return True
    except Exception as e:
        print_status("Performance Gates", False, str(e))
        return False

def check_embedders() -> bool:
    """Test embedder modules load correctly"""
    try:
        from src.embedders.vjepa_embedder import VJEPAEmbedder
        from src.embedders.clip_embedder import CLIPEmbedder
        
        # Quick instantiation test
        vjepa = VJEPAEmbedder()
        clip = CLIPEmbedder()
        return True
    except Exception as e:
        print_status("Embedders", False, str(e))
        return False

def check_pipeline_modules() -> bool:
    """Test all pipeline modules import correctly"""
    modules = [
        'src.ops.silence',
        'src.ops.transcribe', 
        'src.director.creative_director',
        'src.broll.selector',
        'src.ops.timeline_manager'
    ]
    
    for module in modules:
        try:
            __import__(module)
        except ImportError as e:
            print_status(f"Module: {module}", False, str(e))
            return False
    
    return True

def check_security() -> bool:
    """Security validation"""
    # Check for dangerous function calls (excluding legitimate model.eval())
    dangerous_patterns = [
        r'\beval\s*\(',  # eval( but not model.eval()
        r'\bexec\s*\(',  # exec(
        r'__import__\s*\(',  # __import__(
        r'\bcompile\s*\('  # compile(
    ]
    issues = []
    
    import re
    for file_path in Path('src').rglob('*.py'):
        try:
            with open(file_path) as f:
                content = f.read()
                for pattern in dangerous_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        # Skip model.eval() which is safe
                        if 'model.eval(' not in content or pattern != r'\beval\s*\(':
                            issues.append(f"{file_path}: {pattern}")
        except:
            continue
    
    if issues:
        print_status("Security Scan", False, f"Dangerous patterns: {', '.join(issues)}")
        return False
    
    return True

def check_disk_space() -> bool:
    """Check available disk space"""
    try:
        import shutil
        total, used, free = shutil.disk_usage('.')
        free_gb = free // (2**30)
        
        if free_gb < 2:
            print_status("Disk Space", False, f"Only {free_gb}GB free, need 2GB+")
            return False
            
        return True
    except Exception as e:
        print_status("Disk Space", False, str(e))
        return False

def check_memory() -> bool:
    """Check system memory"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        
        if available_gb < 4.0:
            print_status("Memory", False, f"Only {available_gb:.1f}GB available, need 4GB+")
            return False
            
        return True
    except Exception as e:
        print_status("Memory", False, str(e))
        return False

def check_ports() -> bool:
    """Check required ports are available"""
    import socket
    
    def is_port_open(port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('localhost', port))
            sock.close()
            return True
        except:
            return False
    
    if not is_port_open(8000):
        print_status("Port 8000", False, "Port already in use")
        return False
    
    return True

def run_comprehensive_checks() -> bool:
    """Run all startup checks"""
    print("🚀 AutoResolve v3.2 - Startup Health Checks")
    print("=" * 50)
    
    checks = [
        ("System Memory", check_memory),
        ("Disk Space", check_disk_space),
        ("Port Availability", check_ports),
        ("Dependencies", check_dependencies),
        ("File Structure", check_file_structure),
        ("Configuration", check_configuration),
        ("Security", check_security),
        ("Pipeline Modules", check_pipeline_modules),
        ("Embedders", check_embedders),
        ("Performance Gates", check_performance_gates)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            print_status(name, result)
            results.append((name, result))
        except Exception as e:
            print_status(name, False, str(e))
            results.append((name, False))
    
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL CHECKS PASSED ({passed}/{total})")
        print("System ready for production startup!")
        return True
    else:
        print(f"❌ CHECKS FAILED ({passed}/{total})")
        print("Fix issues before starting production!")
        return False

if __name__ == "__main__":
    success = run_comprehensive_checks()
    sys.exit(0 if success else 1)