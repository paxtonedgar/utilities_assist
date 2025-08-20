#!/usr/bin/env python3
"""
Setup script for BGE v2-m3 Reranker

This script helps you get the local reranker up and running with all dependencies
and model files properly configured.

Usage:
    python setup_reranker.py [--check-only]
"""

import os
import sys
import subprocess
from pathlib import Path

def print_status(message, status="info"):
    """Print colored status message."""
    colors = {
        "info": "\033[94m",      # Blue
        "success": "\033[92m",   # Green  
        "warning": "\033[93m",   # Yellow
        "error": "\033[91m",     # Red
        "reset": "\033[0m"       # Reset
    }
    
    icons = {
        "info": "ℹ️",
        "success": "✅", 
        "warning": "⚠️",
        "error": "❌"
    }
    
    color = colors.get(status, colors["info"])
    icon = icons.get(status, "•")
    reset = colors["reset"]
    
    print(f"{color}{icon} {message}{reset}")


def check_python_version():
    """Check Python version compatibility."""
    print_status("Checking Python version...")
    
    version = sys.version_info
    if version.major != 3 or version.minor < 11:
        print_status(f"Python {version.major}.{version.minor} detected. Python 3.11+ recommended for best compatibility.", "warning")
        return False
    else:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} ✓", "success")
        return True


def check_git_lfs():
    """Check if Git LFS is installed and working."""
    print_status("Checking Git LFS...")
    
    try:
        result = subprocess.run(["git", "lfs", "version"], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip().split()[0]
            print_status(f"Git LFS {version} ✓", "success")
            return True
        else:
            print_status("Git LFS not found. Install with: brew install git-lfs", "error")
            return False
    except FileNotFoundError:
        print_status("Git LFS not found. Install with: brew install git-lfs", "error")
        return False


def check_model_files():
    """Check if BGE model files are present."""
    print_status("Checking BGE v2-m3 model files...")
    
    model_path = Path("models/bge-reranker-v2-m3")
    required_files = [
        "config.json",
        "tokenizer.json", 
        "tokenizer_config.json"
    ]
    
    if not model_path.exists():
        print_status("Model directory not found. Run: git lfs pull", "error")
        return False
    
    missing_files = []
    for file in required_files:
        file_path = model_path / file
        if not file_path.exists():
            missing_files.append(file)
    
    # Check for model weights - either extracted or zipped
    model_file = model_path / "model.safetensors"
    zip_file = model_path / "model-weights.zip"
    
    if model_file.exists():
        # Check if it's a proper file (not LFS pointer)
        if model_file.stat().st_size < 1000:
            missing_files.append("model.safetensors (LFS pointer, run: git lfs pull)")
        else:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print_status(f"Model weights extracted ({size_mb:.1f} MB) ✓", "success")
    elif zip_file.exists():
        # Check if zip is proper file (not LFS pointer)  
        if zip_file.stat().st_size < 1000:
            missing_files.append("model-weights.zip (LFS pointer, run: git lfs pull)")
        else:
            size_mb = zip_file.stat().st_size / (1024 * 1024)
            print_status(f"Model weights zipped ({size_mb:.1f} MB) - will auto-extract ✓", "success")
    else:
        missing_files.append("model.safetensors or model-weights.zip")
    
    if missing_files:
        print_status(f"Missing model files: {', '.join(missing_files)}", "error")
        return False
    
    print_status("All required model files present ✓", "success")
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    print_status("Checking Python dependencies...")
    
    required_deps = [
        ("torch", "2.2.2"),
        ("sentence_transformers", "3.0.1"),
        ("pydantic", "2.0"),
        ("numpy", "1.26")  # Should be <2.0
    ]
    
    missing_deps = []
    version_issues = []
    
    for dep, min_version in required_deps:
        try:
            if dep == "sentence_transformers":
                import sentence_transformers
                version = sentence_transformers.__version__
            elif dep == "torch":
                import torch
                version = torch.__version__
            elif dep == "pydantic":
                import pydantic
                version = pydantic.__version__
            elif dep == "numpy":
                import numpy
                version = numpy.__version__
                # Check numpy version constraint
                if version.startswith("2."):
                    version_issues.append(f"{dep} {version} (should be <2.0 for stability)")
                    continue
            else:
                continue
                
            print_status(f"{dep} {version} ✓", "success")
            
        except ImportError:
            missing_deps.append(f"{dep}>={min_version}")
    
    if missing_deps:
        print_status(f"Missing dependencies: {', '.join(missing_deps)}", "error")
        print_status("Install with: pip install -r requirements.txt", "info")
        return False
        
    if version_issues:
        for issue in version_issues:
            print_status(issue, "warning")
        print_status("Consider: pip install 'numpy<2.0'", "info")
    
    return True


def check_device_support():
    """Check device support for acceleration."""
    print_status("Checking device support...")
    
    try:
        import torch
        
        devices = []
        
        if torch.backends.mps.is_available():
            devices.append("MPS (Metal)")
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            devices.append(f"CUDA ({device_count} device{'s' if device_count > 1 else ''})")
        
        devices.append("CPU")
        
        print_status(f"Available devices: {', '.join(devices)}", "success")
        
        # Recommend best device
        if torch.backends.mps.is_available():
            print_status("Recommended device: MPS (optimal for M1/M2 Macs)", "info")
        elif torch.cuda.is_available():
            print_status("Recommended device: CUDA", "info")
        else:
            print_status("Will use CPU (slower but works)", "info")
            
        return True
        
    except ImportError:
        print_status("Cannot check device support (torch not installed)", "warning")
        return False


def run_smoke_test():
    """Run the smoke test to verify everything works."""
    print_status("Running smoke test...")
    
    try:
        result = subprocess.run([sys.executable, "test_bge_reranker.py"], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print_status("Smoke test passed ✓", "success")
            # Print last few lines of output
            lines = result.stdout.strip().split('\n')[-3:]
            for line in lines:
                if "✅" in line:
                    print_status(line.replace("✅", "").strip(), "success")
            return True
        else:
            print_status("Smoke test failed", "error")
            # Print error details
            error_lines = result.stderr.strip().split('\n')[-5:]
            for line in error_lines:
                if line.strip():
                    print_status(f"  {line}", "error")
            return False
            
    except subprocess.TimeoutExpired:
        print_status("Smoke test timed out (>5 minutes)", "error")
        return False
    except Exception as e:
        print_status(f"Failed to run smoke test: {e}", "error")
        return False


def print_next_steps(all_passed):
    """Print next steps based on check results."""
    print("\n" + "="*50)
    
    if all_passed:
        print_status("🎉 Setup complete! BGE v2-m3 reranker is ready to use.", "success")
        print_status("You can now:", "info")
        print_status("  • Run your application with reranking enabled", "info")
        print_status("  • Set RERANKER_ENABLED=true in your environment", "info")
        print_status("  • Configure device with RERANKER_DEVICE=auto|mps|cuda|cpu", "info")
    else:
        print_status("Setup incomplete. Please fix the issues above.", "error")
        print_status("Common solutions:", "info")
        print_status("  • Install dependencies: pip install -r requirements.txt", "info")
        print_status("  • Download model files: git lfs pull", "info")
        print_status("  • Install Git LFS: brew install git-lfs", "info")


def main():
    """Main setup function."""
    print("🚀 BGE v2-m3 Reranker Setup")
    print("="*50)
    
    check_only = "--check-only" in sys.argv
    
    # Run all checks
    checks = [
        ("Python Version", check_python_version),
        ("Git LFS", check_git_lfs), 
        ("Model Files", check_model_files),
        ("Dependencies", check_dependencies),
        ("Device Support", check_device_support)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n📋 {name}:")
        result = check_func()
        results.append(result)
    
    all_passed = all(results)
    
    # Run smoke test if all checks pass and not check-only mode
    if all_passed and not check_only:
        print(f"\n📋 Smoke Test:")
        smoke_test_result = run_smoke_test()
        all_passed = all_passed and smoke_test_result
    
    # Print summary
    print_next_steps(all_passed)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())