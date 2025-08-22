# Provider/model selection is config-only. Do not import OpenAI/Azure SDKs here directly.

"""Streamlit app entry point for Utilities Knowledge Assistant."""

import sys
import os
from pathlib import Path
import types

# HOTFIX: Prevent Streamlit file-watcher from crashing on torch.classes.__path__
# This must run BEFORE importing streamlit or any app code that loads torch
try:
    import torch
    if 'torch.classes' not in sys.modules:
        # Create harmless stub to prevent watcher from dereferencing magic __path__
        sys.modules['torch.classes'] = types.ModuleType('torch.classes')
        sys.modules['torch.classes'].__path__ = []  # Empty package for watcher
except Exception:
    pass  # Safe to ignore if torch not available

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Now run the chat interface
from app.chat_interface import main

if __name__ == "__main__":
    main()