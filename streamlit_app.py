# Provider/model selection is config-only. Do not import OpenAI/Azure SDKs here directly.

"""Streamlit app entry point for Utilities Knowledge Assistant."""

import sys
import os
from pathlib import Path

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