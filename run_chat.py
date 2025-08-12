#!/usr/bin/env python3
"""Main entry point for the refactored chat interface."""

import sys
import os
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    try:
        # Import and run the chat interface
        from app.chat_interface import main
        main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're in the virtual environment:")
        print("   source .venv/bin/activate")
        print("   streamlit run run_chat.py")
    except Exception as e:
        print(f"❌ Error starting chat interface: {e}")
        raise