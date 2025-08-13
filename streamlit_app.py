# Provider/model selection is config-only. Do not import OpenAI/Azure SDKs here directly.

"""Streamlit app entry point for Utilities Knowledge Assistant."""

import sys
import os
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# STARTUP DIAGNOSTIC - Remove after debugging
from utils import load_config
from infra.config import get_settings

print("\n=== STARTUP DIAGNOSTIC ===")
print("CLOUD_PROFILE:", repr(os.getenv("CLOUD_PROFILE")))
print("UTILITIES_CONFIG:", repr(os.getenv("UTILITIES_CONFIG")))
print("OPENSEARCH_ENDPOINT:", repr(os.getenv("OPENSEARCH_ENDPOINT")))
print("OPENSEARCH_INDEX:", repr(os.getenv("OPENSEARCH_INDEX")))
print("OS_HOST:", repr(os.getenv("OS_HOST")))

cfg = load_config()
print("Has [aws_info]:", cfg.has_section("aws_info"))
if cfg.has_section("aws_info"):
    print("config.ini opensearch_endpoint:", repr(cfg.get("aws_info","opensearch_endpoint", fallback="MISSING")))
    print("config.ini index_name:", repr(cfg.get("aws_info","index_name", fallback="MISSING")))

settings = get_settings()
print("settings.profile:", getattr(settings, "profile", None))
print("settings.search.host:", getattr(settings.search, "host", None))
print("settings.search.index_alias:", getattr(settings.search, "index_alias", None))
print("=== END STARTUP DIAGNOSTIC ===\n")

# Now run the chat interface
from app.chat_interface import main

if __name__ == "__main__":
    main()