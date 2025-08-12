# utils.py - Utility functions for configuration loading (phase1 compatibility)
import os
import configparser

def load_config():
    """Load configuration from config.ini file.
    
    This function provides compatibility with the legacy config.ini system
    while working with the new phase1 architecture.
    """
    # Check for UTILITIES_CONFIG environment variable
    config_file = os.getenv('UTILITIES_CONFIG', 'config.local.ini')
    
    # If it's a relative path, look in the current directory first, then src/
    if not os.path.isabs(config_file):
        if os.path.exists(config_file):
            file_path = config_file
        elif os.path.exists(f'src/{config_file}'):
            file_path = f'src/{config_file}'
        else:
            # Fallback to relative path
            file_path = config_file
    else:
        file_path = config_file
    
    llm_config = configparser.ConfigParser()
    
    try:
        llm_config.read(file_path)
        print(f"✅ Loaded config from: {file_path}")
        return llm_config
    except Exception as e:
        print(f"❌ Failed to load config from {file_path}: {e}")
        # Return empty config to prevent crashes
        return configparser.ConfigParser()