# cache.py module - TTL Cache implementation
from cachetools import TTLCache

# Create a TTL cache with a maximum of 5 entries and a time-to-live (TTL) of 10 seconds.
cache = TTLCache(maxsize=1, ttl=60)

def add_to_cache(key, value):
    """Add content to the cache."""
    cache[key] = value
    print(f"Added key '{key}' with value '{value}' to cache.")

def get_from_cache(key):
    """Retrieve content from the cache; returns None if key is not found or expired."""
    value = cache.get(key, None)
    if value is None:
        print(f"Key '{key}' not found or expired.")
    else:
        print(f"Retrieved key '{key}': '{value}'")
    return value

# Example usage:
if __name__ == '__main__':
    # Add an item to the cache
    # add_to_cache('user_query', 'AU')
    
    # Retrieve the item immediately
    get_from_cache('user_query')