"""
LangGraph persistence management with checkpointing and user memory store.

Provides production-ready checkpointers and stores with proper authentication integration.
Supports both in-memory (development) and database (production) persistence.
"""

import logging
import os
import sqlite3
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def get_checkpointer_and_store() -> Tuple[Optional[Any], Optional[Any]]:
    """
    Get appropriate checkpointer and store based on environment configuration.

    Returns:
        Tuple of (checkpointer, store) - both may be None for stateless operation
    """
    try:
        # Check if persistence is enabled
        enable_persistence = (
            os.getenv("ENABLE_LANGGRAPH_PERSISTENCE", "true").lower() == "true"
        )
        if not enable_persistence:
            logger.info("LangGraph persistence disabled via environment variable")
            return None, None

        # Determine persistence backend (default to memory for enterprise deployments)
        persistence_backend = os.getenv(
            "LANGGRAPH_PERSISTENCE_BACKEND", "memory"
        ).lower()

        if persistence_backend == "postgres":
            return _create_postgres_persistence()
        elif persistence_backend == "sqlite":
            logger.info("SQLite requested but may not be available in deployment")
            return _create_sqlite_persistence()
        elif persistence_backend == "memory":
            return _create_memory_persistence()
        else:
            logger.warning(
                f"Unknown persistence backend: {persistence_backend}, falling back to memory"
            )
            return _create_memory_persistence()

    except Exception as e:
        logger.error(f"Failed to initialize persistence: {e}")
        logger.info("Falling back to memory persistence")
        return _create_memory_persistence()


def _create_memory_persistence() -> Tuple[Any, Any]:
    """Create in-memory checkpointer and store for development."""
    try:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.store.memory import InMemoryStore

        checkpointer = InMemorySaver()
        store = InMemoryStore()

        logger.info("Initialized in-memory persistence (development mode)")
        return checkpointer, store

    except ImportError as e:
        logger.error(f"Failed to import in-memory persistence: {e}")
        return None, None


def _create_sqlite_persistence() -> Tuple[Any, Any]:
    """Create SQLite-based checkpointer and store."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        # Create data directory if it doesn't exist
        data_dir = Path("data/langgraph")
        data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite database
        db_path = data_dir / "checkpoints.db"
        conn = sqlite3.connect(str(db_path), check_same_thread=False)

        checkpointer = SqliteSaver(conn)
        checkpointer.setup()

        # For now, use in-memory store with SQLite checkpointer
        # (SQLite store is not commonly available)
        try:
            from langgraph.store.memory import InMemoryStore

            store = InMemoryStore()
        except ImportError:
            store = None

        logger.info(f"Initialized SQLite persistence: {db_path}")
        return checkpointer, store

    except ImportError as e:
        logger.warning(f"SQLite persistence not available: {e}")
        logger.info("Falling back to memory persistence")
        return _create_memory_persistence()
    except Exception as e:
        logger.error(f"Failed to create SQLite persistence: {e}")
        return _create_memory_persistence()


def _create_postgres_persistence() -> Tuple[Any, Any]:
    """Create PostgreSQL-based checkpointer and store for production."""
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from langgraph.store.postgres import PostgresStore

        # Get database connection string
        db_uri = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
        if not db_uri:
            # Build from individual components
            host = os.getenv("POSTGRES_HOST", "localhost")
            port = os.getenv("POSTGRES_PORT", "5432")
            user = os.getenv("POSTGRES_USER", "postgres")
            password = os.getenv("POSTGRES_PASSWORD", "")
            database = os.getenv("POSTGRES_DB", "utilities_assist")

            db_uri = f"postgresql://{user}:{password}@{host}:{port}/{database}"

        # Initialize PostgreSQL persistence
        checkpointer = PostgresSaver.from_conn_string(db_uri)
        store = PostgresStore.from_conn_string(db_uri)

        # Setup database tables
        checkpointer.setup()
        store.setup()

        logger.info("Initialized PostgreSQL persistence for production")
        return checkpointer, store

    except ImportError as e:
        logger.warning(f"PostgreSQL persistence not available: {e}")
        logger.info("Falling back to SQLite persistence")
        return _create_sqlite_persistence()
    except Exception as e:
        logger.error(f"Failed to create PostgreSQL persistence: {e}")
        return _create_sqlite_persistence()


def extract_user_context(resources) -> Dict[str, Any]:
    """
    Extract user context from JPMC authentication and resources.

    Args:
        resources: RAGResources containing authentication info

    Returns:
        Dict containing user context information
    """
    try:
        user_context = {}

        # Extract user info from environment (JPMC specific)
        user_sid = os.getenv("JPMC_USER_SID", "unknown")
        if user_sid != "REPLACE" and user_sid != "unknown":
            user_context["jpmc_user_sid"] = user_sid
            user_context["user_id"] = user_sid  # Use SID as user ID
        else:
            # Fallback user ID for development
            user_context["user_id"] = os.getenv("DEV_USER_ID", "dev_user_1")

        # Extract additional context from resources
        if resources and hasattr(resources, "settings"):
            user_context["profile"] = resources.settings.profile
            user_context["resource_age"] = resources.get_age_seconds()

        # Add session metadata
        user_context["session_metadata"] = {
            "cloud_profile": os.getenv("CLOUD_PROFILE", "local"),
            "utilities_config": os.getenv("UTILITIES_CONFIG", "config.local.ini"),
        }

        return user_context

    except Exception as e:
        logger.warning(f"Failed to extract user context: {e}")
        return {"user_id": "fallback_user", "error": str(e)}


def generate_thread_id(user_id: str, session_context: Optional[Dict] = None) -> str:
    """
    Generate a thread ID for conversation persistence.

    Args:
        user_id: User identifier
        session_context: Optional session information

    Returns:
        Thread ID string
    """
    import time

    # Create deterministic thread ID based on user and timestamp
    timestamp = int(time.time())

    # Include session context in thread ID if available
    context_str = ""
    if session_context:
        # Use first few chars of important session info
        profile = session_context.get("cloud_profile", "")
        config = session_context.get("utilities_config", "")
        context_str = f"{profile}_{config}"[:20]

    # Create thread ID: user_timestamp_context
    thread_id = f"{user_id}_{timestamp}"
    if context_str:
        thread_id += f"_{context_str}"

    # Ensure thread ID is valid and not too long
    thread_id = thread_id.replace("/", "_").replace(".", "_")[:100]

    return thread_id


def create_langgraph_config(
    user_context: Dict[str, Any], thread_id: str
) -> Dict[str, Any]:
    """
    Create LangGraph configuration with user context and thread ID.

    Args:
        user_context: User context information
        thread_id: Thread identifier for conversation

    Returns:
        LangGraph configuration dict
    """
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_context.get("user_id", "unknown"),
        }
    }

    # Add optional context
    if "jpmc_user_sid" in user_context:
        config["configurable"]["jpmc_user_sid"] = user_context["jpmc_user_sid"]

    if "profile" in user_context:
        config["configurable"]["profile"] = user_context["profile"]

    return config
