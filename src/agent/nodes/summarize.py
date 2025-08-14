"""Summarize node - query normalization using jinja template."""

import logging
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from langchain_core.messages import HumanMessage, SystemMessage

from services.normalize import normalize_query  # Keep existing logic as fallback
from src.telemetry.logger import stage
from .base_node import BaseNodeHandler

logger = logging.getLogger(__name__)

# Load jinja templates
template_dir = Path(__file__).parent.parent / "prompts"
jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))


@stage("normalize")
async def summarize_node(state: dict, config, *, store=None) -> dict:
    """
    Summarize/normalize the user query using LLM with jinja template.
    
    Follows LangGraph pattern with config parameter and optional store injection.
    
    Args:
        state: Workflow state containing original_query
        config: RunnableConfig with user context and configuration
        store: Optional BaseStore for cross-thread user memory
        
    Returns:
        State update with normalized_query
    """
    import time
    from infra.telemetry import log_normalize_stage
    
    start_time = time.perf_counter()
    req_id = getattr(config, 'run_id', 'unknown') if config else 'unknown'
    
    try:
        user_input = state["original_query"]
        
        # Try LLM-based normalization first
        try:
            template = jinja_env.get_template("summarize.jinja")
            prompt = template.render(user_input=user_input)
            
            # Extract resources from config
            from infra.resource_manager import get_resources
            resources = get_resources()
            
            # Use the working Azure OpenAI client configuration for LangChain
            from langchain_openai import AzureChatOpenAI
            
            # Get authentication parameters from working clients.py approach
            from src.infra.clients import make_chat_client
            from src.infra.config import ChatCfg
            from utils import load_config
            import os
            
            # Load config for API key (matching clients.py pattern)
            auth_config = load_config()
            api_key = auth_config.get('azure_openai', 'api_key', fallback=None)
            
            # Get Bearer token from token provider if available
            headers = {"user_sid": os.getenv("JPMC_USER_SID", "REPLACE")}
            if resources.token_provider:
                try:
                    bearer_token = resources.token_provider()
                    headers["Authorization"] = f"Bearer {bearer_token}"
                except Exception as e:
                    logger.warning(f"Bearer token failed, using API key only: {e}")
            
            # Create LangChain client with same auth pattern as clients.py
            langchain_client = AzureChatOpenAI(
                api_version=resources.settings.chat.api_version,
                azure_deployment=resources.settings.chat.model,
                azure_endpoint=resources.settings.chat.api_base,
                api_key=api_key,
                default_headers=headers,
                temperature=0.1,
                max_tokens=500
            )
            
            response = await langchain_client.ainvoke([
                SystemMessage(content="You are a query normalization assistant."),
                HumanMessage(content=prompt)
            ])
            
            normalized = response.content.strip()
            
            # Sanity check - if LLM response is too different or empty, use fallback
            if not normalized or len(normalized) < 3:
                raise ValueError("LLM normalization produced empty result")
            
            logger.info(f"LLM normalized: '{user_input}' -> '{normalized}'")
            
        except Exception as e:
            logger.warning(f"LLM normalization failed, using fallback: {e}")
            normalized = normalize_query(user_input)
        
        # MISSING: Add telemetry logging (from traditional pipeline)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        log_normalize_stage(req_id, user_input, normalized, elapsed_ms)
        
        return {
            "normalized_query": normalized,
            "workflow_path": state.get("workflow_path", []) + ["summarize"]
        }
        
    except Exception as e:
        logger.error(f"Summarize node failed: {e}")
        # Fallback to original query
        return {
            "normalized_query": state["original_query"],
            "workflow_path": state.get("workflow_path", []) + ["summarize_error"],
            "error_messages": state.get("error_messages", []) + [f"Summarize failed: {e}"]
        }


class SummarizeNode(BaseNodeHandler):
    """Class-based wrapper for summarize functionality."""
    
    def __init__(self):
        super().__init__("summarize")
    
    async def execute(self, state: dict) -> dict:
        """Execute the summarize logic using the existing function."""
        # The existing function expects config parameter, but we don't have it in execute()
        # We need to call the function directly with a mock config
        config = {"configurable": {"thread_id": "unknown"}}
        return await summarize_node(state, config)