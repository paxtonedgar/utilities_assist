# Provider/model selection is config-only. Do not import OpenAI/Azure SDKs here directly.

"""Clean, minimal chat interface for utilities assistant."""

import streamlit as st
import asyncio
import time
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import only infra and controllers - no direct SDK imports
# sys.path already configured by streamlit_app.py

from src.infra.settings import get_settings
from controllers.graph_integration import handle_turn
from infra.telemetry import get_telemetry_collector, format_event_for_display
from infra.resource_manager import initialize_resources, get_resources, health_check
from src.telemetry.logger import get_stage_logs

logger = logging.getLogger(__name__)

def _extract_user_context() -> Dict[str, Any]:
    """Extract user context for the session with graceful fallback."""
    try:
        from infra.persistence import extract_user_context
        # Pass None resources to extract from environment 
        return extract_user_context(None)
    except ImportError:
        logger.info("Persistence module not available, using local user context")
        return {
            "user_id": "streamlit_user",
            "session_metadata": {
                "cloud_profile": os.getenv("CLOUD_PROFILE", "local"),
                "utilities_config": os.getenv("UTILITIES_CONFIG", "config.local.ini")
            }
        }
    except Exception as e:
        logger.warning(f"Failed to extract user context: {e}")
        return {
            "user_id": "streamlit_user",
            "session_metadata": {
                "cloud_profile": os.getenv("CLOUD_PROFILE", "local"),
                "utilities_config": os.getenv("UTILITIES_CONFIG", "config.local.ini")
            }
        }

def inject_minimal_css():
    """Clean, professional styling."""
    st.markdown("""
    <style>
    /* Clean modern theme */
    .main {
        background: #f8f9fa;
        color: #2c3e50;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Simple message styling */
    .user-message {
        background: #007bff;
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0 8px 40px;
        max-width: 70%;
        margin-left: auto;
    }
    
    .assistant-message {
        background: white;
        border: 1px solid #e9ecef;
        color: #2c3e50;
        padding: 16px;
        border-radius: 4px 18px 18px 18px;
        margin: 8px 40px 8px 0;
        max-width: 85%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Simple source list */
    .sources {
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px solid #e9ecef;
        font-size: 0.9rem;
    }
    
    .source-item {
        display: inline-block;
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 4px 8px;
        margin: 2px;
        border-radius: 4px;
        text-decoration: none;
        color: #007bff;
    }
    
    /* Clean header */
    .header {
        text-align: center;
        padding: 20px 0;
        border-bottom: 1px solid #e9ecef;
        margin-bottom: 20px;
    }
    
    .header h1 {
        color: #2c3e50;
        font-size: 1.8rem;
        margin: 0;
        font-weight: 600;
    }
    
    .header p {
        color: #6c757d;
        margin: 8px 0 0 0;
    }
    
    /* Typing indicator animation */
    .typing-indicator {
        display: inline-block;
        animation: blink 1s infinite;
        color: #007bff;
        font-weight: bold;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    /* Thinking animation */
    .thinking {
        color: #6c757d;
        font-style: italic;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session():
    """Initialize Streamlit session state with user context and thread management."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "settings" not in st.session_state:
        st.session_state.settings = get_settings()
        logger.info(f"Initialized with {st.session_state.settings.cloud_profile} profile")
    
    # Removed mock corpus - only use production Confluence and OpenSearch
    
    # Initialize user context and thread management
    if "user_context" not in st.session_state:
        st.session_state.user_context = _extract_user_context()
        logger.info(f"Initialized user context: {st.session_state.user_context.get('user_id', 'unknown')}")
    
    if "thread_id" not in st.session_state:
        try:
            from infra.persistence import generate_thread_id
            st.session_state.thread_id = generate_thread_id(
                st.session_state.user_context.get("user_id", "unknown"),
                st.session_state.user_context.get("session_metadata")
            )
        except ImportError:
            # Fallback thread ID generation
            import time
            user_id = st.session_state.user_context.get("user_id", "unknown")
            st.session_state.thread_id = f"{user_id}_{int(time.time())}"
        logger.info(f"Generated new thread ID: {st.session_state.thread_id}")
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    # Phase 1 Optimization: Initialize shared resources once at startup (LangGraph pattern)
    if "resources_initialized" not in st.session_state:
        try:
            logger.info("Phase 1: Initializing shared resources for performance optimization...")
            st.session_state.resources = initialize_resources(st.session_state.settings)
            st.session_state.resources_initialized = True
            
            # Log resource health and performance benefits
            health = health_check()
            logger.info(f"Resource health: {health['status']} (eliminates 25-50% response time overhead)")
            
        except Exception as e:
            logger.error(f"Failed to initialize resources: {e}")
            st.error(f"⚠️ Resource initialization failed: {e}")
            st.error("Performance will be degraded. Please refresh the page.")
            st.session_state.resources = None
            st.session_state.resources_initialized = False

def render_header():
    """Simple, clean header."""
    st.markdown("""
    <div class="header">
        <h1>Utilities Assistant</h1>
        <p>Enterprise Knowledge Search</p>
    </div>
    """, unsafe_allow_html=True)

def render_sources(sources: List[Dict[str, Any]]) -> None:
    """Simple source display."""
    if not sources:
        return
    
    st.markdown('<div class="sources"><strong>Sources:</strong><br>', unsafe_allow_html=True)
    for source in sources:
        title = source.get("title", "Source")
        url = source.get("url", "#")
        st.markdown(f'<a href="{url}" class="source-item">{title}</a>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_simple_stats(req_id: str):
    """Show basic performance stats if available."""
    if not req_id:
        return
        
    collector = get_telemetry_collector()
    events = collector.get_events(req_id)
    
    if not events:
        return
    
    overall_event = next((e for e in events if e.stage == "overall"), None)
    if overall_event:
        total_time = getattr(overall_event, 'ms', 0)
        if total_time > 0:
            st.caption(f"⏱️ Response time: {total_time:.0f}ms")

def render_stage_logs(req_id: str):
    """Render structured stage logs for the request."""
    if not req_id:
        return
    
    try:
        # Get stage logs for this request
        stage_logs = get_stage_logs(req_id=req_id, last_n=20)
        
        if not stage_logs:
            return
        
        # Group logs by stage for better display
        stages = {}
        for log in stage_logs:
            stage_name = log.get("stage", "unknown")
            if stage_name not in stages:
                stages[stage_name] = []
            stages[stage_name].append(log)
        
        # Only show if we have interesting stages to display
        if stages:
            with st.expander("🔍 Stage Details", expanded=False):
                for stage_name, logs in stages.items():
                    st.subheader(f"📊 {stage_name.title()}")
                    
                    # Find the main events (start/end pairs)
                    start_log = next((log for log in logs if log.get("event") == "start"), None)
                    end_log = next((log for log in logs if log.get("event") in ["success", "end"]), None)
                    error_log = next((log for log in logs if log.get("event") == "error"), None)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if end_log and "ms" in end_log:
                            st.metric("Duration", f"{end_log['ms']:.0f}ms")
                        elif start_log:
                            st.metric("Status", "Started")
                    
                    with col2:
                        if end_log and "result_count" in end_log:
                            st.metric("Results", f"{end_log['result_count']}")
                        elif end_log and "hits" in end_log:
                            st.metric("Hits", f"{end_log['hits']}")
                        elif start_log and "k" in start_log:
                            st.metric("Requested", f"{start_log['k']}")
                    
                    with col3:
                        if error_log:
                            st.metric("Status", "❌ Error", delta="Failed")
                        elif end_log:
                            st.metric("Status", "✅ Success")
                        else:
                            st.metric("Status", "🔄 Running")
                    
                    # Show additional details
                    details = []
                    if start_log:
                        if "index" in start_log:
                            details.append(f"**Index**: {start_log['index']}")
                        if "query_type" in start_log:
                            details.append(f"**Type**: {start_log['query_type']}")
                        if "filters_enabled" in start_log:
                            details.append(f"**Filters**: {'Yes' if start_log['filters_enabled'] else 'No'}")
                    
                    if error_log:
                        details.append(f"**Error**: {error_log.get('error_message', 'Unknown error')}")
                    
                    if details:
                        st.markdown(" • ".join(details))
                    
                    st.divider()
                        
    except Exception as e:
        logger.debug(f"Failed to render stage logs: {e}")

async def process_user_input(user_input: str) -> None:
    """Process user input with thinking animation."""
    # Add user message immediately and show it
    user_message = {"role": "user", "content": user_input}
    st.session_state.messages.append(user_message)
    st.session_state.conversation_history.append(user_message)
    
    # Add thinking indicator
    thinking_message = {"role": "assistant", "content": "", "thinking": True}
    st.session_state.messages.append(thinking_message)
    st.rerun()  # Show user message and thinking indicator
    
    # Process the response
    assistant_response = {
        "role": "assistant",
        "content": "",
        "sources": [],
        "req_id": None
    }
    
    try:
        async for chunk in handle_turn(
            user_input,
            st.session_state.resources,
            chat_history=st.session_state.conversation_history[-10:],
            use_mock_corpus=False,
            thread_id=st.session_state.thread_id,
            user_context=st.session_state.user_context
        ):
            if chunk["type"] == "response_chunk":
                assistant_response["content"] += chunk["content"]
            elif chunk["type"] == "complete":
                result = chunk["result"]
                assistant_response["sources"] = result.get("sources", [])
                assistant_response["req_id"] = chunk.get("req_id")
                break
            elif chunk["type"] == "error":
                assistant_response["content"] = f"❌ {chunk['result'].get('answer', 'An error occurred')}"
                break
    
    except Exception as e:
        assistant_response["content"] = f"❌ Error: {str(e)}"
        logger.error(f"Error in process_user_input: {e}")
    
    # Remove thinking indicator and add final response
    st.session_state.messages.pop()  # Remove thinking message
    st.session_state.messages.append(assistant_response)
    
    # Add to conversation history for context
    assistant_history = {"role": "assistant", "content": assistant_response["content"]}
    st.session_state.conversation_history.append(assistant_history)

def main():
    """Simple chat interface."""
    st.set_page_config(
        page_title="Utilities Assistant",
        page_icon="🔧",
        layout="centered"
    )
    
    inject_minimal_css()
    initialize_session()
    render_header()
    
    # Chat history with thinking animation
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            # Handle assistant messages
            content = message["content"] 
            is_thinking = message.get("thinking", False)
            
            if is_thinking:
                # Show animated thinking indicator
                st.markdown('<div class="assistant-message"><span class="thinking">🤔 thinking...</span></div>', unsafe_allow_html=True)
            elif content:
                # Show final message
                st.markdown(f'<div class="assistant-message">{content}</div>', unsafe_allow_html=True)
                
                # Show sources and stats for completed messages
                if message.get("sources"):
                    render_sources(message["sources"])
                if message.get("req_id"):
                    render_simple_stats(message["req_id"])
                    render_stage_logs(message["req_id"])
    
    # Input
    user_input = st.chat_input("Ask about utilities, APIs, or procedures...")
    if user_input:
        # Run async function synchronously  
        asyncio.run(process_user_input(user_input))

if __name__ == "__main__":
    main()