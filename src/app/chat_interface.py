# src/app/chat_interface.py
# Local/JPMC switch is config-only; do not import Azure/OpenAI SDKs here.

"""Beautiful, responsive chat interface for the utilities assistant.

This is a pure UI layer that delegates all business logic to controllers and services.
Configuration switching between local/JPMC is handled transparently by the infra layer.
"""

import streamlit as st
import asyncio
import time
import logging
import html
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import only infra and controllers - no direct SDK imports
from ..infra.config import get_settings
from ..controllers.turn_controller import handle_turn

logger = logging.getLogger(__name__)

# UI Configuration
CHAT_CONTAINER_HEIGHT = 600
MAX_CHAT_HISTORY = 20
TYPING_DELAY = 0.02


def initialize_chat_session():
    """Initialize Streamlit session state for chat."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{int(time.time())}"
    
    if "settings" not in st.session_state:
        st.session_state.settings = get_settings()
        logger.info(f"Initialized with {st.session_state.settings.profile} profile")


def render_chat_header():
    """Render the chat header with branding and status."""
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #0047AB 0%, #003875 100%); margin: -1rem -1rem 2rem -1rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
        <h1 style='color: #ffffff; margin: 0; font-size: 2.4rem; font-weight: 300; font-family: "Helvetica Neue", Arial, sans-serif;'>
            Enterprise Knowledge Hub
        </h1>
        <p style='color: rgba(255,255,255,0.9); margin: 0.8rem 0 0 0; font-size: 1rem; font-weight: 400;'>
            Intelligent Assistant for Utilities and APIs
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_profile_indicator():
    """Render current profile indicator."""
    profile = st.session_state.settings.profile
    profile_colors = {
        "local": "#28a745",
        "jpmc_azure": "#0047AB", 
        "tests": "#ffc107"
    }
    
    color = profile_colors.get(profile, "#6c757d")
    
    st.markdown(f"""
    <div style='position: fixed; top: 20px; right: 20px; z-index: 999;'>
        <div style='background: {color}; color: white; padding: 8px 16px; border-radius: 4px; 
                    font-size: 0.85rem; font-weight: 500; font-family: "Helvetica Neue", Arial, sans-serif; 
                    box-shadow: 0 2px 8px rgba(0,0,0,0.15); border: 1px solid rgba(255,255,255,0.2);'>
            {profile.replace("_", " ").title()}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_message_bubble(message: Dict[str, Any], index: int):
    """Render a chat message bubble with beautiful styling."""
    is_user = message["role"] == "user"
    content = message["content"]
    timestamp = message.get("timestamp", "")
    
    # Use Streamlit's chat message components
    with st.chat_message(message["role"]):
        # For assistant messages, show intent if available
        if not is_user and message.get("intent"):
            intent = message["intent"]
            st.markdown(f"""
            <span style='background: #f8f9fa; color: #495057; padding: 4px 12px; 
                        border-radius: 12px; font-size: 0.8rem; font-weight: 500; 
                        font-family: "Helvetica Neue", Arial, sans-serif; border: 1px solid #dee2e6;'>
                {intent['intent'].title()} Query
            </span>
            """, unsafe_allow_html=True)
            st.markdown("")  # Add some spacing
        
        # Display the main content
        st.markdown(content)
        
        # Show sources if available
        if not is_user and message.get("sources"):
            sources = message["sources"]
            if sources:
                st.markdown("**Sources:**")
                cols = st.columns(min(len(sources), 3))
                for i, source in enumerate(sources[:3]):
                    with cols[i % 3]:
                        title = source.get("title", f"Source {i+1}")
                        excerpt = source.get("excerpt", "")
                        st.button(
                            title[:30] + "..." if len(title) > 30 else title,
                            key=f"source_{index}_{i}",
                            help=excerpt,
                            use_container_width=True
                        )
        
        # Show timestamp
        if timestamp:
            st.caption(f"⏰ {timestamp}")


def render_source_chips(sources: List[Dict[str, Any]]) -> str:
    """Render source citation chips."""
    if not sources:
        return ""
    
    chips_html = "<div style='margin-top: 1rem; border-top: 1px solid #dee2e6; padding-top: 1rem;'>"
    chips_html += "<div style='font-size: 0.85rem; color: #6c757d; margin-bottom: 0.8rem; font-weight: 500;'>Sources:</div>"
    
    for i, source in enumerate(sources[:5]):  # Limit to 5 sources
        title = source.get("title", f"Source {i+1}")
        excerpt = source.get("excerpt", "")
        
        chips_html += f"""
        <div style='display: inline-block; margin: 0.3rem 0.5rem 0.3rem 0; 
                    background: #f8f9fa; 
                    border: 1px solid #dee2e6; border-radius: 16px; padding: 0.5rem 1rem; font-size: 0.8rem; 
                    color: #495057; cursor: pointer; transition: all 0.2s ease; 
                    font-family: "Helvetica Neue", Arial, sans-serif;'
                    title='{excerpt}' onmouseover='this.style.backgroundColor="#e9ecef"' onmouseout='this.style.backgroundColor="#f8f9fa"'>
            <span style='font-weight: 500;'>{title[:40]}{'...' if len(title) > 40 else ''}</span>
        </div>
        """
    
    chips_html += "</div>"
    return chips_html


def render_typing_indicator():
    """Render animated typing indicator."""
    st.markdown("""
    <div style='display: flex; justify-content: flex-start; margin: 1rem 0;'>
        <div style='background: white; border: 1px solid #dee2e6; color: #495057; padding: 1rem 1.5rem; 
                    border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
            <div style='display: flex; align-items: center; gap: 0.8rem;'>
                <div style='color: #6c757d; font-size: 0.9rem; font-family: "Helvetica Neue", Arial, sans-serif; font-weight: 400;'>Assistant is typing</div>
                <div style='display: flex; gap: 4px;'>
                    <div style='width: 8px; height: 8px; background: #0047AB; border-radius: 50%; 
                                animation: bounce 1.4s infinite ease-in-out both; animation-delay: -0.32s;'></div>
                    <div style='width: 8px; height: 8px; background: #0047AB; border-radius: 50%; 
                                animation: bounce 1.4s infinite ease-in-out both; animation-delay: -0.16s;'></div>
                    <div style='width: 8px; height: 8px; background: #0047AB; border-radius: 50%; 
                                animation: bounce 1.4s infinite ease-in-out both;'></div>
                </div>
            </div>
        </div>
    </div>
    
    <style>
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
    </style>
    """, unsafe_allow_html=True)


def render_chat_container():
    """Render the main chat container."""
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .stApp {
        background: #f8f9fa;
    }
    
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1.5rem;
        border-radius: 8px;
        background: white;
        border: 1px solid #dee2e6;
        margin-bottom: 2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }
    
    .stChatInputContainer {
        border-top: 1px solid #dee2e6;
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
    }
    
    .stChatInput > div > div > textarea {
        border-radius: 8px !important;
        border: 2px solid #dee2e6 !important;
        padding: 1rem 1.5rem !important;
        font-size: 1rem !important;
        transition: all 0.2s ease !important;
        background: white !important;
        color: #212529 !important;
        font-family: "Helvetica Neue", Arial, sans-serif !important;
        font-weight: 400 !important;
    }
    
    .stChatInput > div > div > textarea:focus {
        border-color: #0047AB !important;
        box-shadow: 0 0 0 3px rgba(0, 71, 171, 0.1) !important;
    }
    
    .stChatInput > div > div > textarea::placeholder {
        color: #6c757d !important;
    }
    
    .quick-actions {
        display: flex;
        gap: 0.8rem;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }
    
    .quick-action-btn {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 0.8rem 1.2rem;
        font-size: 0.9rem;
        color: #495057;
        cursor: pointer;
        transition: all 0.2s ease;
        text-decoration: none;
        font-family: "Helvetica Neue", Arial, sans-serif;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .quick-action-btn:hover {
        background: #0047AB;
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 71, 171, 0.15);
        border-color: #0047AB;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Render messages
    with st.container():
        if not st.session_state.messages:
            render_welcome_message()
        else:
            for i, message in enumerate(st.session_state.messages):
                render_message_bubble(message, i)


def render_welcome_message():
    """Render welcome message for new sessions."""
    profile = st.session_state.settings.profile
    
    welcome_content = f"""
    <div style='text-align: center; padding: 3rem 2rem; color: #6c757d; background: white; border: 1px solid #dee2e6; border-radius: 8px; margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
        <div style='font-size: 3.5rem; margin-bottom: 1.5rem; color: #0047AB;'>💼</div>
        <h2 style='color: #212529; margin-bottom: 1rem; font-weight: 500; font-family: "Helvetica Neue", Arial, sans-serif; font-size: 1.8rem;'>
            Welcome to Enterprise Knowledge Hub
        </h2>
        <p style='font-size: 1.1rem; line-height: 1.6; margin-bottom: 2rem; max-width: 600px; margin-left: auto; margin-right: auto; color: #495057;'>
            Your intelligent assistant for enterprise utilities, APIs, and documentation. 
            Ask questions about API specifications, utility information, or explore our knowledge base.
        </p>
        <div style='background: #f8f9fa; border: 1px solid #dee2e6; padding: 0.8rem 1.5rem; display: inline-block; border-radius: 6px;'>
            <span style='color: #495057; font-family: "Helvetica Neue", Arial, sans-serif; font-weight: 500; font-size: 0.9rem;'>
                Environment: {profile.replace("_", " ").title()}
            </span>
        </div>
    </div>
    """
    
    st.markdown(welcome_content, unsafe_allow_html=True)


def render_quick_actions():
    """Render quick action buttons."""
    st.markdown("""
    <div class='quick-actions'>
        <div class='quick-action-btn' onclick='document.querySelector("textarea").value="List all available APIs"; document.querySelector("textarea").dispatchEvent(new Event("input"));'>
            📋 List APIs
        </div>
        <div class='quick-action-btn' onclick='document.querySelector("textarea").value="How do I use the Customer Summary Utility?"; document.querySelector("textarea").dispatchEvent(new Event("input"));'>
            💡 Usage Guide
        </div>
        <div class='quick-action-btn' onclick='document.querySelector("textarea").value="Show me API endpoints for Account Utility"; document.querySelector("textarea").dispatchEvent(new Event("input"));'>
            🔍 API Details
        </div>
        <div class='quick-action-btn' onclick='document.querySelector("textarea").value="Start Over"; document.querySelector("textarea").dispatchEvent(new Event("input"));'>
            🔄 New Session
        </div>
    </div>
    
    <script>
    // Add click handlers for quick actions
    document.addEventListener('DOMContentLoaded', function() {
        const buttons = document.querySelectorAll('.quick-action-btn');
        buttons.forEach(btn => {
            btn.addEventListener('click', function() {
                const text = this.getAttribute('data-text');
                if (text) {
                    const textarea = document.querySelector('textarea');
                    if (textarea) {
                        textarea.value = text;
                        textarea.focus();
                    }
                }
            });
        });
    });
    </script>
    """, unsafe_allow_html=True)


async def process_user_message(user_input: str):
    """Process user message and handle streaming response."""
    if not user_input.strip():
        return
    
    # Add user message to chat
    user_message = {
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().strftime("%H:%M")
    }
    st.session_state.messages.append(user_message)
    
    # Create placeholder for assistant response
    response_placeholder = st.empty()
    
    # Show typing indicator
    with response_placeholder:
        render_typing_indicator()
    
    # Initialize assistant message
    assistant_message = {
        "role": "assistant", 
        "content": "",
        "timestamp": "",
        "sources": [],
        "intent": None
    }
    
    try:
        # Handle the turn with streaming
        async for update in handle_turn(
            user_input,
            st.session_state.settings,
            st.session_state.messages[-10:],  # Recent history
            token_provider=None  # TODO: Add token provider for Azure
        ):
            
            if update["type"] == "response_chunk":
                # Accumulate response content
                assistant_message["content"] += update["content"]
                
                # Update the placeholder with current content
                with response_placeholder:
                    render_message_bubble(assistant_message, len(st.session_state.messages))
                
                # Small delay for smooth typing effect
                await asyncio.sleep(TYPING_DELAY)
            
            elif update["type"] == "intent":
                # Store intent information
                assistant_message["intent"] = update["intent"]
            
            elif update["type"] == "complete":
                # Final result with sources
                result = update["result"]
                assistant_message["content"] = result["answer"]
                assistant_message["sources"] = result["sources"]
                assistant_message["timestamp"] = datetime.now().strftime("%H:%M")
                
                # Final render
                with response_placeholder:
                    render_message_bubble(assistant_message, len(st.session_state.messages))
            
            elif update["type"] == "error":
                # Handle errors gracefully
                result = update["result"]
                assistant_message["content"] = result["answer"]
                assistant_message["timestamp"] = datetime.now().strftime("%H:%M")
                
                with response_placeholder:
                    render_message_bubble(assistant_message, len(st.session_state.messages))
    
    except Exception as e:
        logger.error(f"Message processing failed: {e}")
        assistant_message["content"] = f"I encountered an error: {str(e)}"
        assistant_message["timestamp"] = datetime.now().strftime("%H:%M")
        
        with response_placeholder:
            render_message_bubble(assistant_message, len(st.session_state.messages))
    
    # Add completed message to history
    st.session_state.messages.append(assistant_message)
    
    # Trim chat history if too long
    if len(st.session_state.messages) > MAX_CHAT_HISTORY:
        st.session_state.messages = st.session_state.messages[-MAX_CHAT_HISTORY:]


def main():
    """Main chat interface application."""
    st.set_page_config(
        page_title="Digital Knowledge Hub",
        page_icon="🌟",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize session
    initialize_chat_session()
    
    # Render UI components
    render_chat_header()
    render_profile_indicator()
    render_quick_actions()
    render_chat_container()
    
    # Chat input
    user_input = st.chat_input(
        placeholder="Ask me about utilities, APIs, or documentation...",
        key="chat_input"
    )
    
    # Process message if provided
    if user_input:
        # Run async message processing
        asyncio.run(process_user_message(user_input))
        st.rerun()


if __name__ == "__main__":
    main()