"""Modern JP Morgan meets Discord chat interface demo.

Showcases:
- Glassomorphism design with backdrop blur effects
- JP Morgan blue gradient headers and accents
- Discord-inspired dark theme with smooth animations
- Horizontal source chips with hover effects
- Professional diagnostics panel with grid layouts
- Functional action buttons with tooltips
"""

import streamlit as st
import time
from datetime import datetime
from typing import List, Dict, Any

# JP Morgan Blue palette
JP_BLUE = "#005A8B"
JP_LIGHT_BLUE = "#0066CC" 
JP_DARK_BLUE = "#003E5C"
JP_SILVER = "#F5F7FA"
JP_GRAY = "#6B7280"
DISCORD_DARK = "#2F3136"
DISCORD_DARKER = "#1E1F22"

def inject_modern_css():
    """Inject ultra-modern CSS with JP Morgan meets Discord styling."""
    st.markdown(f"""
    <style>
    /* Global Dark Theme with JP Morgan accents */
    .main {{
        background: linear-gradient(135deg, #0a0c1a 0%, #1a1d29 25%, #2d3748 100%);
        color: #ffffff;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    
    /* Hide Streamlit elements */
    #MainMenu, footer, header {{visibility: hidden;}}
    .stDeployButton {{display: none;}}
    
    /* Ultra-modern glassmorphic container */
    .chat-container {{
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        padding: 32px;
        margin: 20px 0;
        animation: fadeInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }}
    
    .chat-container::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, {JP_LIGHT_BLUE}, transparent);
        opacity: 0.6;
    }}
    
    @keyframes fadeInUp {{
        from {{ 
            opacity: 0; 
            transform: translateY(30px) scale(0.95);
        }}
        to {{ 
            opacity: 1; 
            transform: translateY(0) scale(1);
        }}
    }}
    
    /* Stunning header with gradient and glow */
    .app-header {{
        background: linear-gradient(135deg, {JP_BLUE} 0%, {JP_LIGHT_BLUE} 50%, {JP_DARK_BLUE} 100%);
        padding: 32px 24px;
        border-radius: 20px;
        margin-bottom: 32px;
        text-align: center;
        box-shadow: 
            0 8px 32px rgba(0, 90, 139, 0.4),
            0 1px 2px rgba(255, 255, 255, 0.1) inset;
        position: relative;
        overflow: hidden;
    }}
    
    .app-header::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -50%;
        width: 200%;
        height: 100%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        animation: shimmer 3s infinite;
    }}
    
    @keyframes shimmer {{
        0% {{ transform: translateX(-100%); }}
        100% {{ transform: translateX(100%); }}
    }}
    
    .app-title {{
        color: white;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 
            0 2px 4px rgba(0, 0, 0, 0.4),
            0 0 20px rgba(255, 255, 255, 0.1);
        letter-spacing: -0.02em;
        position: relative;
        z-index: 2;
    }}
    
    .app-subtitle {{
        margin: 12px 0 0 0; 
        color: rgba(255, 255, 255, 0.9); 
        font-size: 1.1rem;
        font-weight: 500;
        position: relative;
        z-index: 2;
    }}
    
    /* Discord-inspired message bubbles */
    .user-message {{
        background: linear-gradient(135deg, {JP_BLUE} 0%, {JP_LIGHT_BLUE} 100%);
        color: white;
        padding: 20px 24px;
        border-radius: 24px 24px 8px 24px;
        margin: 12px 0 12px 80px;
        box-shadow: 
            0 4px 16px rgba(0, 90, 139, 0.3),
            0 1px 2px rgba(0, 0, 0, 0.2);
        animation: slideInRight 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        font-weight: 500;
        position: relative;
    }}
    
    .user-message::before {{
        content: '👤';
        position: absolute;
        left: -60px;
        top: 20px;
        font-size: 1.5rem;
        width: 40px;
        height: 40px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        backdrop-filter: blur(10px);
    }}
    
    .assistant-message {{
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        color: #ffffff;
        padding: 24px;
        border-radius: 8px 24px 24px 24px;
        margin: 12px 80px 12px 0;
        box-shadow: 
            0 4px 16px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        animation: slideInLeft 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        line-height: 1.6;
        position: relative;
    }}
    
    .assistant-message::before {{
        content: '🤖';
        position: absolute;
        right: -60px;
        top: 24px;
        font-size: 1.5rem;
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, {JP_LIGHT_BLUE}, {JP_BLUE});
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(0, 90, 139, 0.3);
    }}
    
    @keyframes slideInRight {{
        from {{ opacity: 0; transform: translateX(40px) scale(0.95); }}
        to {{ opacity: 1; transform: translateX(0) scale(1); }}
    }}
    
    @keyframes slideInLeft {{
        from {{ opacity: 0; transform: translateX(-40px) scale(0.95); }}
        to {{ opacity: 1; transform: translateX(0) scale(1); }}
    }}
    
    /* Ultra-modern horizontal chips */
    .sources-container {{
        margin-top: 20px;
        padding: 16px 0;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        animation: fadeIn 0.5s ease-out 0.2s both;
    }}
    
    .sources-header {{
        color: {JP_LIGHT_BLUE};
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    
    .chip-container {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
    }}
    
    .source-chip {{
        display: inline-flex;
        align-items: center;
        background: linear-gradient(135deg, rgba(0, 90, 139, 0.7) 0%, rgba(0, 102, 204, 0.7) 100%);
        color: white !important;
        text-decoration: none !important;
        padding: 10px 16px;
        border-radius: 25px;
        font-size: 0.8rem;
        font-weight: 500;
        border: 1px solid rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            0 2px 8px rgba(0, 90, 139, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }}
    
    .source-chip::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }}
    
    .source-chip:hover {{
        transform: translateY(-2px) scale(1.05);
        box-shadow: 
            0 8px 20px rgba(0, 90, 139, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        background: linear-gradient(135deg, {JP_LIGHT_BLUE} 0%, {JP_BLUE} 100%);
        color: white !important;
        text-decoration: none !important;
    }}
    
    .source-chip:hover::before {{
        left: 100%;
    }}
    
    /* Ultra-sleek action buttons */
    .action-buttons {{
        display: flex;
        gap: 16px;
        margin: 24px 0;
        flex-wrap: wrap;
        justify-content: center;
    }}
    
    .stButton > button {{
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        padding: 12px 20px !important;
        border-radius: 12px !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: none !important;
        min-width: 120px !important;
        position: relative !important;
        overflow: hidden !important;
    }}
    
    .stButton > button:hover {{
        background: rgba(0, 90, 139, 0.2) !important;
        border-color: {JP_LIGHT_BLUE} !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(0, 90, 139, 0.3) !important;
    }}
    
    .stButton > button:active {{
        transform: translateY(0) !important;
    }}
    
    /* Modern diagnostics panel */
    .diagnostics-drawer {{
        background: rgba(30, 31, 34, 0.8);
        backdrop-filter: blur(25px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        padding: 28px;
        margin: 24px 0;
        box-shadow: 
            0 12px 40px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        animation: slideInUp 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    
    @keyframes slideInUp {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .drawer-header {{
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 24px;
        padding-bottom: 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }}
    
    .drawer-title {{
        color: {JP_LIGHT_BLUE};
        font-weight: 600;
        font-size: 1.3rem;
        margin: 0;
        text-shadow: 0 0 10px rgba(0, 102, 204, 0.3);
    }}
    
    .metric-group {{
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
        border-left: 4px solid {JP_LIGHT_BLUE};
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
    }}
    
    .metric-group:hover {{
        background: rgba(255, 255, 255, 0.04);
        border-left-color: {JP_BLUE};
        transform: translateX(4px);
    }}
    
    .metric-label {{
        color: {JP_SILVER};
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
        opacity: 0.8;
    }}
    
    .metric-value {{
        color: white;
        font-size: 0.95rem;
        line-height: 1.5;
        font-weight: 500;
    }}
    
    .latency-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(90px, 1fr));
        gap: 12px;
        margin-top: 12px;
    }}
    
    .latency-item {{
        background: linear-gradient(135deg, rgba(0, 90, 139, 0.15), rgba(0, 102, 204, 0.15));
        backdrop-filter: blur(10px);
        padding: 12px;
        border-radius: 10px;
        text-align: center;
        font-size: 0.8rem;
        font-weight: 500;
        border: 1px solid rgba(0, 102, 204, 0.2);
        transition: all 0.3s ease;
    }}
    
    .latency-item:hover {{
        background: linear-gradient(135deg, rgba(0, 90, 139, 0.25), rgba(0, 102, 204, 0.25));
        transform: scale(1.05);
    }}
    
    /* Enhanced input styling */
    .stTextInput > div > div > input {{
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 16px !important;
        color: white !important;
        padding: 16px 20px !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }}
    
    .stTextInput > div > div > input:focus {{
        border-color: {JP_LIGHT_BLUE} !important;
        box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.15) !important;
        background: rgba(255, 255, 255, 0.08) !important;
    }}
    
    /* Animated thinking indicator */
    .thinking-indicator {{
        display: flex;
        align-items: center;
        gap: 12px;
        color: {JP_LIGHT_BLUE};
        font-style: italic;
        padding: 20px;
        background: rgba(0, 102, 204, 0.05);
        border-radius: 12px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 102, 204, 0.2);
        margin: 16px 0;
    }}
    
    .thinking-dots {{
        display: flex;
        gap: 6px;
    }}
    
    .thinking-dot {{
        width: 8px;
        height: 8px;
        background: {JP_LIGHT_BLUE};
        border-radius: 50%;
        animation: pulse 1.8s ease-in-out infinite;
        box-shadow: 0 0 10px rgba(0, 102, 204, 0.5);
    }}
    
    .thinking-dot:nth-child(2) {{ animation-delay: 0.3s; }}
    .thinking-dot:nth-child(3) {{ animation-delay: 0.6s; }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 0.3; transform: scale(0.8); }}
        50% {{ opacity: 1; transform: scale(1.3); }}
    }}
    
    /* Success/Warning banners */
    .success-banner {{
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.9) 0%, rgba(56, 142, 60, 0.9) 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 12px;
        margin: 12px 0;
        border-left: 4px solid #4caf50;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 16px rgba(76, 175, 80, 0.3);
    }}
    
    .warning-banner {{
        background: linear-gradient(135deg, rgba(255, 152, 0, 0.9) 0%, rgba(245, 124, 0, 0.9) 100%);
        color: #1a1a1a;
        padding: 16px 20px;
        border-radius: 12px;
        margin: 12px 0;
        border-left: 4px solid #ff9800;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 16px rgba(255, 152, 0, 0.3);
        font-weight: 500;
    }}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: rgba(255, 255, 255, 0.05);
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(135deg, {JP_BLUE}, {JP_LIGHT_BLUE});
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(135deg, {JP_LIGHT_BLUE}, {JP_BLUE});
    }}
    </style>
    """, unsafe_allow_html=True)

def initialize_session():
    """Initialize session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "show_diagnostics" not in st.session_state:
        st.session_state.show_diagnostics = False  # Hidden by default
    
    if "last_query_diagnostics" not in st.session_state:
        st.session_state.last_query_diagnostics = None

def render_header():
    """Render the stunning header."""
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">⚡ Utilities Knowledge Assistant</h1>
        <p class="app-subtitle">JP Morgan Enterprise AI • Powered by Azure OpenAI</p>
    </div>
    """, unsafe_allow_html=True)

def render_export_button():
    """Render export button in top right corner."""
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("💾 Export Chat", key="export", help="Export conversation to file"):
            st.success("Chat exported successfully!")

def render_source_chips(sources: List[Dict[str, Any]]) -> None:
    """Render ultra-modern horizontal source chips."""
    if not sources:
        return
    
    st.markdown('<div class="sources-container">', unsafe_allow_html=True)
    st.markdown('<div class="sources-header">📎 Sources</div>', unsafe_allow_html=True)
    st.markdown('<div class="chip-container">', unsafe_allow_html=True)
    
    for source in sources:
        title = source.get("title", "Source")
        section = source.get("section", "")
        updated_at = source.get("updated_at", "")
        url = source.get("url", "#")
        
        chip_text = title
        if section:
            chip_text += f" §{section}"
        if updated_at:
            chip_text += f" • {updated_at}"
        
        st.markdown(
            f'<a href="{url}" target="_blank" class="source-chip">{chip_text}</a>',
            unsafe_allow_html=True
        )
    
    st.markdown('</div></div>', unsafe_allow_html=True)

def render_diagnostics_drawer():
    """Render collapsible diagnostics drawer."""
    # Toggle button for diagnostics
    if st.button("🔧 Query Diagnostics" if not st.session_state.show_diagnostics else "🔧 Hide Diagnostics", 
                key="diag_toggle", help="Toggle diagnostics panel"):
        st.session_state.show_diagnostics = not st.session_state.show_diagnostics
        st.rerun()
    
    # Only show diagnostics if there are messages and user wants to see them
    if not st.session_state.show_diagnostics or not st.session_state.messages:
        return
        
    st.markdown('<div class="diagnostics-drawer">', unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="drawer-header">
        <span style="font-size: 1.4rem;">🔧</span>
        <h3 class="drawer-title">Query Diagnostics</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-group">
            <div class="metric-label">🎯 Intent Analysis</div>
            <div class="metric-value">
                <strong>workflow</strong><br>
                Confidence: 0.92
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-group">
            <div class="metric-label">🪙 Token Usage</div>
            <div class="metric-value">
                Input: 485<br>
                Output: 156<br>
                Total: 641
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-group">
            <div class="metric-label">📄 Top BM25 Results</div>
            <div class="metric-value" style="font-family: monospace; font-size: 0.85rem;">
                • UTILS:CUST:START_SERVICE:v4#overview<br>
                • UTILS:CUST:START_SERVICE:v4#eligibility<br>
                • UTILS:API:ACCOUNT_UTILITY:v3
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-group">
            <div class="metric-label">🔍 Top kNN Results</div>
            <div class="metric-value" style="font-family: monospace; font-size: 0.85rem;">
                • UTILS:CUST:START_SERVICE:v4#overview<br>
                • UTILS:API:CUSTOMER_SUMMARY:v2<br>
                • UTILS:GLOBAL:PLATFORM:v2
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-group">
            <div class="metric-label">⚡ Performance</div>
            <div class="latency-grid">
                <div class="latency-item">Intent<br>52ms</div>
                <div class="latency-item">BM25<br>134ms</div>
                <div class="latency-item">kNN<br>189ms</div>
                <div class="latency-item">Fuse<br>28ms</div>
                <div class="latency-item">LLM<br>892ms</div>
            </div>
            <div style="text-align: center; margin-top: 16px; font-weight: 600; color: {JP_LIGHT_BLUE}; font-size: 1.1rem;">
                Total: 1,295ms
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def simulate_thinking():
    """Show beautiful thinking animation."""
    thinking_placeholder = st.empty()
    thinking_placeholder.markdown("""
    <div class="thinking-indicator">
        <span>💭 Thinking</span>
        <div class="thinking-dots">
            <div class="thinking-dot"></div>
            <div class="thinking-dot"></div>
            <div class="thinking-dot"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    time.sleep(2)  # Simulate processing time
    thinking_placeholder.empty()

def main():
    """Main ultra-modern interface."""
    st.set_page_config(
        page_title="JP Morgan Utilities AI",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Inject ultra-modern CSS
    inject_modern_css()
    
    # Initialize
    initialize_session()
    
    # Header with export button
    render_header()
    render_export_button()
    
    # Chat messages using Streamlit's native chat components
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                render_source_chips(message["sources"])
    
    # Diagnostics drawer (collapsible)
    render_diagnostics_drawer()
    
    # Chat input
    user_input = st.chat_input("💬 Ask about utilities, APIs, procedures, or anything else...")
    
    # Process input
    if user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Show thinking animation
        with st.chat_message("assistant"):
            simulate_thinking()
        
        # Mock response based on query
        if "start" in user_input.lower() and "service" in user_input.lower():
            response = """To start utility service, you'll need to follow these streamlined steps:

**🏠 Residential Setup:**
1. **Document Preparation** - Valid ID, property proof, credit report
2. **Credit Assessment** - Customers with 650+ credit skip deposits  
3. **Application Submission** - Online, phone, or in-person options
4. **Installation Scheduling** - Same-day activation for standard accounts
5. **Service Activation** - Field technician completes setup

**🏢 Business Accounts:**
- Require business license and tax ID
- Follow expedited commercial pathway
- Priority scheduling available

**🎖️ Special Cases:**
- Military personnel get expedited processing
- Students can use parent/guardian co-signature
- Senior discounts automatically applied

The process has been significantly streamlined in 2025 to reduce wait times and paperwork."""
            
            sources = [
                {
                    "title": "Start Service Process (v4)",
                    "section": "overview", 
                    "updated_at": "2025-01-15",
                    "url": "https://confluence.jpmc.com/utilities/start-service-v4#overview"
                },
                {
                    "title": "Start Service Process (v4)",
                    "section": "eligibility",
                    "updated_at": "2025-01-15",
                    "url": "https://confluence.jpmc.com/utilities/start-service-v4#eligibility"
                },
                {
                    "title": "Account Utility API (v3)",
                    "section": "api_spec",
                    "updated_at": "2024-11-25",
                    "url": "https://api-docs.jpmc.com/account-utility-v3"
                }
            ]
        else:
            response = f"""I can help you with utilities information and enterprise APIs. 

**Available Services:**
- 🏠 Residential utility setup and management
- 🏢 Commercial account services  
- 📊 API documentation and integration guides
- 🔧 Technical support and troubleshooting
- 📋 Compliance and regulatory information

**Popular Topics:**
- Starting or stopping utility service
- Payment plans and billing options
- API endpoints and authentication
- Service outage reporting
- Energy efficiency programs

What specific information can I help you find?"""
            
            sources = [
                {
                    "title": "Utilities Help Center",
                    "section": "overview",
                    "updated_at": "2025-01-10", 
                    "url": "https://help.jpmc.com/utilities"
                }
            ]
        
        # Add assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })
        
        st.rerun()

if __name__ == "__main__":
    main()