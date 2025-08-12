# Provider/model selection is config-only. Do not import OpenAI/Azure SDKs here directly.

"""Demo UI for Utilities Knowledge Assistant showcasing the clean interface design."""

import streamlit as st
import time
from datetime import datetime
from typing import List, Dict, Any

# Mock data for demonstration
MOCK_TELEMETRY = {
    "intent": {"intent": "workflow", "confidence": 0.85},
    "bm25_docs": ["UTILS:CUST:START_SERVICE:v4#overview", "UTILS:CUST:START_SERVICE:v4#eligibility", "UTILS:API:ACCOUNT_UTILITY:v3"],
    "knn_docs": ["UTILS:CUST:START_SERVICE:v4#overview", "UTILS:API:CUSTOMER_SUMMARY:v2", "UTILS:GLOBAL:PLATFORM:v2"],
    "latencies": {"intent": 45, "bm25": 120, "knn": 180, "fuse": 25, "llm": 850, "total": 1220},
    "tokens": {"in": 450, "out": 120},
    "budget_warning": False,
    "errors": 0
}

MOCK_SOURCES = [
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

MOCK_CONTEXT = """Based on the retrieved documents, here's the comprehensive information about starting utility service:

## Start Service Process (v4) - Overview
The Start Service process has been streamlined in 2025. Standard residential customers can now be activated same-day with just ID verification. No deposits required for customers with credit score >650. Corporate accounts follow expedited pathway.

## Start Service Process (v4) - Eligibility  
Eligibility requirements: 1) Valid government ID, 2) Credit check (>650 score preferred), 3) Property ownership or lease verification. Special cases: Corporate accounts require business license.

## Account Utility API (v3)
GET /api/v3/account/{account_number}/utility - Retrieve account utility information including service type, rate schedule, meter details."""


def initialize_session():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "use_mock_corpus" not in st.session_state:
        st.session_state.use_mock_corpus = True
        
    if "show_raw_context" not in st.session_state:
        st.session_state.show_raw_context = False
        
    if "current_telemetry" not in st.session_state:
        st.session_state.current_telemetry = None


def render_source_chips(sources: List[Dict[str, Any]]) -> None:
    """Render source citation chips."""
    if not sources:
        return
    
    st.markdown("**Sources:**")
    
    for source in sources:
        title = source.get("title", "Source")
        section = source.get("section", "")
        updated_at = source.get("updated_at", "")
        url = source.get("url", "#")
        
        # Format: Title §Section • 2025-07-31 → clickable url
        chip_text = title
        if section:
            chip_text += f" §{section}"
        if updated_at:
            chip_text += f" • {updated_at}"
        
        # Create clickable chip
        st.markdown(
            f'<a href="{url}" target="_blank" style="'
            f'display: inline-block; '
            f'margin: 2px 4px; '
            f'padding: 4px 8px; '
            f'background: #f0f2f6; '
            f'border: 1px solid #d1d5db; '
            f'border-radius: 4px; '
            f'text-decoration: none; '
            f'font-size: 0.85em; '
            f'color: #374151; '
            f'">{chip_text}</a>',
            unsafe_allow_html=True
        )


def render_diagnostics_sidebar():
    """Render diagnostics sidebar."""
    with st.sidebar:
        st.markdown("### 🔧 Diagnostics")
        
        # Corpus selection
        use_mock = st.selectbox(
            "Index",
            ["Production (confluence_current)", "Mock (confluence_mock)"],
            index=1 if st.session_state.use_mock_corpus else 0
        )
        st.session_state.use_mock_corpus = "Mock" in use_mock
        
        st.markdown("---")
        
        if st.session_state.current_telemetry:
            telem = st.session_state.current_telemetry
            
            # Intent
            if telem.get("intent"):
                intent_data = telem["intent"]
                st.markdown("**Intent:**")
                st.text(f"{intent_data['intent']} ({intent_data['confidence']:.2f})")
                st.markdown("")
            
            # Top doc IDs
            if telem.get("bm25_docs"):
                st.markdown("**BM25 Top Docs:**")
                for doc_id in telem["bm25_docs"][:3]:
                    st.text(f"• {doc_id}")
                st.markdown("")
            
            if telem.get("knn_docs"):
                st.markdown("**kNN Top Docs:**")
                for doc_id in telem["knn_docs"][:3]:
                    st.text(f"• {doc_id}")
                st.markdown("")
            
            # Stage latencies
            if telem.get("latencies"):
                st.markdown("**Stage Latencies:**")
                latencies = telem["latencies"]
                st.text(f"Intent: {latencies['intent']:.0f}ms")
                st.text(f"BM25: {latencies['bm25']:.0f}ms")
                st.text(f"kNN: {latencies['knn']:.0f}ms")
                st.text(f"Fuse: {latencies['fuse']:.0f}ms")
                st.text(f"LLM: {latencies['llm']:.0f}ms")
                st.text(f"**Total: {latencies['total']:.0f}ms**")
            
            # Token counts
            if telem.get("tokens"):
                st.markdown("**Tokens:**")
                tokens = telem["tokens"]
                st.text(f"In: {tokens['in']}")
                st.text(f"Out: {tokens['out']}")
            
            # Raw context toggle
            st.markdown("---")
            show_raw = st.toggle("Show raw context", value=st.session_state.show_raw_context)
            st.session_state.show_raw_context = show_raw
            
            # Budget warnings
            if telem.get("budget_warning"):
                st.warning("⚠️ Response time exceeded budget (>5s)")
            
            # Error warnings  
            if telem.get("errors", 0) > 0:
                st.error(f"❌ {telem['errors']} stage error(s)")
                
        else:
            st.info("No telemetry data yet. Send a message to see diagnostics.")


def render_warning_banner(telemetry: Dict[str, Any]) -> None:
    """Render non-blocking warning banner if budget exceeded or errors occurred."""
    warnings = []
    
    if telemetry.get("budget_warning"):
        warnings.append("Response time exceeded budget (>5s)")
    
    if telemetry.get("errors", 0) > 0:
        warnings.append(f"{telemetry['errors']} stage error(s) occurred")
    
    if warnings:
        st.warning(" • ".join(warnings))


def simulate_response(user_input: str) -> str:
    """Simulate response based on user input."""
    if "start" in user_input.lower() and "service" in user_input.lower():
        return """To start utility service, you'll need to follow these steps:

1. **Gather Required Documents**: Valid government ID, proof of property ownership or lease, and recent credit report
2. **Credit Check**: Customers with credit scores >650 can skip deposit requirements  
3. **Submit Application**: Apply online, by phone, or at customer service centers
4. **Schedule Installation**: Same-day activation available for standard residential customers
5. **Complete Setup**: Field technician will set up service and verify meter installation

**Special Cases:**
- Corporate accounts require business license and follow an expedited pathway
- Students can use parent/guardian co-signature
- Military personnel receive expedited processing

The process has been streamlined in 2025 to reduce wait times and simplify requirements."""
    
    elif "api" in user_input.lower():
        return """Here are the key utility APIs available:

**Account Utility API (v3)**
- Endpoint: `GET /api/v3/account/{account_number}/utility`
- Returns: service type, rate schedule, meter details
- Authorization required

**Customer Summary API (v2)** 
- Endpoint: `GET /api/v2/customer/{id}/summary`
- Returns: account balance, service status, payment history
- Rate limit: 1000 req/min per API key

**Enhanced Transaction API (v1)**
- Endpoint: `POST /api/v1/transactions/search`
- Supports: date ranges, amount filters, transaction types
- Returns: paginated transaction results

All APIs require proper authorization headers and follow standard REST conventions."""
    
    else:
        return f"""I can help you with information about utility services, APIs, and procedures. 

Some things you can ask about:
- Starting or stopping utility service
- Payment plans and billing questions
- API documentation and endpoints
- Safety procedures and requirements
- Rate structures and time-of-use plans

What specific information are you looking for?"""


def main():
    """Main chat interface."""
    st.set_page_config(
        page_title="Utilities Assistant",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session
    initialize_session()
    
    # Render diagnostics sidebar
    render_diagnostics_sidebar()
    
    # Main chat interface
    st.title("⚡ Utilities Knowledge Assistant")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                render_source_chips(message["sources"])
    
    # Chat input
    user_input = st.chat_input("Ask about utilities, APIs, or procedures...")
    
    # Process input
    if user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Show thinking spinner and generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Simulate processing time
                time.sleep(1.2)
                
                # Generate response
                response = simulate_response(user_input)
                
                # Set telemetry
                st.session_state.current_telemetry = MOCK_TELEMETRY
            
            # Display response
            st.markdown(response)
            
            # Display sources
            render_source_chips(MOCK_SOURCES)
            
            # Show raw context if enabled
            if st.session_state.show_raw_context:
                with st.expander("Raw Context", expanded=False):
                    st.text(MOCK_CONTEXT)
            
            # Show warning banner (demo - no warnings)
            render_warning_banner(MOCK_TELEMETRY)
        
        # Add assistant response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": MOCK_SOURCES
        })
        
        # Trim history
        if len(st.session_state.messages) > 20:
            st.session_state.messages = st.session_state.messages[-20:]
        
        st.rerun()


if __name__ == "__main__":
    main()