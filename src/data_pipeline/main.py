# main.py - Main Streamlit application for utilities assistance chat interface
import streamlit as st
import logging
from datetime import datetime
import time

from cachetools import TTLCache

from chat_interface import load_synonyms
from token_manager import TokenManager
from chat_interface import generate_response
from client_manager import ClientSingleton
import asyncio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
st.set_page_config(layout="wide")


async def stream_response(response, placeholder):
    full_response = ""
    async for chunk in response:  # Iterate over the async generator
        full_response += chunk
        # Update the placeholder with the current response
        placeholder.markdown(full_response.strip() + "Œ", unsafe_allow_html=True)
        await asyncio.sleep(0.05)  # Adjust the sleep time for smoother streaming
    
    placeholder.markdown(full_response.strip(), unsafe_allow_html=True)  # Finalize the response
    
    return full_response.strip()


# ---------- Application Initialization ---------- #

async def initialize_application():
    """
    Initialize the application by pre-fetching the token, pre-computing embeddings, and warming up the model.
    """
    start_time = time.time()
    logging.info("Initializing application...")
    cache = TTLCache(maxsize=5, ttl=60)
    load_synonyms()
    
    token_manager = TokenManager()
    token_manager.pre_fetch_token()  # Pre-fetch the token during startup
    token_manager.start_background_refresh()  # Start the background token refresh task
    logging.info("Token pre-fetching completed in %.2f seconds.", time.time() - start_time)
    # Initialize Azure clients
    client = ClientSingleton.get_instance(token_manager)
    client.get_chat_client()  # Initialize AzureChatOpenAI client
    client.get_embeddings_client()  # Initialize AzureOpenAIEmbeddings client
    logging.info("AzureChatOpenAI and AzureOpenAIEmbeddings clients initialized.")
    
    logging.info("Application initialized successfully.")


# ---------- Chat UI & Export ---------- #

def render_message(msg):
    role = msg["role"]
    content = msg["content"]
    
    # Style and labels based on role
    if role == "user":
        alignment = "right"
        color = "#007BFF"  # Bootstrap blue
        text_color = "white"
        label = "You"
    else:
        alignment = "left"
        color = "#e0e0e0"  # Light gray
        text_color = "black"
        label = "Assistant"
    
    # CSS to create a custom footer on the bottom of the viewport
    footer_style = """
    <style>
    .custom-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f2f2f2;
        color: #555;
        text-align: center;
        z-index: 9999;
    }
    </style>
    """
    
    # The footer HTML snippet
    footer_html = """
    <div class="custom-footer">
        <p>If you see error from assistant or wish to restart the interaction and reset the context, Please type "Start Over". </p>
    </div>
    """
    # Render the CSS and the footer HTML
    st.markdown(footer_style + footer_html, unsafe_allow_html=True)
    
    st.markdown(
        f"""
        <div style='display: flex; justify-content: {alignment}; margin: 0px 0;'>
            <div style='background-color: {color}; color: {text_color};
            padding: 10px 14px; border-radius: 12px;
            max-width: 75%; font-size: 15px;'>
                <strong>{label}:</strong> {content}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def open_popup():
    st.session_state.show_popup = True


async def main():
    # #check and reload browser
    #     INACTIVITY_TIMEOUT_MS: int = 1800000
    #
    #     # The JavaScript listens for various events and resets the timer on each event.
    #     components.html(
    #         f"""
    #         <script>
    #         let inactivityTimer;
    #
    #         function resetInactivityTimer() {{
    #             clearTimeout(inactivityTimer);
    #             inactivityTimer = setTimeout(() => {{
    #                 // Action to perform after inactivity timeout
    #                 // For example: reload the page or display a message
    #                 window.location.reload();
    #                 console.log('[INFO]: ${"Page Reloaded"} ');
    #
    #             }}, {INACTIVITY_TIMEOUT_MS});
    #         }}
    #
    #         // List of events that indicate user activity
    
    # Initialize the application
    await initialize_application()
    
    # Initialize session state
    if "token_manager" not in st.session_state:
        st.session_state.token_manager = TokenManager()
        st.session_state.token_manager.pre_fetch_token()
        st.session_state.token_manager.start_background_refresh()
    
    if "aws_auth" not in st.session_state:
        st.session_state.aws_auth = ClientSingleton.get_awsauth()
    
    st.markdown(
        """
        <div style='display: flex; justify-content: center; align-items: center; height: 100px;'>
            <h1 style='color: #007BFF; text-align: center;'>Digital Knowledge Hub</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}
    
    if "active_session" not in st.session_state:
        st.session_state.active_session = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    if st.session_state.active_session not in st.session_state.chat_sessions:
        st.session_state.chat_sessions[st.session_state.active_session] = []
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Render the initial chat history
    for msg in st.session_state.chat_history:
        render_message(msg)
    
    user_input = st.chat_input("Ask your question...")
    
    if user_input and user_input.strip():
        user_input = user_input.strip()
        # Append user input to chat history and render it
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        render_message({"role": "user", "content": user_input})
        
        placeholder = st.empty()
        with placeholder.container():
            st.markdown(
                """
                <div style='
                background-color: #e0e0e0;
                color: black;
                padding: 10px 14px;
                border-radius: 12px;
                max-width: 75%;
                font-size: 15px;
                margin: 0px 0;
                '>
                <strong>Assistant:</strong> <em>Thinking...</em>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        try:
            start_time = time.time()
            response_stream = await generate_response(user_input, st.session_state.token_manager,
                                                     st.session_state.aws_auth)
            
            streamed = await stream_response(response_stream, placeholder)
            logging.info("Generated response: {streamed}")
            # follow_up_questions = parse_follow_up_questions(streamed)
            # display_follow_up_questions(follow_up_questions)
            end_time = time.time()
            logging.info(f"Response generated and streamed in {end_time - start_time:.2f} seconds.")
            # Append assistant response to chat history and render it
            st.session_state.chat_history.append({"role": "assistant", "content": streamed})
            render_message({"role": "assistant", "content": streamed})
        except Exception as e:
            generic_message = "I'm sorry, I don't have the information you're looking for. Could you please rephrase your question?"
            st.session_state.chat_history.append({"role": "assistant", "content": generic_message})
            render_message({"role": "assistant", "content": generic_message})
        finally:
            placeholder.empty()


if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.get_event_loop().run_until_complete(main())