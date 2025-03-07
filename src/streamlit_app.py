import asyncio
import os
import urllib.parse
from collections.abc import AsyncGenerator
from typing import Union

import streamlit as st
from dotenv import load_dotenv
from streamlit.runtime.scriptrunner import get_script_run_ctx

from client import LLMClient, LLMClientError
from schema import ChatMessage

# Set up page configuration
st.set_page_config(
    page_title="Simple LLM Chat",
    page_icon="💬",
    layout="wide",
)


async def main():
    """Main application flow."""
    # Initialize application state
    if "llm_client" not in st.session_state:
        load_dotenv()
        service_url = os.getenv("AGENT_URL")
        
        if not service_url:
            host = os.getenv("HOST", "0.0.0.0")
            port = os.getenv("PORT", "8080")
            service_url = f"http://{host}:{port}"
        
        try:
            with st.spinner("Connecting to service..."):
                st.session_state.llm_client = LLMClient(base_url=service_url)
        except LLMClientError as e:
            st.error(f"Error connecting to service at {service_url}: {e}")
            st.info("The service might be starting up. Please try again in a few seconds.")
            st.stop()
    
    # Get or create thread ID
    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = get_script_run_ctx().session_id
            messages = []
        else:
            try:
                history = st.session_state.llm_client.get_history(thread_id=thread_id)
                messages = history.messages
            except LLMClientError:
                st.error("No message history found for this Thread ID.")
                messages = []
        
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id
    
    # Sidebar with configuration
    with st.sidebar:
        st.title("💬 Simple LLM Chat")
        
        if st.session_state.llm_client.info:
            models = st.session_state.llm_client.info.models
            default_model = st.session_state.llm_client.info.default_model
            model_idx = models.index(default_model) if default_model in models else 0
            
            selected_model = st.selectbox(
                "Select model",
                options=models,
                index=model_idx,
            )
        else:
            selected_model = None
            st.warning("Could not load available models.")
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make output more random, lower values more deterministic.",
        )
        
        use_streaming = st.toggle("Stream output", value=True)
        
        @st.dialog("Share conversation")
        def share_dialog():
            session = st.runtime.get_instance()._session_mgr.list_active_sessions()[0]
            st_base_url = urllib.parse.urlunparse(
                [session.client.request.protocol, session.client.request.host, "", "", "", ""]
            )
            
            # If not localhost, switch to https by default
            if not st_base_url.startswith("https") and "localhost" not in st_base_url:
                st_base_url = st_base_url.replace("http", "https")
            
            chat_url = f"{st_base_url}?thread_id={st.session_state.thread_id}"
            st.text_input("Share this URL:", value=chat_url, disabled=True)
            st.info("Copy the above URL to share or revisit this conversation")
        
        if st.button("Share Conversation", use_container_width=True):
            share_dialog()
        
        st.divider()
        st.caption("A simple LLM chat application built with FastAPI and Streamlit.")
    
    # Display chat interface
    st.title("💬 Chat")
    
    # Display welcome message if no messages
    if len(st.session_state.messages) == 0:
        with st.chat_message("assistant"):
            st.write("👋 Hello! I'm an AI assistant. How can I help you today?")
    
    # Display message history
    for msg in st.session_state.messages:
        with st.chat_message(msg.role):
            st.write(msg.content)
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to state and display
        user_msg = ChatMessage(role="user", content=prompt)
        st.session_state.messages.append(user_msg)
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        try:
            with st.chat_message("assistant"):
                if use_streaming:
                    # Handle streaming response
                    placeholder = st.empty()
                    message_content = ""
                    
                    # Create stream iterator
                    stream = st.session_state.llm_client.astream(
                        message=prompt,
                        model=selected_model,
                        thread_id=st.session_state.thread_id,
                        temperature=temperature,
                    )
                    
                    # Display streaming content
                    async for chunk in stream:
                        if isinstance(chunk, str):
                            message_content += chunk
                            placeholder.write(message_content)
                        elif isinstance(chunk, ChatMessage):
                            placeholder.write(chunk.content)
                            st.session_state.messages.append(chunk)
                else:
                    # Handle non-streaming response
                    with st.spinner("Thinking..."):
                        response = await st.session_state.llm_client.achat(
                            message=prompt,
                            model=selected_model,
                            thread_id=st.session_state.thread_id,
                            temperature=temperature,
                        )
                        st.write(response.content)
                        st.session_state.messages.append(response)
        
        except LLMClientError as e:
            st.error(f"Error: {e}")
        
        # Force a rerun to update the UI
        st.rerun()


# Run the main function
if __name__ == "__main__":
    asyncio.run(main())