import json
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Annotated, Dict, List, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from core import settings, get_completion, stream_completion
from schema import (
    ChatHistory, 
    ChatHistoryInput,
    ChatMessage, 
    ServiceInfo, 
    StreamInput, 
    UserInput
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Create router with authentication
router = APIRouter()

# In-memory storage for chat threads
chat_threads: Dict[str, List[Dict[str, str]]] = {}


def verify_bearer(
    http_auth: Annotated[
        Optional[HTTPAuthorizationCredentials],
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.", auto_error=False)),
    ],
) -> None:
    """Verify Bearer token if AUTH_SECRET is set."""
    if not settings.AUTH_SECRET:
        return
    auth_secret = settings.AUTH_SECRET.get_secret_value()
    if not http_auth or http_auth.credentials != auth_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


def get_thread_messages(thread_id: str) -> List[Dict[str, str]]:
    """Get messages for a thread, creating it if it doesn't exist."""
    if thread_id not in chat_threads:
        chat_threads[thread_id] = []
    return chat_threads[thread_id]


@router.get("/info")
async def info() -> ServiceInfo:
    """Get information about the service."""
    return ServiceInfo(
        models=settings.AVAILABLE_MODELS,
        default_model=settings.DEFAULT_MODEL,
    )


@router.post("/chat")
async def chat(user_input: UserInput) -> ChatMessage:
    """Handle a chat request."""
    try:
        # Generate a thread ID if not provided
        thread_id = user_input.thread_id or str(uuid.uuid4())
        
        # Get existing messages for this thread
        messages = get_thread_messages(thread_id)
        
        # Add the new message
        user_message = {"role": "user", "content": user_input.message}
        messages.append(user_message)
        
        # Get completion from LLM
        content, metadata = await get_completion(
            messages=messages,
            model=user_input.model,
            temperature=user_input.temperature,
        )
        
        # Create assistant message
        assistant_message = {"role": "assistant", "content": content}
        
        # Add to thread history
        messages.append(assistant_message)
        
        # Return the response
        return ChatMessage(
            role="assistant",
            content=content,
            metadata={
                **metadata,
                "thread_id": thread_id,
            },
        )
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}",
        )


async def stream_generator(
    user_input: StreamInput,
) -> AsyncGenerator[str, None]:
    """Generate stream response for chat streaming."""
    try:
        # Generate a thread ID if not provided
        thread_id = user_input.thread_id or str(uuid.uuid4())
        
        # Get existing messages for this thread
        messages = get_thread_messages(thread_id)
        
        # Add the new message
        user_message = {"role": "user", "content": user_input.message}
        messages.append(user_message)
        
        # Prepare for collecting the full assistant response
        full_response = ""
        
        # Stream tokens from the LLM
        async for token, metadata in stream_completion(
            messages=messages,
            model=user_input.model,
            temperature=user_input.temperature,
        ):
            full_response += token
            
            if user_input.stream_tokens:
                # Yield token events
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                
        # Add completed message to thread history
        assistant_message = {"role": "assistant", "content": full_response}
        messages.append(assistant_message)
        
        # Yield final message event
        final_message = ChatMessage(
            role="assistant",
            content=full_response,
            metadata={
                "thread_id": thread_id,
                "model": user_input.model or settings.DEFAULT_MODEL,
            },
        )
        
        yield f"data: {json.dumps({'type': 'message', 'content': final_message.model_dump()})}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Error in streaming: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


@router.post("/stream")
async def stream(user_input: StreamInput) -> StreamingResponse:
    """Stream a chat response."""
    return StreamingResponse(
        stream_generator(user_input),
        media_type="text/event-stream",
    )


@router.post("/history")
async def history(history_input: ChatHistoryInput) -> ChatHistory:
    """Get chat history for a thread."""
    thread_id = history_input.thread_id
    
    if thread_id not in chat_threads:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread {thread_id} not found",
        )
    
    messages = [
        ChatMessage(
            role=msg["role"],
            content=msg["content"],
        )
        for msg in chat_threads[thread_id]
    ]
    
    return ChatHistory(
        messages=messages,
        thread_id=thread_id,
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


# Include router with auth dependency
app.include_router(router, dependencies=[Depends(verify_bearer)])