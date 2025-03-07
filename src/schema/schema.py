from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Message in a chat conversation."""
    
    role: Literal["system", "user", "assistant"] = Field(
        description="Role of the message.",
        examples=["user", "assistant", "system"],
    )
    content: str = Field(
        description="Content of the message.",
        examples=["Hello, world!"],
    )
    metadata: Dict[str, Any] = Field(
        description="Additional metadata for the message.",
        default_factory=dict,
    )
    
    def pretty_repr(self) -> str:
        """Get a pretty representation of the message."""
        role_title = self.role.title()
        padded = f" {role_title} Message "
        sep_len = (80 - len(padded)) // 2
        sep = "=" * sep_len
        second_sep = sep + "=" if len(padded) % 2 else sep
        title = f"{sep}{padded}{second_sep}"
        return f"{title}\n\n{self.content}"

    def pretty_print(self) -> None:
        """Print the message in a pretty format."""
        print(self.pretty_repr())


class UserInput(BaseModel):
    """User input for the chat service."""
    
    message: str = Field(
        description="User input to the service.",
        examples=["What is the weather in Tokyo?"],
    )
    model: Optional[str] = Field(
        description="LLM model to use.",
        default=None,
        examples=["gpt-4o", "claude-3-haiku-20240307"],
    )
    thread_id: Optional[str] = Field(
        description="Thread ID to continue a conversation.",
        default=None,
        examples=["123e4567-e89b-12d3-a456-426614174000"],
    )
    temperature: float = Field(
        description="Temperature for the model generation.",
        default=0.7,
        ge=0.0,
        le=1.0,
    )


class StreamInput(UserInput):
    """User input for streaming responses."""
    
    stream_tokens: bool = Field(
        description="Whether to stream tokens to the client.",
        default=True,
    )


class ChatHistory(BaseModel):
    """History of a chat conversation."""
    
    messages: List[ChatMessage] = Field(
        description="Messages in the conversation.",
        default_factory=list,
    )
    thread_id: str = Field(
        description="Thread ID for the conversation.",
        examples=["123e4567-e89b-12d3-a456-426614174000"],
    )


class ChatHistoryInput(BaseModel):
    """Input for retrieving chat history."""
    
    thread_id: str = Field(
        description="Thread ID for the conversation.",
        examples=["123e4567-e89b-12d3-a456-426614174000"],
    )


class ServiceInfo(BaseModel):
    """Information about the service."""
    
    models: List[str] = Field(
        description="Available LLM models.",
    )
    default_model: str = Field(
        description="Default model used when none is specified.",
    )