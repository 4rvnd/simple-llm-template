import json
import os
from collections.abc import AsyncGenerator, Generator
from typing import Any, Dict, List, Optional, Union

import httpx

from schema import ChatHistory, ChatHistoryInput, ChatMessage, ServiceInfo, StreamInput, UserInput


class LLMClientError(Exception):
    """Exception raised for errors in the LLM client."""
    pass


class LLMClient:
    """Client for interacting with the LLM service."""
    
    def __init__(
        self,
        base_url: str = "http://0.0.0.0:8080",
        timeout: Optional[float] = None,
        get_info: bool = True,
    ) -> None:
        """
        Initialize the client.
        
        Args:
            base_url: The base URL of the service
            timeout: Request timeout in seconds
            get_info: Whether to fetch service info on initialization
        """
        self.base_url = base_url
        self.auth_secret = os.getenv("AUTH_SECRET")
        self.timeout = timeout
        self.info: Optional[ServiceInfo] = None
        
        if get_info:
            self.retrieve_info()
    
    @property
    def _headers(self) -> Dict[str, str]:
        """Get headers for requests, including authentication if provided."""
        headers = {}
        if self.auth_secret:
            headers["Authorization"] = f"Bearer {self.auth_secret}"
        return headers
    
    def retrieve_info(self) -> None:
        """Fetch service information."""
        try:
            response = httpx.get(
                f"{self.base_url}/info",
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            self.info = ServiceInfo.model_validate(response.json())
        except httpx.HTTPError as e:
            raise LLMClientError(f"Error getting service info: {e}")
    
    async def achat(
        self,
        message: str,
        model: Optional[str] = None,
        thread_id: Optional[str] = None,
        temperature: float = 0.7,
    ) -> ChatMessage:
        """
        Send a chat request asynchronously.
        
        Args:
            message: The message to send
            model: Model to use (optional)
            thread_id: Thread ID for continuing a conversation (optional)
            temperature: Temperature for generation (0.0 to 1.0)
            
        Returns:
            The response message
        """
        request = UserInput(
            message=message,
            model=model,
            thread_id=thread_id,
            temperature=temperature,
        )
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise LLMClientError(f"Error: {e}")
        
        return ChatMessage.model_validate(response.json())
    
    def chat(
        self,
        message: str,
        model: Optional[str] = None,
        thread_id: Optional[str] = None,
        temperature: float = 0.7,
    ) -> ChatMessage:
        """
        Send a chat request synchronously.
        
        Args:
            message: The message to send
            model: Model to use (optional)
            thread_id: Thread ID for continuing a conversation (optional)
            temperature: Temperature for generation (0.0 to 1.0)
            
        Returns:
            The response message
        """
        request = UserInput(
            message=message,
            model=model,
            thread_id=thread_id,
            temperature=temperature,
        )
        
        try:
            response = httpx.post(
                f"{self.base_url}/chat",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMClientError(f"Error: {e}")
        
        return ChatMessage.model_validate(response.json())
    
    def _parse_stream_line(self, line: str) -> Optional[Union[ChatMessage, str]]:
        """Parse a line from the SSE stream."""
        line = line.strip()
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                return None
            
            try:
                parsed = json.loads(data)
            except Exception as e:
                raise LLMClientError(f"Error parsing message from server: {e}")
            
            match parsed["type"]:
                case "message":
                    try:
                        return ChatMessage.model_validate(parsed["content"])
                    except Exception as e:
                        raise LLMClientError(f"Server returned invalid message: {e}")
                case "token":
                    return parsed["content"]
                case "error":
                    raise LLMClientError(parsed["content"])
        
        return None
    
    def stream(
        self,
        message: str,
        model: Optional[str] = None,
        thread_id: Optional[str] = None,
        temperature: float = 0.7,
        stream_tokens: bool = True,
    ) -> Generator[Union[ChatMessage, str], None, None]:
        """
        Stream a chat response synchronously.
        
        Args:
            message: The message to send
            model: Model to use (optional)
            thread_id: Thread ID for continuing a conversation (optional)
            temperature: Temperature for generation (0.0 to 1.0)
            stream_tokens: Whether to stream individual tokens
            
        Yields:
            Either tokens (str) or complete messages (ChatMessage)
        """
        request = StreamInput(
            message=message,
            model=model,
            thread_id=thread_id,
            temperature=temperature,
            stream_tokens=stream_tokens,
        )
        
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/stream",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.strip():
                        parsed = self._parse_stream_line(line)
                        if parsed is None:
                            break
                        yield parsed
        except httpx.HTTPError as e:
            raise LLMClientError(f"Error: {e}")
    
    async def astream(
        self,
        message: str,
        model: Optional[str] = None,
        thread_id: Optional[str] = None,
        temperature: float = 0.7,
        stream_tokens: bool = True,
    ) -> AsyncGenerator[Union[ChatMessage, str], None]:
        """
        Stream a chat response asynchronously.
        
        Args:
            message: The message to send
            model: Model to use (optional)
            thread_id: Thread ID for continuing a conversation (optional)
            temperature: Temperature for generation (0.0 to 1.0)
            stream_tokens: Whether to stream individual tokens
            
        Yields:
            Either tokens (str) or complete messages (ChatMessage)
        """
        request = StreamInput(
            message=message,
            model=model,
            thread_id=thread_id,
            temperature=temperature,
            stream_tokens=stream_tokens,
        )
        
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/stream",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            parsed = self._parse_stream_line(line)
                            if parsed is None:
                                break
                            yield parsed
            except httpx.HTTPError as e:
                raise LLMClientError(f"Error: {e}")
    
    def get_history(
        self,
        thread_id: str,
    ) -> ChatHistory:
        """
        Get chat history for a thread.
        
        Args:
            thread_id: Thread ID for the conversation
            
        Returns:
            Chat history for the thread
        """
        request = ChatHistoryInput(thread_id=thread_id)
        
        try:
            response = httpx.post(
                f"{self.base_url}/history",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMClientError(f"Error: {e}")
        
        return ChatHistory.model_validate(response.json())