import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx
from openai import AsyncOpenAI, OpenAI
from anthropic import Anthropic, AsyncAnthropic
import google.generativeai as genai
from google.generativeai.types import content_types
from google.generativeai.types.generation_types import GenerationConfig

from core.settings import settings, ModelProvider, OpenAIModel, AnthropicModel, GoogleModel


def get_model_provider(model: str) -> str:
    """Determine the provider based on the model name."""
    if model in [m.value for m in OpenAIModel]:
        return ModelProvider.OPENAI
    elif model in [m.value for m in AnthropicModel]:
        return ModelProvider.ANTHROPIC
    elif model in [m.value for m in GoogleModel]:
        return ModelProvider.GOOGLE
    else:
        raise ValueError(f"Unsupported model: {model}")


async def get_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.7,
) -> Tuple[str, Dict[str, Any]]:
    """
    Get a completion from the specified model.
    
    Args:
        messages: List of messages in the conversation
        model: Model name to use
        temperature: Temperature for generation
        
    Returns:
        Tuple of (generated text, metadata)
    """
    model = model or settings.DEFAULT_MODEL
    provider = get_model_provider(model)
    
    # Create metadata for tracking
    metadata = {
        "model": model,
        "provider": provider,
    }
    
    if provider == ModelProvider.OPENAI:
        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not provided")
        
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY.get_secret_value())
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": msg["role"], "content": msg["content"]} for msg in messages],
            temperature=temperature,
        )
        return response.choices[0].message.content, metadata
    
    elif provider == ModelProvider.ANTHROPIC:
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("Anthropic API key not provided")
        
        client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY.get_secret_value())
        
        # Convert messages to Anthropic format
        system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
        conversation = [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in messages 
            if msg["role"] in ["user", "assistant"]
        ]
        
        response = await client.messages.create(
            model=model,
            messages=conversation,
            system=system_message,
            temperature=temperature,
            max_tokens=4096,
        )
        return response.content[0].text, metadata
    
    elif provider == ModelProvider.GOOGLE:
        if not settings.GOOGLE_API_KEY:
            raise ValueError("Google API key not provided")
        
        genai.configure(api_key=settings.GOOGLE_API_KEY.get_secret_value())
        model_obj = genai.GenerativeModel(
            model_name=model,
            generation_config=GenerationConfig(temperature=temperature),
        )
        
        # Convert messages to Google format
        prompt_parts = []
        for msg in messages:
            if msg["role"] == "system":
                prompt_parts.append(content_types.FunctionCall(name="system", args={"content": msg["content"]}))
            elif msg["role"] == "user":
                prompt_parts.append(content_types.Part(text=msg["content"], role="user"))
            elif msg["role"] == "assistant":
                prompt_parts.append(content_types.Part(text=msg["content"], role="model"))
        
        response = await asyncio.to_thread(
            model_obj.generate_content,
            prompt_parts,
        )
        return response.text, metadata
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")


async def stream_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.7,
) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
    """
    Stream a completion from the specified model.
    
    Args:
        messages: List of messages in the conversation
        model: Model name to use
        temperature: Temperature for generation
        
    Yields:
        Tuples of (token, metadata)
    """
    model = model or settings.DEFAULT_MODEL
    provider = get_model_provider(model)
    
    # Create metadata for tracking
    metadata = {
        "model": model,
        "provider": provider,
    }
    
    if provider == ModelProvider.OPENAI:
        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not provided")
        
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY.get_secret_value())
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": msg["role"], "content": msg["content"]} for msg in messages],
            temperature=temperature,
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content, metadata
    
    elif provider == ModelProvider.ANTHROPIC:
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("Anthropic API key not provided")
        
        client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY.get_secret_value())
        
        # Convert messages to Anthropic format
        system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
        conversation = [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in messages 
            if msg["role"] in ["user", "assistant"]
        ]
        
        with client.messages.stream(
            model=model,
            messages=conversation,
            system=system_message,
            temperature=temperature,
            max_tokens=4096,
        ) as stream:
            async for text in stream.text_stream:
                yield text, metadata
    
    elif provider == ModelProvider.GOOGLE:
        if not settings.GOOGLE_API_KEY:
            raise ValueError("Google API key not provided")
        
        genai.configure(api_key=settings.GOOGLE_API_KEY.get_secret_value())
        model_obj = genai.GenerativeModel(
            model_name=model,
            generation_config=GenerationConfig(temperature=temperature),
        )
        
        # Convert messages to Google format
        prompt_parts = []
        for msg in messages:
            if msg["role"] == "system":
                prompt_parts.append(content_types.FunctionCall(name="system", args={"content": msg["content"]}))
            elif msg["role"] == "user":
                prompt_parts.append(content_types.Part(text=msg["content"], role="user"))
            elif msg["role"] == "assistant":
                prompt_parts.append(content_types.Part(text=msg["content"], role="model"))
        
        # Google doesn't have a native streaming API in their Python client yet
        # So we're using a fake streaming approach by yielding the response in chunks
        response = await asyncio.to_thread(
            model_obj.generate_content,
            prompt_parts,
        )
        
        # Simulate streaming by yielding chunks of the response
        text = response.text
        chunk_size = max(1, len(text) // 20)  # Split into ~20 chunks
        
        for i in range(0, len(text), chunk_size):
            yield text[i:i+chunk_size], metadata
            await asyncio.sleep(0.05)  # Add a small delay to simulate streaming
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")