from typing import Dict, List, Optional, Union, Any
import json

from openai import (
    APIError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential
import aiohttp

from app.config import LLMSettings, config
from app.logger import logger
from app.schema import Message, TOOL_CHOICE_TYPE, ROLE_VALUES, TOOL_CHOICE_VALUES, ToolChoice


class LLMProvider:
    """Base class for LLM providers"""
    
    async def create_client(self, llm_config: LLMSettings):
        """Create a client for the LLM provider"""
        raise NotImplementedError("Subclasses must implement create_client")
    
    async def format_request(self, messages, system_msgs=None, **kwargs):
        """Format the request for the LLM provider"""
        raise NotImplementedError("Subclasses must implement format_request")
    
    async def format_response(self, response):
        """Format the response from the LLM provider"""
        raise NotImplementedError("Subclasses must implement format_response")
    
    async def send_request(self, request_data, client):
        """Send the request to the LLM provider"""
        raise NotImplementedError("Subclasses must implement send_request")
    
    async def send_streaming_request(self, request_data, client):
        """Send a streaming request to the LLM provider"""
        raise NotImplementedError("Subclasses must implement send_streaming_request")


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI models"""
    
    async def create_client(self, llm_config: LLMSettings):
        """Create an OpenAI client"""
        return AsyncOpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)
    
    async def format_request(self, messages, system_msgs=None, **kwargs):
        """Format request for OpenAI API"""
        if system_msgs:
            messages = system_msgs + messages
        
        request_data = {
            "model": kwargs.get("model"),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.0),
        }
        
        # Add tools if provided
        if "tools" in kwargs and kwargs["tools"]:
            request_data["tools"] = kwargs["tools"]
            
        # Add tool_choice if provided
        if "tool_choice" in kwargs:
            request_data["tool_choice"] = kwargs["tool_choice"]
            
        return request_data
    
    async def format_response(self, response):
        """Format response from OpenAI API"""
        return response.choices[0].message
    
    async def send_request(self, request_data, client):
        """Send request to OpenAI API"""
        response = await client.chat.completions.create(**request_data)
        return await self.format_response(response)
    
    async def send_streaming_request(self, request_data, client):
        """Send streaming request to OpenAI API"""
        request_data["stream"] = True
        response = await client.chat.completions.create(**request_data)
        
        collected_messages = []
        async for chunk in response:
            chunk_message = chunk.choices[0].delta.content or ""
            collected_messages.append(chunk_message)
            print(chunk_message, end="", flush=True)

        print()  # Newline after streaming
        return "".join(collected_messages).strip()


class AzureOpenAIProvider(OpenAIProvider):
    """Provider for Azure OpenAI models"""
    
    async def create_client(self, llm_config: LLMSettings):
        """Create an Azure OpenAI client"""
        return AsyncAzureOpenAI(
            base_url=llm_config.base_url,
            api_key=llm_config.api_key,
            api_version=llm_config.api_version,
        )


class AnthropicProvider(LLMProvider):
    """Provider for Anthropic Claude models"""
    
    async def create_client(self, llm_config: LLMSettings):
        """Create an HTTP client for Anthropic API"""
        return aiohttp.ClientSession(
            headers={
                "X-API-Key": llm_config.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
        )
    
    async def format_request(self, messages, system_msgs=None, **kwargs):
        """Format request for Anthropic API"""
        # Convert to Anthropic message format
        formatted_messages = []
        system_prompt = None
        
        # Extract system message if present
        if system_msgs:
            for msg in system_msgs:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    system_prompt = msg.get("content", "")
                    break
                elif hasattr(msg, "role") and msg.role == "system":
                    system_prompt = msg.content
                    break
        
        # Format the main messages
        for message in messages:
            if isinstance(message, dict):
                role = message.get("role")
                content = message.get("content", "")
                
                if role == "system":
                    system_prompt = content
                elif role == "user":
                    formatted_messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    formatted_messages.append({"role": "assistant", "content": content})
            elif hasattr(message, "role"):
                if message.role == "system":
                    system_prompt = message.content
                elif message.role == "user":
                    formatted_messages.append({"role": "user", "content": message.content})
                elif message.role == "assistant":
                    formatted_messages.append({"role": "assistant", "content": message.content})
        
        request_data = {
            "model": kwargs.get("model"),
            "messages": formatted_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.0),
        }
        
        # Add system prompt if present
        if system_prompt:
            request_data["system"] = system_prompt
            
        # Add tools if requested
        if "tools" in kwargs and kwargs["tools"]:
            request_data["tools"] = kwargs["tools"]
        
        return request_data
    
    async def format_response(self, response):
        """Format response from Anthropic API"""
        content = response.get("content", [{"text": ""}])
        return Message.assistant_message(content[0].get("text", ""))
    
    async def send_request(self, request_data, client):
        """Send request to Anthropic API"""
        async with client.post(
            "https://api.anthropic.com/v1/messages",
            json=request_data
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise APIError(f"Anthropic API error: {error_text}")
            
            result = await response.json()
            return await self.format_response(result)
    
    async def send_streaming_request(self, request_data, client):
        """Send streaming request to Anthropic API"""
        request_data["stream"] = True
        
        async with client.post(
            "https://api.anthropic.com/v1/messages",
            json=request_data
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise APIError(f"Anthropic API error: {error_text}")
            
            collected_content = []
            async for line in response.content:
                if line.startswith(b"data: "):
                    data = line[6:].decode('utf-8')
                    if data.strip() == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(data)
                        if "content" in chunk and len(chunk["content"]) > 0:
                            text = chunk["content"][0].get("text", "")
                            if text:
                                collected_content.append(text)
                                print(text, end="", flush=True)
                    except json.JSONDecodeError:
                        continue
            
            print()  # Newline after streaming
            return "".join(collected_content).strip()


class GoogleProvider(LLMProvider):
    """Provider for Google Gemini models"""
    
    async def create_client(self, llm_config: LLMSettings):
        """Create an HTTP client for Google AI API"""
        return aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {llm_config.api_key}",
                "Content-Type": "application/json",
            }
        )
    
    async def format_request(self, messages, system_msgs=None, **kwargs):
        """Format request for Google AI API"""
        formatted_messages = []
        
        # Add system message as a user message with [system] prefix
        if system_msgs:
            for msg in system_msgs:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    formatted_messages.append({
                        "role": "user",
                        "parts": [{"text": f"[system] {msg.get('content', '')}"}]
                    })
                elif hasattr(msg, "role") and msg.role == "system":
                    formatted_messages.append({
                        "role": "user",
                        "parts": [{"text": f"[system] {msg.content}"}]
                    })
        
        # Format the main messages
        for message in messages:
            if isinstance(message, dict):
                role = message.get("role")
                content = message.get("content", "")
                
                if role == "user":
                    formatted_messages.append({"role": "user", "parts": [{"text": content}]})
                elif role == "assistant":
                    formatted_messages.append({"role": "model", "parts": [{"text": content}]})
            elif hasattr(message, "role"):
                if message.role == "user":
                    formatted_messages.append({"role": "user", "parts": [{"text": message.content}]})
                elif message.role == "assistant":
                    formatted_messages.append({"role": "model", "parts": [{"text": message.content}]})
        
        # Build request data
        model_id = kwargs.get("model", "gemini-pro")
        base_url = kwargs.get("base_url", "https://generativelanguage.googleapis.com/v1")
        
        request_data = {
            "contents": formatted_messages,
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.0),
                "maxOutputTokens": kwargs.get("max_tokens", 4096),
            },
        }
        
        # Additional metadata needed for actual request
        request_data["_url"] = f"{base_url}/models/{model_id}:generateContent"
        
        return request_data
    
    async def format_response(self, response):
        """Format response from Google AI API"""
        try:
            candidates = response.get("candidates", [])
            if not candidates:
                return Message.assistant_message("")
            
            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return Message.assistant_message(text)
        except Exception as e:
            logger.error(f"Error formatting Google response: {e}")
            return Message.assistant_message("")
    
    async def send_request(self, request_data, client):
        """Send request to Google AI API"""
        # Extract the URL and remove it from the data
        url = request_data.pop("_url")
        
        async with client.post(url, json=request_data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise APIError(f"Google AI API error: {error_text}")
            
            result = await response.json()
            return await self.format_response(result)
    
    async def send_streaming_request(self, request_data, client):
        """Send streaming request to Google AI API"""
        # Extract the URL and remove it from the data
        url = request_data.pop("_url")
        request_data["generationConfig"]["streamGenerationConfig"] = {"streamContentTokens": True}
        
        async with client.post(f"{url}:streamGenerateContent", json=request_data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise APIError(f"Google AI API error: {error_text}")
            
            collected_content = []
            async for line in response.content:
                try:
                    chunk = json.loads(line)
                    candidates = chunk.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        if parts:
                            text = parts[0].get("text", "")
                            if text:
                                collected_content.append(text)
                                print(text, end="", flush=True)
                except json.JSONDecodeError:
                    continue
            
            print()  # Newline after streaming
            return "".join(collected_content).strip()


class MistralProvider(LLMProvider):
    """Provider for Mistral AI models"""
    
    async def create_client(self, llm_config: LLMSettings):
        """Create an HTTP client for Mistral AI API"""
        return aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {llm_config.api_key}",
                "Content-Type": "application/json",
            }
        )
    
    async def format_request(self, messages, system_msgs=None, **kwargs):
        """Format request for Mistral AI API"""
        formatted_messages = []
        
        # Add system message if present
        if system_msgs:
            for msg in system_msgs:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    formatted_messages.append({
                        "role": "system",
                        "content": msg.get("content", "")
                    })
                elif hasattr(msg, "role") and msg.role == "system":
                    formatted_messages.append({
                        "role": "system",
                        "content": msg.content
                    })
        
        # Format the main messages
        for message in messages:
            if isinstance(message, dict):
                role = message.get("role")
                content = message.get("content", "")
                
                if role == "user":
                    formatted_messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    formatted_messages.append({"role": "assistant", "content": content})
            elif hasattr(message, "role"):
                if message.role == "user":
                    formatted_messages.append({"role": "user", "content": message.content})
                elif message.role == "assistant":
                    formatted_messages.append({"role": "assistant", "content": message.content})
        
        request_data = {
            "model": kwargs.get("model", "mistral-large-latest"),
            "messages": formatted_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.0),
        }
        
        return request_data
    
    async def format_response(self, response):
        """Format response from Mistral AI API"""
        try:
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            return Message.assistant_message(content)
        except Exception as e:
            logger.error(f"Error formatting Mistral response: {e}")
            return Message.assistant_message("")
    
    async def send_request(self, request_data, client):
        """Send request to Mistral AI API"""
        async with client.post(
            "https://api.mistral.ai/v1/chat/completions",
            json=request_data
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise APIError(f"Mistral AI API error: {error_text}")
            
            result = await response.json()
            return await self.format_response(result)
    
    async def send_streaming_request(self, request_data, client):
        """Send streaming request to Mistral AI API"""
        request_data["stream"] = True
        
        async with client.post(
            "https://api.mistral.ai/v1/chat/completions",
            json=request_data
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise APIError(f"Mistral AI API error: {error_text}")
            
            collected_content = []
            async for line in response.content:
                if line.startswith(b"data: "):
                    data = line[6:].decode('utf-8')
                    if data.strip() == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            collected_content.append(content)
                            print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        continue
            
            print()  # Newline after streaming
            return "".join(collected_content).strip()


class OllamaProvider(LLMProvider):
    """Provider for Ollama models"""
    
    async def create_client(self, llm_config: LLMSettings):
        """Create an HTTP client for Ollama API"""
        return aiohttp.ClientSession(
            headers={"Content-Type": "application/json"}
        )
    
    async def format_request(self, messages, system_msgs=None, **kwargs):
        """Format request for Ollama API"""
        formatted_messages = []
        system_prompt = None
        
        # Extract system message if present
        if system_msgs:
            for msg in system_msgs:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    system_prompt = msg.get("content", "")
                    break
                elif hasattr(msg, "role") and msg.role == "system":
                    system_prompt = msg.content
                    break
        
        # Format the main messages
        for message in messages:
            if isinstance(message, dict):
                role = message.get("role")
                content = message.get("content", "")
                
                if role == "system":
                    system_prompt = content
                elif role == "user":
                    formatted_messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    formatted_messages.append({"role": "assistant", "content": content})
            elif hasattr(message, "role"):
                if message.role == "system":
                    system_prompt = message.content
                elif message.role == "user":
                    formatted_messages.append({"role": "user", "content": message.content})
                elif message.role == "assistant":
                    formatted_messages.append({"role": "assistant", "content": message.content})
        
        request_data = {
            "model": kwargs.get("model", "llama3"),
            "messages": formatted_messages,
            "options": {
                "num_predict": kwargs.get("max_tokens", 4096),
                "temperature": kwargs.get("temperature", 0.0),
            }
        }
        
        # Add system prompt if present
        if system_prompt:
            request_data["system"] = system_prompt
            
        base_url = kwargs.get("base_url", "http://localhost:11434")
        request_data["_url"] = f"{base_url}/api/chat"
        
        return request_data
    
    async def format_response(self, response):
        """Format response from Ollama API"""
        try:
            content = response.get("message", {}).get("content", "")
            return Message.assistant_message(content)
        except Exception as e:
            logger.error(f"Error formatting Ollama response: {e}")
            return Message.assistant_message("")
    
    async def send_request(self, request_data, client):
        """Send request to Ollama API"""
        # Extract the URL and remove it from the data
        url = request_data.pop("_url")
        
        async with client.post(url, json=request_data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise APIError(f"Ollama API error: {error_text}")
            
            result = await response.json()
            return await self.format_response(result)
    
    async def send_streaming_request(self, request_data, client):
        """Send streaming request to Ollama API"""
        # Extract the URL and remove it from the data
        url = request_data.pop("_url")
        request_data["stream"] = True
        
        async with client.post(url, json=request_data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise APIError(f"Ollama API error: {error_text}")
            
            collected_content = []
            async for line in response.content:
                try:
                    chunk = json.loads(line)
                    if "message" in chunk:
                        content = chunk["message"].get("content", "")
                        if content:
                            collected_content.append(content)
                            print(content, end="", flush=True)
                except json.JSONDecodeError:
                    continue
            
            print()  # Newline after streaming
            return "".join(collected_content).strip()


class LLM:
    _instances: Dict[str, "LLM"] = {}
    _provider_classes = {
        "openai": OpenAIProvider,
        "azure": AzureOpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "mistral": MistralProvider,
        "ollama": OllamaProvider,
    }

    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if not hasattr(self, "provider"):  # Only initialize if not already initialized
            llm_config_dict = llm_config or config.llm
            llm_settings = llm_config_dict.get(config_name, llm_config_dict["default"])
            
            self.model = llm_settings.model
            self.max_tokens = llm_settings.max_tokens
            self.temperature = llm_settings.temperature
            self.api_type = llm_settings.api_type.lower()
            self.api_key = llm_settings.api_key
            self.api_version = llm_settings.api_version
            self.base_url = llm_settings.base_url
            
            # Set the appropriate provider based on api_type
            self.provider = self._provider_classes.get(self.api_type, OpenAIProvider)()
            
            # Client will be initialized on first use
            self.client = None

    async def _ensure_client(self):
        """Ensure client is initialized"""
        if self.client is None:
            llm_settings = LLMSettings(
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                api_type=self.api_type,
                api_version=self.api_version,
            )
            self.client = await self.provider.create_client(llm_settings)
        return self.client

    @staticmethod
    def format_messages(messages: List[Union[dict, Message]]) -> List[dict]:
        """
        Format messages for LLM by converting them to OpenAI message format.
        """
        formatted_messages = []

        for message in messages:
            if isinstance(message, dict):
                # If message is already a dict, ensure it has required fields
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")
                formatted_messages.append(message)
            elif isinstance(message, Message):
                # If message is a Message object, convert it to dict
                formatted_messages.append(message.to_dict())
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # Validate all messages have required fields
        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"Invalid role: {msg['role']}")
            if "content" not in msg and "tool_calls" not in msg:
                raise ValueError(
                    "Message must contain either 'content' or 'tool_calls'"
                )

        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt to the LLM and get the response.
        """
        try:
            # Format system and user messages
            formatted_messages = self.format_messages(messages)
            formatted_system_msgs = None
            if system_msgs:
                formatted_system_msgs = self.format_messages(system_msgs)
            
            client = await self._ensure_client()
            
            # Prepare common parameters
            request_params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": temperature or self.temperature,
                "base_url": self.base_url,
            }
            
            # Format request based on provider
            request_data = await self.provider.format_request(
                formatted_messages, 
                formatted_system_msgs,
                **request_params
            )
            
            if stream:
                full_response = await self.provider.send_streaming_request(request_data, client)
                if not full_response:
                    raise ValueError("Empty response from streaming LLM")
                return full_response
            else:
                response = await self.provider.send_request(request_data, client)
                
                if hasattr(response, "content"):
                    return response.content
                
                return str(response)

        except ValueError as ve:
            logger.error(f"Validation error: {ve}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask: {e}")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO, # type: ignore
        temperature: Optional[float] = None,
        **kwargs,
    ):
        """
        Ask LLM using functions/tools and return the response.
        """
        try:
            # Validate tool_choice
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            # Format messages
            formatted_messages = self.format_messages(messages)
            formatted_system_msgs = None
            if system_msgs:
                formatted_system_msgs = self.format_messages(system_msgs)

            # Validate tools if provided
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")
            
            client = await self._ensure_client()
            
            # Currently, only OpenAI and potentially Azure support functions/tools
            if self.api_type not in ["openai", "azure"]:
                logger.warning(f"Tool calling may not be fully supported for provider: {self.api_type}")
            
            # Prepare common parameters
            request_params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": temperature or self.temperature,
                "tools": tools,
                "tool_choice": tool_choice,
                "timeout": timeout,
                "base_url": self.base_url,
                **kwargs
            }
            
            # Format request based on provider
            request_data = await self.provider.format_request(
                formatted_messages, 
                formatted_system_msgs,
                **request_params
            )
            
            response = await self.provider.send_request(request_data, client)
            return response

        except ValueError as ve:
            logger.error(f"Validation error in ask_tool: {ve}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_tool: {e}")
            raise

    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Close any HTTP session-based clients
        if self.client and hasattr(self.client, "close"):
            await self.client.close()
        self.client = None
