import threading
import tomllib
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()
WORKSPACE_ROOT = PROJECT_ROOT / "workspace"


class LLMSettings(BaseModel):
    model: str = Field(..., description="Model name")
    base_url: str = Field(..., description="API base URL")
    api_key: str = Field(..., description="API key")
    max_tokens: int = Field(4096, description="Maximum number of tokens per request")
    temperature: float = Field(1.0, description="Sampling temperature")
    api_type: str = Field("openai", description="LLM provider type (openai, azure, anthropic, google, mistral, ollama)")
    api_version: str = Field("", description="API version if needed")


class ProxySettings(BaseModel):
    server: str = Field(None, description="Proxy server address")
    username: Optional[str] = Field(None, description="Proxy username")
    password: Optional[str] = Field(None, description="Proxy password")


class BrowserSettings(BaseModel):
    headless: bool = Field(False, description="Whether to run browser in headless mode")
    disable_security: bool = Field(
        True, description="Disable browser security features"
    )
    extra_chromium_args: List[str] = Field(
        default_factory=list, description="Extra arguments to pass to the browser"
    )
    chrome_instance_path: Optional[str] = Field(
        None, description="Path to a Chrome instance to use"
    )
    wss_url: Optional[str] = Field(
        None, description="Connect to a browser instance via WebSocket"
    )
    cdp_url: Optional[str] = Field(
        None, description="Connect to a browser instance via CDP"
    )
    proxy: Optional[ProxySettings] = Field(
        None, description="Proxy settings for the browser"
    )


class AppConfig(BaseModel):
    llm: Dict[str, LLMSettings]
    browser_config: Optional[BrowserSettings] = Field(
        None, description="Browser configuration"
    )

    class Config:
        arbitrary_types_allowed = True


class Config:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._config = None
                    self._load_initial_config()
                    self._initialized = True

    @staticmethod
    def _get_config_path() -> Path:
        root = PROJECT_ROOT
        config_path = root / "config" / "config.toml"
        if config_path.exists():
            return config_path
        example_path = root / "config" / "config.example.toml"
        if example_path.exists():
            return example_path
        raise FileNotFoundError("No configuration file found in config directory")

    def _load_config(self) -> dict:
        config_path = self._get_config_path()
        with config_path.open("rb") as f:
            return tomllib.load(f)

    def _load_initial_config(self):
        raw_config = self._load_config()
        base_llm = raw_config.get("llm", {})
        
        # Get provider-specific settings
        provider_configs = {}
        
        # Find all provider sections in config
        for provider in ["openai", "anthropic", "google", "mistral", "azure", "ollama"]:
            if provider in raw_config:
                provider_configs[provider] = raw_config[provider]
        
        # Check for any other LLM sections that might be provider configurations
        llm_sections = {k: v for k, v in base_llm.items() if isinstance(v, dict)}
        for section, config_data in llm_sections.items():
            if section not in provider_configs:
                provider_configs[section] = config_data
        
        # Create default settings from global LLM section
        default_settings = {
            "model": base_llm.get("model"),
            "base_url": base_llm.get("base_url"),
            "api_key": base_llm.get("api_key"),
            "max_tokens": base_llm.get("max_tokens", 4096),
            "temperature": base_llm.get("temperature", 1.0),
            "api_type": base_llm.get("api_type", "openai"),
            "api_version": base_llm.get("api_version", ""),
        }
        
        # Determine the default provider from llm section or use "openai" as default
        default_provider = base_llm.get("provider", "openai")
        
        # If the default provider has a specific config, merge it with defaults
        if default_provider in provider_configs:
            for key, value in provider_configs[default_provider].items():
                if key not in ["provider"]:  # Skip provider key to avoid confusion
                    default_settings[key] = value
        
        # Update api_type to match provider if not explicitly set
        if "api_type" not in default_settings or not default_settings["api_type"]:
            default_settings["api_type"] = default_provider

        # Handle browser config
        browser_config = raw_config.get("browser", {})
        browser_settings = None

        if browser_config:
            # Handle proxy settings
            proxy_config = browser_config.get("proxy", {})
            proxy_settings = None

            if proxy_config and proxy_config.get("server"):
                proxy_settings = ProxySettings(
                    **{
                        k: v
                        for k, v in proxy_config.items()
                        if k in ["server", "username", "password"] and v
                    }
                )

            # Filter valid browser config parameters
            valid_browser_params = {
                k: v
                for k, v in browser_config.items()
                if k in BrowserSettings.__annotations__ and v is not None
            }

            # If there is proxy settings, add it to the parameters
            if proxy_settings:
                valid_browser_params["proxy"] = proxy_settings

            # Only create BrowserSettings when there are valid parameters
            if valid_browser_params:
                browser_settings = BrowserSettings(**valid_browser_params)

        # Create the final config dictionary
        llm_config = {"default": default_settings}
        
        # Add provider-specific configurations
        for provider, config_data in provider_configs.items():
            if provider != default_provider:  # Skip default provider as it's already handled
                # Start with default settings
                provider_settings = dict(default_settings)
                
                # Override with provider-specific settings
                for key, value in config_data.items():
                    if key not in ["provider"]:  # Skip provider key
                        provider_settings[key] = value
                
                # Set api_type to provider name if not specified
                if "api_type" not in provider_settings or not provider_settings["api_type"]:
                    provider_settings["api_type"] = provider
                
                llm_config[provider] = provider_settings
        
        # Ensure the config has the required structure
        config_dict = {
            "llm": llm_config,
            "browser_config": browser_settings,
        }

        self._config = AppConfig(**config_dict)

    @property
    def llm(self) -> Dict[str, LLMSettings]:
        return self._config.llm

    @property
    def browser_config(self) -> Optional[BrowserSettings]:
        return self._config.browser_config


config = Config()
