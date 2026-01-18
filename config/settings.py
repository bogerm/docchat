from pydantic_settings import BaseSettings
from typing import Dict, Any, Optional
from .constants import MAX_FILE_SIZE, MAX_TOTAL_SIZE, ALLOWED_TYPES

class Settings(BaseSettings):

    # Optional settings with defaults
    MAX_FILE_SIZE: int = MAX_FILE_SIZE
    MAX_TOTAL_SIZE: int = MAX_TOTAL_SIZE
    ALLOWED_TYPES: list = ALLOWED_TYPES

    # Database settings
    CHROMA_DB_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "documents"

    # Retrieval settings
    VECTOR_SEARCH_K: int = 10
    HYBRID_RETRIEVER_WEIGHTS: list = [0.4, 0.6]

    # Logging settings
    LOG_LEVEL: str = "INFO"

    # New cache settings with type annotations
    CACHE_DIR: str = "document_cache"
    CACHE_EXPIRE_DAYS: int = 7

    # IBM Watson settings
    IBM_WATSON_URL: str = "https://us-south.ml.cloud.ibm.com"
    IBM_WATSON_PROJECT_ID: str = "skills-network"
    IBM_WATSON_API_KEY: Optional[str] = None

    # OpenRouter settings
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # OpenAI settings
    OPENAI_API_KEY: Optional[str] = None

    # Ollama settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Model provider selection (ibm_watson, openai, ollama, openrouter)
    DEFAULT_MODEL_PROVIDER: str = "ibm_watson"

    # Model configurations by role
    RESEARCH_MODEL_PROVIDER: Optional[str] = None
    RESEARCH_MODEL_ID: str = "meta-llama/llama-3-2-90b-vision-instruct"
    RESEARCH_MODEL_TEMPERATURE: float = 0.3
    RESEARCH_MODEL_MAX_TOKENS: int = 3000

    VERIFICATION_MODEL_PROVIDER: Optional[str] = None
    VERIFICATION_MODEL_ID: str = "ibm/granite-4-h-small"
    VERIFICATION_MODEL_TEMPERATURE: float = 0.0
    VERIFICATION_MODEL_MAX_TOKENS: int = 2000

    RELEVANCE_MODEL_PROVIDER: Optional[str] = None
    RELEVANCE_MODEL_ID: str = "ibm/granite-3-3-8b-instruct"
    RELEVANCE_MODEL_TEMPERATURE: float = 0.0
    RELEVANCE_MODEL_MAX_TOKENS: int = 16  

    # Ollama-specific settings
    OLLAMA_NUM_CTX: int = 32768  # Context window size for large prompts
    OLLAMA_TIMEOUT: int = 120  # Timeout in seconds

    # Embedding model configurations
    EMBEDDING_MODEL_PROVIDER: Optional[str] = None
    EMBEDDING_MODEL_ID: str = "ibm/slate-125m-english-rtrvr-v2"
    EMBEDDING_TRUNCATE_INPUT_TOKENS: int = 3
    EMBEDDING_RETURN_INPUT_TEXT: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

        
    def _role_params(self, temperature: float, max_tokens: int, provider: str) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if provider == "ollama":
            params["num_ctx"] = self.OLLAMA_NUM_CTX
            params["timeout"] = self.OLLAMA_TIMEOUT
        return params

    def get_model_config(self, role: str) -> Dict[str, Any]:
        """
        Get model configuration for a specific role.
        
        Args:
            role: The model role (research, verification, relevance)
            
        Returns:
            Dictionary with provider, model_id, and params
        """        
        role = role.lower()

        if role == "research":
            provider = (self.RESEARCH_MODEL_PROVIDER or self.DEFAULT_MODEL_PROVIDER).lower()
            return {
                "provider": provider,
                "model_id": self.RESEARCH_MODEL_ID,
                "params": self._role_params(self.RESEARCH_MODEL_TEMPERATURE, self.RESEARCH_MODEL_MAX_TOKENS, provider),
            }

        if role == "verification":
            provider = (self.VERIFICATION_MODEL_PROVIDER or self.DEFAULT_MODEL_PROVIDER).lower()
            return {
                "provider": provider,
                "model_id": self.VERIFICATION_MODEL_ID,
                "params": self._role_params(self.VERIFICATION_MODEL_TEMPERATURE, self.VERIFICATION_MODEL_MAX_TOKENS, provider),
            }

        if role == "relevance":
            provider = (self.RELEVANCE_MODEL_PROVIDER or self.DEFAULT_MODEL_PROVIDER).lower()
            max_tokens = self.RELEVANCE_MODEL_MAX_TOKENS
            if provider == "ollama":
                # Avoid any “total token budget” interpretation in wrappers.
                max_tokens = max(max_tokens, 1000)

            return {
                "provider": provider,
                "model_id": self.RELEVANCE_MODEL_ID,
                "params": self._role_params(
                    self.RELEVANCE_MODEL_TEMPERATURE, 
                    max_tokens, 
                    provider),
            }

        raise ValueError(f"Unknown model role: {role}")


    def get_embedding_config(self) -> Dict[str, Any]:
        """
        Get embedding model configuration.
        
        Returns:
            Dictionary with provider, model_id, and params
        """
        provider = (self.EMBEDDING_MODEL_PROVIDER or self.DEFAULT_MODEL_PROVIDER).lower()
        
        return {
            "provider": provider,
            "model_id": self.EMBEDDING_MODEL_ID,
            "params": {
                "truncate_input_tokens": self.EMBEDDING_TRUNCATE_INPUT_TOKENS,
                "return_options": {"input_text": self.EMBEDDING_RETURN_INPUT_TEXT}
            }
        }

settings = Settings()