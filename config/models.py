"""
Model Factory for managing different LLM and Embedding providers.

This module implements the Factory design pattern to provide a unified interface
for creating and managing different LLM models and embeddings across providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from enum import Enum

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_ibm import WatsonxEmbeddings

from config.settings import settings
from loguru import logger


class ModelProvider(str, Enum):
    """Supported model providers."""
    IBM_WATSON = "ibm_watson"
    OPENAI = "openai"
    OLLAMA = "ollama"


class ModelRole(str, Enum):
    """Roles for different models in the system."""
    RESEARCH = "research"
    VERIFICATION = "verification"
    RELEVANCE = "relevance"


class BaseLLMWrapper(ABC):
    """Abstract base class for LLM wrappers to provide a unified interface."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from the model given a prompt."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name/identifier."""
        pass


class IBMWatsonWrapper(BaseLLMWrapper):
    """Wrapper for IBM Watson models."""
    
    def __init__(self, model_id: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize IBM Watson model.
        
        Args:
            model_id: IBM Watson model identifier
            params: Model parameters (max_tokens, temperature, etc.)
        """
        self.model_id = model_id
        self.params = params or {}
        self.model_initialized = False
 
    def initialize_model(self):
        """
        Initialize IBM Watson model.
        """
        logger.debug(f"Initializing IBM Watson model: {self.model_id} with params: {self.params}")         
        # Initialize credentials and client
        self.credentials = Credentials(url=settings.IBM_WATSON_URL)
        self.client = APIClient(self.credentials)
        
        # Initialize the model
        self.model = ModelInference(
            model_id=self.model_id,
            credentials=self.credentials,
            project_id=settings.IBM_WATSON_PROJECT_ID,
            params=self.params
        )
    

    def generate(self, prompt: str) -> str:
        """Generate response using IBM Watson model."""

        if not self.model_initialized:
            self.initialize_model()
            self.model_initialized = True

        response = self.model.generate_text(prompt=prompt)
        return response.strip() if response else ""
    
    def get_model_name(self) -> str:
        """Return the IBM Watson model ID."""
        return f"ibm_watson:{self.model_id}"


class OpenAIWrapper(BaseLLMWrapper):
    """Wrapper for OpenAI models."""
    
    def __init__(self, model_id: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize OpenAI model.
        
        Args:
            model_id: OpenAI model identifier (e.g., gpt-4, gpt-3.5-turbo)
            params: Model parameters (temperature, max_tokens, etc.)
        """
        self.model_id = model_id
        self.params = params or {}
        self.model_initialized = False


    def initialize_model(self):    
        # Initialize the ChatOpenAI model

        logger.debug(f"Initializing OpenAI model: {self.model_id} with params: {self.params}")
        self.model = ChatOpenAI(
            model=self.model_id,
            temperature=self.params.get("temperature", 0.3),
            max_tokens=self.params.get("max_tokens", 300),
            api_key=settings.OPENAI_API_KEY
        )
    
    
    def generate(self, prompt: str) -> str:
        """Generate response using OpenAI model."""

        if not self.model_initialized:
            self.initialize_model()
            self.initialized = True
            
        response = self.model.invoke(prompt)
        return response.content.strip() if hasattr(response, 'content') else str(response).strip()
    
    def get_model_name(self) -> str:
        """Return the OpenAI model ID."""
        return f"openai:{self.model_id}"


class OllamaWrapper(BaseLLMWrapper):
    """Wrapper for Ollama models."""
    
    def __init__(self, model_id: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Ollama model.
        
        Args:
            model_id: Ollama model identifier (e.g., llama2, mistral)
            params: Model parameters (temperature, num_predict, etc.)
        """
        self.model_id = model_id
        self.params = params or {}
        self.model_initialized = False


    def initialize_model(self):    
        # Initialize the Ollama model
        logger.debug(f"Initializing Ollama model: {self.model_id} with params: {self.params}")
        self.model = OllamaLLM(
            model=self.model_id,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=self.params.get("temperature", 0.3),
            num_predict=self.params.get("max_tokens", 2000),
            num_ctx=self.params.get("num_ctx", 32768),  # Context window for large prompts
            timeout=self.params.get("timeout", 300)  # Timeout in seconds
        )
    
    def generate(self, prompt: str) -> str:
        """Generate response using Ollama model."""
        logger.debug(f"OllamaWrapper.generate() called with prompt length: {len(prompt)}")

        if not self.model_initialized:
            self.initialize_model()
            self.model_initialized = True

        try:
            response = self.model.invoke(prompt)
            logger.debug(f"Ollama raw response type: {type(response)}, length: {len(response) if response else 0}")
            logger.debug(f"Ollama raw response: {repr(response)[:200]}")
            
            result = response.strip() if isinstance(response, str) else str(response).strip()
            
            if not result:
                logger.warning(f"Ollama returned empty response for prompt length {len(prompt)}. "
                             f"Consider reducing prompt size or increasing timeout.")
            
            logger.debug(f"OllamaWrapper.generate() returning length: {len(result)}")
            return result
        except Exception as e:
            logger.error(f"Error in OllamaWrapper.generate(): {type(e).__name__}: {e}")
            raise
    
    def get_model_name(self) -> str:
        """Return the Ollama model ID."""
        return f"ollama:{self.model_id}"


class ModelFactory:
    """
    Factory class for creating LLM model instances.
    
    This implements the Factory design pattern to provide a centralized way
    to create and manage different LLM providers.
    """
    
    @staticmethod
    def create_model(
        provider: ModelProvider,
        model_id: str,
        params: Optional[Dict[str, Any]] = None
    ) -> BaseLLMWrapper:
        """
        Create a model instance based on the provider.
        
        Args:
            provider: The model provider (IBM_WATSON, OPENAI, OLLAMA)
            model_id: The specific model identifier
            params: Optional model parameters
            
        Returns:
            BaseLLMWrapper: A wrapper instance for the specified model
            
        Raises:
            ValueError: If the provider is not supported
        """
        if provider == ModelProvider.IBM_WATSON:
            return IBMWatsonWrapper(model_id, params)
        elif provider == ModelProvider.OPENAI:
            return OpenAIWrapper(model_id, params)
        elif provider == ModelProvider.OLLAMA:
            return OllamaWrapper(model_id, params)
        else:
            raise ValueError(f"Unsupported model provider: {provider}")
    
    @staticmethod
    def create_model_from_config(role: ModelRole) -> BaseLLMWrapper:
        """
        Create a model instance from configuration settings.
        
        This is the recommended way to create models, as it uses the
        centralized configuration from settings.
        
        Args:
            role: The role for which to create the model (RESEARCH, VERIFICATION, RELEVANCE)
            
        Returns:
            BaseLLMWrapper: A configured model instance
        """
        config = settings.get_model_config(role.value)
        
        return ModelFactory.create_model(
            provider=ModelProvider(config["provider"]),
            model_id=config["model_id"],
            params=config["params"]
        )


# Convenience functions for creating models by role
def get_research_model() -> BaseLLMWrapper:
    """Get the configured research model."""
    return ModelFactory.create_model_from_config(ModelRole.RESEARCH)


def get_verification_model() -> BaseLLMWrapper:
    """Get the configured verification model."""
    return ModelFactory.create_model_from_config(ModelRole.VERIFICATION)


def get_relevance_model() -> BaseLLMWrapper:
    """Get the configured relevance checking model."""
    return ModelFactory.create_model_from_config(ModelRole.RELEVANCE)


# ============================================================================
#                        EMBEDDING MODELS
# ============================================================================

class BaseEmbeddingWrapper(ABC):
    """Abstract base class for embedding model wrappers."""
    
    @abstractmethod
    def get_embeddings(self):
        """Return the underlying embeddings object for LangChain integration."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the embedding model name/identifier."""
        pass


class IBMWatsonEmbeddingWrapper(BaseEmbeddingWrapper):
    """Wrapper for IBM Watson embedding models."""
    
    def __init__(self, model_id: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize IBM Watson embedding model.
        
        Args:
            model_id: IBM Watson embedding model identifier
            params: Embedding parameters
        """
        self.model_id = model_id
        self.params = params or {}
        
        # Set default embedding parameters
        embed_params = {
            EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: self.params.get("truncate_input_tokens", 3),
            EmbedTextParamsMetaNames.RETURN_OPTIONS: self.params.get("return_options", {"input_text": True}),
        }
        
        # Initialize the embeddings
        self.embeddings = WatsonxEmbeddings(
            model_id=model_id,
            url=settings.IBM_WATSON_URL,
            project_id=settings.IBM_WATSON_PROJECT_ID,
            params=embed_params
        )
    
    def get_embeddings(self):
        """Return the WatsonX embeddings object."""
        return self.embeddings
    
    def get_model_name(self) -> str:
        """Return the IBM Watson embedding model ID."""
        return f"ibm_watson_embedding:{self.model_id}"


class OpenAIEmbeddingWrapper(BaseEmbeddingWrapper):
    """Wrapper for OpenAI embedding models."""
    
    def __init__(self, model_id: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize OpenAI embedding model.
        
        Args:
            model_id: OpenAI embedding model identifier (e.g., text-embedding-ada-002)
            params: Embedding parameters
        """
        self.model_id = model_id
        self.params = params or {}
        
        # Initialize the embeddings
        self.embeddings = OpenAIEmbeddings(
            model=model_id,
            api_key=settings.OPENAI_API_KEY
        )
    
    def get_embeddings(self):
        """Return the OpenAI embeddings object."""
        return self.embeddings
    
    def get_model_name(self) -> str:
        """Return the OpenAI embedding model ID."""
        return f"openai_embedding:{self.model_id}"

class OllamaEmbeddingWrapper(BaseEmbeddingWrapper):
    """Wrapper for Ollama embedding models."""
    
    def __init__(self, model_id: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Ollama embedding model.
        
        Args:
            model_id: Ollama embedding model identifier
            params: Embedding parameters
        """
        self.model_id = model_id
        self.params = params or {}
        
        self.embeddings = OllamaEmbeddings(
            model=model_id,
            base_url=settings.OLLAMA_BASE_URL
        )

    
    def get_embeddings(self):
        """Return the Ollama embeddings object."""
        return self.embeddings
    
    def get_model_name(self) -> str:
        """Return the Ollama embedding model ID."""
        return f"ollama_embedding:{self.model_id}"


class EmbeddingFactory:
    """
    Factory class for creating embedding model instances.
    
    This implements the Factory design pattern for embedding models.
    """
    
    @staticmethod
    def create_embeddings(
        provider: ModelProvider,
        model_id: str,
        params: Optional[Dict[str, Any]] = None
    ) -> BaseEmbeddingWrapper:
        """
        Create an embedding model instance based on the provider.
        
        Args:
            provider: The embedding provider (IBM_WATSON, OPENAI)
            model_id: The specific embedding model identifier
            params: Optional embedding parameters
            
        Returns:
            BaseEmbeddingWrapper: A wrapper instance for the specified embedding model
            
        Raises:
            ValueError: If the provider is not supported
        """
        if provider == ModelProvider.IBM_WATSON:
            return IBMWatsonEmbeddingWrapper(model_id, params)
        elif provider == ModelProvider.OPENAI:
            return OpenAIEmbeddingWrapper(model_id, params)
        elif provider == ModelProvider.OLLAMA:
            return OllamaEmbeddingWrapper(model_id, params)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}. Ollama embeddings not yet supported.")
    
    @staticmethod
    def create_embeddings_from_config() -> BaseEmbeddingWrapper:
        """
        Create an embedding model instance from configuration settings.
        
        Returns:
            BaseEmbeddingWrapper: A configured embedding model instance
        """
        config = settings.get_embedding_config()
        
        return EmbeddingFactory.create_embeddings(
            provider=ModelProvider(config["provider"]),
            model_id=config["model_id"],
            params=config["params"]
        )


# Convenience function for creating embeddings
def get_embeddings() -> BaseEmbeddingWrapper:
    """Get the configured embedding model."""
    return EmbeddingFactory.create_embeddings_from_config()
    return ModelFactory.create_model_from_config(ModelRole.RELEVANCE)
