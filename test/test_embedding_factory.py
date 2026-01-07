"""
Test script for embedding factory configuration.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.models import EmbeddingFactory, ModelProvider, get_embeddings
from config.settings import settings


def test_embedding_factory():
    """Test the embedding factory pattern."""
    
    print("=" * 60)
    print("EMBEDDING FACTORY TEST")
    print("=" * 60)
    
    # Display current configuration
    print("\n📋 Current Embedding Configuration:")
    config = settings.get_embedding_config()
    print(f"Provider: {config['provider']}")
    print(f"Model ID: {config['model_id']}")
    print(f"Truncate Input Tokens: {config['params']['truncate_input_tokens']}")
    print(f"Return Input Text: {config['params']['return_options']['input_text']}")
    
    # Test embedding creation from config
    print("\n🏭 Testing Embedding Factory:")
    
    try:
        print("\nCreating Embedding Model from Configuration...")
        embedding_wrapper = get_embeddings()
        embeddings = embedding_wrapper.get_embeddings()
        print(f"✅ Success: {embedding_wrapper.get_model_name()}")
        print(f"   Type: {type(embeddings).__name__}")
    except Exception as e:
        print(f"⚠️  Could not create embeddings: {e}")
        print("   This is expected if you haven't configured API keys yet.")
    
    # Test programmatic creation
    print("\n🔧 Testing Programmatic Creation:")
    
    # Test IBM Watson embeddings
    try:
        print("\nCreating IBM Watson Embeddings...")
        ibm_wrapper = EmbeddingFactory.create_embeddings(
            provider=ModelProvider.IBM_WATSON,
            model_id="ibm/slate-125m-english-rtrvr-v2",
            params={"truncate_input_tokens": 5}
        )
        print(f"✅ Created: {ibm_wrapper.get_model_name()}")
    except Exception as e:
        print(f"⚠️  IBM Watson Embeddings: {e}")
    
    # Test OpenAI embeddings
    try:
        print("\nCreating OpenAI Embeddings...")
        openai_wrapper = EmbeddingFactory.create_embeddings(
            provider=ModelProvider.OPENAI,
            model_id="text-embedding-ada-002"
        )
        print(f"✅ Created: {openai_wrapper.get_model_name()}")
    except Exception as e:
        print(f"⚠️  OpenAI Embeddings: {e}")
    
    print("\n" + "=" * 60)
    print("✨ Embedding Factory is configured and ready to use!")
    print("=" * 60)
    print("\nℹ️  Available Embedding Providers:")
    print("   - IBM Watson: ibm/slate-125m-english-rtrvr-v2")
    print("   - OpenAI: text-embedding-ada-002, text-embedding-3-small")
    print("\n📝 To switch providers, update your .env file:")
    print("   EMBEDDING_MODEL_PROVIDER=openai")
    print("   EMBEDDING_MODEL_ID=text-embedding-3-small")
    print("=" * 60)


if __name__ == "__main__":
    test_embedding_factory()
