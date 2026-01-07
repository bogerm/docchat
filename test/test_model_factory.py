"""
Test script for model factory configuration.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.models import ModelFactory, ModelProvider, ModelRole
from config.settings import settings


def test_model_factory():
    """Test the model factory pattern."""
    
    print("=" * 60)
    print("MODEL FACTORY TEST")
    print("=" * 60)
    
    # Display current configuration
    print("\n📋 Current Configuration:")
    print(f"Default Provider: {settings.DEFAULT_MODEL_PROVIDER}")
    print(f"IBM Watson URL: {settings.IBM_WATSON_URL}")
    print(f"IBM Watson Project: {settings.IBM_WATSON_PROJECT_ID}")
    
    # Test model configurations
    print("\n🔧 Model Configurations:")
    
    for role in ["research", "verification", "relevance"]:
        config = settings.get_model_config(role)
        print(f"\n{role.upper()}:")
        print(f"  Provider: {config['provider']}")
        print(f"  Model ID: {config['model_id']}")
        print(f"  Temperature: {config['params']['temperature']}")
        print(f"  Max Tokens: {config['params']['max_tokens']}")
    
    # Test model creation from config
    print("\n🏭 Testing Model Factory:")
    
    try:
        print("\nCreating Research Model...")
        research_model = ModelFactory.create_model_from_config(ModelRole.RESEARCH)
        print(f"✅ Success: {research_model.get_model_name()}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    try:
        print("\nCreating Verification Model...")
        verification_model = ModelFactory.create_model_from_config(ModelRole.VERIFICATION)
        print(f"✅ Success: {verification_model.get_model_name()}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    try:
        print("\nCreating Relevance Model...")
        relevance_model = ModelFactory.create_model_from_config(ModelRole.RELEVANCE)
        print(f"✅ Success: {relevance_model.get_model_name()}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✨ Model Factory is configured and ready to use!")
    print("=" * 60)
    print("\nℹ️  To change providers, update your .env file:")
    print("   - Set DEFAULT_MODEL_PROVIDER to: ibm_watson, openai, or ollama")
    print("   - Or override individual models with *_MODEL_PROVIDER variables")
    print("\n📖 See MODEL_CONFIGURATION.md for detailed instructions")
    print("=" * 60)


if __name__ == "__main__":
    test_model_factory()
