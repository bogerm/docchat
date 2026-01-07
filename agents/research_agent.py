from typing import Dict, List
from langchain_classic.schema import Document
from config.models import get_research_model
# import json
from loguru import logger

class ResearchAgent:
    def __init__(self):
        """
        Initialize the research agent with the configured model.
        """
        logger.info("Initializing ResearchAgent with configured model...")
        self.model = get_research_model()
        logger.info(f"Model initialized successfully: {self.model.get_model_name()}")

    def sanitize_response(self, response_text: str) -> str:
        """
        Sanitize the LLM's response by stripping unnecessary whitespace.
        """
        return response_text.strip()

    def generate_prompt(self, question: str, context: str) -> str:
        """
        Generate a structured prompt for the LLM to generate a precise and factual answer.
        """
        prompt = f"""
        You are an AI assistant designed to provide precise and factual answers based on the given context.

        **Instructions:**
        - Answer the following question using only the provided context.
        - Be clear, concise, and factual.
        - Return as much information as you can get from the context.
        
        **Question:** {question}
        **Context:**
        {context}

        **Provide your answer below:**
        """
        return prompt

    def generate(self, question: str, documents: List[Document]) -> Dict:
        """
        Generate an initial answer using the provided documents.
        """
        logger.debug(f"ResearchAgent.generate called with question='{question}' and {len(documents)} documents.")

        # Combine the top document contents into one string
        context = "\n\n".join([doc.page_content for doc in documents])
        logger.debug(f"Combined context length: {len(context)} characters.")
        # Create a prompt for the LLM
        prompt = self.generate_prompt(question, context)
        logger.debug("Prompt created for the LLM.")

        # Call the LLM to generate the answer
        try:
            logger.debug("Sending prompt to the model...")
            llm_response = self.model.generate(prompt)
            logger.debug(f"Raw LLM response:\n{llm_response}")
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            llm_response = "I cannot answer this question based on the provided documents."

        # Sanitize the response
        draft_answer = self.sanitize_response(llm_response) if llm_response else "I cannot answer this question based on the provided documents."

        logger.debug(f"Generated answer: {draft_answer}")

        return {
            "draft_answer": draft_answer,
            "context_used": context
        }