from config.models import get_relevance_model
# import re
from loguru import logger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.retrievers import BaseRetriever

# logger = logging.getLogger(__name__)

# Valid classification labels
VALID_LABELS = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}

class RelevanceChecker:
    def __init__(self):
        # Initialize the configured model
        logger.info("Initializing RelevanceChecker with configured model...")
        self.model = get_relevance_model()
        logger.info(f"Model initialized: {self.model.get_model_name()}")

    def check(self, question: str, retriever: "BaseRetriever", k: int = 3) -> str:
        """
        1. Retrieve the top-k document chunks from the global retriever.
        2. Combine them into a single text string.
        3. Pass that text + question to the LLM for classification.
        Returns: "CAN_ANSWER", "PARTIAL", or "NO_MATCH".
        """
        logger.debug(f"RelevanceChecker.check called with question='{question}' and k={k}")
        # Retrieve doc chunks from the ensemble retriever
        top_docs = retriever.invoke(question)
        if not top_docs:
            logger.debug("No documents returned from retriever.invoke(). Classifying as NO_MATCH.")
            return "NO_MATCH"
        # Combine the top k chunk texts into one string
        document_content = "\n\n".join(doc.page_content for doc in top_docs[:k])
        # Create a prompt for the LLM to classify relevance
        prompt = f"""
        You are an AI relevance checker between a user's question and provided document content.
        **Instructions:**
        - Classify how well the document content addresses the user's question.
        - Respond with only one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH.
        - Do not include any additional text or explanation.
        **Labels:**
        1) "CAN_ANSWER": The passages contain enough explicit information to fully answer the question.
        2) "PARTIAL": The passages mention or discuss the question's topic but do not provide all the details needed for a complete answer.
        3) "NO_MATCH": The passages do not discuss or mention the question's topic at all.
        **Important:** If the passages mention or reference the topic or timeframe of the question in any way, even if incomplete, respond with "PARTIAL" instead of "NO_MATCH".
        **Question:** {question}
        **Passages:** {document_content}
        **Respond ONLY with one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH**
        """
        # Call the LLM
        try:
            logger.debug(f"Sending prompt ({len(prompt)} bytes) to the model for relevance classification")
            # Log truncated prompt to avoid excessive output
            prompt_preview = prompt[:2000] + "..." if len(prompt) > 2000 else prompt
            logger.debug(f"Prompt preview: {prompt_preview}")
            llm_response = self.model.generate(prompt).strip().upper()
            logger.debug(f"LLM response: {llm_response}")
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            return "NO_MATCH"
        
        logger.info(f"Checker response: {llm_response}")
        
        # Validate the response
        if llm_response not in VALID_LABELS:
            logger.debug("LLM did not respond with a valid label. Forcing 'NO_MATCH'.")
            classification = "NO_MATCH"
        else:
            logger.debug(f"Classification recognized as '{llm_response}'.")
            classification = llm_response
        return classification