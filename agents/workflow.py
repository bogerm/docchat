from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any

from .research_agent import ResearchAgent
from .verification_agent import VerificationAgent
from .relevance_checker import RelevanceChecker

from loguru import logger

# Pick ONE Document import across the project
from langchain_core.documents import Document  # recommended

# And your retriever type (adjust if you truly use langchain_classic)
from langchain.retrievers import EnsembleRetriever


class AgentState(TypedDict):
    question: str
    documents: List[Document]
    draft_answer: str
    verification_report: str
    verification: Dict[str, Any]
    is_relevant: bool
    retriever: EnsembleRetriever
    tries: int


class AgentWorkflow:
    def __init__(self, *, max_tries: int = 2):
        self.researcher = ResearchAgent()
        self.verifier = VerificationAgent()
        self.relevance_checker = RelevanceChecker()
        self.max_tries = max_tries
        self.compiled_workflow = self.build_workflow()

    def build_workflow(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("check_relevance", self._check_relevance_step)
        workflow.add_node("research", self._research_step)
        workflow.add_node("verify", self._verification_step)

        workflow.set_entry_point("check_relevance")
        workflow.add_conditional_edges(
            "check_relevance",
            self._decide_after_relevance_check,
            {"relevant": "research", "irrelevant": END},
        )

        workflow.add_edge("research", "verify")

        workflow.add_conditional_edges(
            "verify",
            self._decide_next_step,
            {"re_research": "research", "end": END},
        )

        return workflow.compile()

    def full_pipeline(self, question: str, retriever: EnsembleRetriever):
        logger.debug(f"Starting full_pipeline with question={question!r}")
        documents = retriever.invoke(question)
        logger.info(f"Retrieved {len(documents)} relevant documents")

        initial_state: AgentState = {
            "question": question,
            "documents": documents,
            "draft_answer": "",
            "verification_report": "",
            "verification": {},
            "is_relevant": False,
            "retriever": retriever,
            "tries": 0,
        }

        final_state = self.compiled_workflow.invoke(initial_state)
        return {
            "draft_answer": final_state["draft_answer"],
            "verification_report": final_state["verification_report"],
        }

    def _check_relevance_step(self, state: AgentState) -> Dict[str, Any]:
        classification = self.relevance_checker.check(
            question=state["question"],
            retriever=state["retriever"],
            k=20,
        )

        if classification in {"CAN_ANSWER", "PARTIAL"}:
            return {"is_relevant": True}

        return {
            "is_relevant": False,
            "draft_answer": "This question isn't related (or there's no data) for your query. "
                            "Please ask another question relevant to the uploaded document(s).",
            "verification_report": "",
            "verification": {},
        }

    def _decide_after_relevance_check(self, state: AgentState) -> str:
        decision = "relevant" if state["is_relevant"] else "irrelevant"
        logger.debug(f"_decide_after_relevance_check -> {decision}")
        return decision

    def _research_step(self, state: AgentState) -> Dict[str, Any]:
        logger.debug(f"Entered _research_step with question={state['question']!r}")
        result = self.researcher.generate(state["question"], state["documents"])
        return {"draft_answer": result["draft_answer"]}

    def _verification_step(self, state: AgentState) -> Dict[str, Any]:
        logger.debug("Entered _verification_step. Verifying the draft answer...")
        result = self.verifier.check(state["draft_answer"], state["documents"])
        # Expect result to include both formatted and structured forms
        return {
            "verification_report": result.get("verification_report", ""),
            "verification": result.get("verification", {}) or {},
        }

    def _decide_next_step(self, state: AgentState) -> str:
        ver = state.get("verification") or {}
        supported = ver.get("supported", "NO")
        relevant = ver.get("relevant", "NO")

        if supported == "NO" or relevant == "NO":
            if state["tries"] >= self.max_tries:
                logger.info("Max re-research tries reached, ending workflow.")
                return "end"

            state["tries"] += 1
            logger.info(f"Verification indicates re-research needed (try {state['tries']}/{self.max_tries}).")
            return "re_research"

        logger.info("Verification successful, ending workflow.")
        return "end"
