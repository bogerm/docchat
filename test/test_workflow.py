import pytest

from agents.workflow import AgentWorkflow


class DummyDoc:
    def __init__(self, text: str, metadata=None):
        self.page_content = text
        self.metadata = metadata or {}


class DummyRetriever:
    def __init__(self, docs):
        self._docs = docs
        self.invocations = 0

    def invoke(self, question, **kwargs):
        self.invocations += 1
        return self._docs


@pytest.fixture
def workflow():
    # If your AgentWorkflow doesn't accept max_tries, just do AgentWorkflow()
    return AgentWorkflow(max_tries=2)


def test_workflow_irrelevant_exits_early(monkeypatch, workflow):
    # Arrange: relevance says NO_MATCH
    monkeypatch.setattr(workflow.relevance_checker, "check_documents", lambda **kwargs: "NO_MATCH")

    # Track whether research/verify were called
    called = {"research": 0, "verify": 0}

    def fake_generate(question, documents):
        called["research"] += 1
        return {"draft_answer": "should not happen", "context_used": ""}

    def fake_verify(answer, documents):
        called["verify"] += 1
        return {"verification_report": "should not happen", "verification": {}, "context_used": ""}

    monkeypatch.setattr(workflow.researcher, "generate", fake_generate)
    monkeypatch.setattr(workflow.verifier, "check", fake_verify)

    retriever = DummyRetriever([DummyDoc("some doc")])

    # Act
    out = workflow.full_pipeline("q", retriever)

    # Assert: stops without research/verify
    assert called["research"] == 0
    assert called["verify"] == 0
    assert "isn't related" in out["draft_answer"].lower()
    assert out["verification_report"] == ""


def test_workflow_reruns_research_once_on_failed_verification(monkeypatch, workflow):
    # Arrange: relevance says PARTIAL (proceed)
    monkeypatch.setattr(workflow.relevance_checker, "check_documents", lambda **kwargs: "PARTIAL")

    # Research returns different answers each time so we can see it re-ran
    answers = iter(["draft v1", "draft v2"])
    def fake_generate(question, documents):
        return {"draft_answer": next(answers), "context_used": "ctx"}

    monkeypatch.setattr(workflow.researcher, "generate", fake_generate)

    # Verification fails first time, passes second time
    calls = {"verify": 0}
    def fake_verify(answer, documents):
        calls["verify"] += 1
        if calls["verify"] == 1:
            return {
                "verification": {"supported": "NO", "relevant": "YES", "unsupported_claims": ["x"], "contradictions": [], "additional_details": ""},
                "verification_report": "**Supported:** NO\n**Relevant:** YES\n",
                "context_used": "ctx",
            }
        return {
            "verification": {"supported": "YES", "relevant": "YES", "unsupported_claims": [], "contradictions": [], "additional_details": ""},
            "verification_report": "**Supported:** YES\n**Relevant:** YES\n",
            "context_used": "ctx",
        }

    monkeypatch.setattr(workflow.verifier, "check", fake_verify)

    retriever = DummyRetriever([DummyDoc("doc")])

    # Act
    out = workflow.full_pipeline("q", retriever)

    # Assert
    assert calls["verify"] == 2  # verify ran twice
    # Final answer should be from the second research run
    assert out["draft_answer"] == "draft v2"
    assert "supported:** yes" in out["verification_report"].lower()


def test_workflow_stops_after_max_tries(monkeypatch):
    wf = AgentWorkflow(max_tries=1)

    monkeypatch.setattr(wf.relevance_checker, "check_documents", lambda **kwargs: "CAN_ANSWER")

    # Research always returns same
    monkeypatch.setattr(wf.researcher, "generate", lambda q, d: {"draft_answer": "draft", "context_used": "ctx"})

    # Verification always fails
    calls = {"verify": 0}
    def always_fail(answer, documents):
        calls["verify"] += 1
        return {
            "verification": {"supported": "NO", "relevant": "NO", "unsupported_claims": ["x"], "contradictions": [], "additional_details": ""},
            "verification_report": "**Supported:** NO\n**Relevant:** NO\n",
            "context_used": "ctx",
        }

    monkeypatch.setattr(wf.verifier, "check", always_fail)

    retriever = DummyRetriever([DummyDoc("doc")])

    out = wf.full_pipeline("q", retriever)

    # With max_tries=1, it should not loop forever; it ends with the last draft + report.
    assert out["draft_answer"] == "draft"
    assert "supported:** no" in out["verification_report"].lower()
