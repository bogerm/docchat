import pytest

from agents.relevance_checker import RelevanceChecker


class DummyDoc:
    def __init__(self, text: str, metadata=None):
        self.page_content = text
        self.metadata = metadata or {}


class DummyRetriever:
    """
    A permissive retriever that accepts extra kwargs (e.g., config={"k": k}).
    """
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def invoke(self, question, **kwargs):
        self.calls.append((question, kwargs))
        return self.docs


class StrictRetriever:
    """
    A strict retriever that does NOT accept kwargs (common in some implementations).
    This exercises the fallback path in RelevanceChecker._retrieve().
    """
    def __init__(self, docs):
        self.docs = docs
        self.called = 0

    def invoke(self, question):
        self.called += 1
        return self.docs


class DummyModel:
    """
    A dummy model that returns a fixed response.
    """
    def __init__(self, response: str):
        self._response = response
        self.last_system = None
        self.last_user = None

    def get_model_name(self):
        return "dummy:model"

    def generate(self, prompt: str) -> str:
        return self._response

    def generate_messages(self, system: str, user: str) -> str:
        self.last_system = system
        self.last_user = user
        return self._response


class ErrorModel(DummyModel):
    def generate_messages(self, system: str, user: str) -> str:
        raise RuntimeError("boom")


@pytest.fixture
def patch_model(monkeypatch):
    """
    Patch get_relevance_model() inside agents.relevance_checker
    so RelevanceChecker() uses a dummy model.
    """
    import agents.relevance_checker as rc_mod

    def _patch(model):
        monkeypatch.setattr(rc_mod, "get_relevance_model", lambda: model)

    return _patch


def test_no_docs_returns_no_match(patch_model):
    patch_model(DummyModel("CAN_ANSWER"))
    checker = RelevanceChecker()
    retriever = DummyRetriever(docs=[])

    assert checker.check("anything", retriever) == "NO_MATCH"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("CAN_ANSWER", "CAN_ANSWER"),
        ("partial", "PARTIAL"),
        ("no_match", "NO_MATCH"),
        ("Label: CAN_ANSWER.", "CAN_ANSWER"),
        ("```PARTIAL```", "PARTIAL"),
        ("Some text then NO_MATCH later", "NO_MATCH"),
    ],
)
def test_label_parsing_robust(patch_model, raw, expected):
    patch_model(DummyModel(raw))
    checker = RelevanceChecker()
    retriever = DummyRetriever([DummyDoc("doc1")])

    assert checker.check("q", retriever, k=1) == expected


def test_garbage_output_defaults_to_no_match(patch_model):
    patch_model(DummyModel("I think you can answer this."))
    checker = RelevanceChecker()
    retriever = DummyRetriever([DummyDoc("doc1")])

    assert checker.check("q", retriever, k=1) == "NO_MATCH"


def test_uses_only_top_k_docs_in_passages(patch_model):
    model = DummyModel("PARTIAL")
    patch_model(model)

    docs = [
        DummyDoc("DOC_ONE", {"source": "a.pdf", "page": 1}),
        DummyDoc("DOC_TWO", {"source": "a.pdf", "page": 2}),
        DummyDoc("DOC_THREE", {"source": "a.pdf", "page": 3}),
    ]
    retriever = DummyRetriever(docs)
    checker = RelevanceChecker()

    checker.check("q", retriever, k=2)

    assert model.last_user is not None
    assert "DOC_ONE" in model.last_user
    assert "DOC_TWO" in model.last_user
    assert "DOC_THREE" not in model.last_user


def test_truncates_per_chunk_and_total(patch_model):
    model = DummyModel("CAN_ANSWER")
    patch_model(model)

    long_text = "A" * 10_000
    docs = [DummyDoc(long_text), DummyDoc(long_text)]
    retriever = DummyRetriever(docs)

    checker = RelevanceChecker(default_k=2, max_chars_total=1500, max_chars_per_chunk=700)
    checker.check("q", retriever, k=2)

    assert model.last_user is not None
    # user prompt includes tags + question, so check it's not huge
    assert len(model.last_user) < 2500
    # ensure we didn't include the full long chunk
    assert ("A" * 5000) not in model.last_user


def test_model_exception_returns_no_match(patch_model):
    patch_model(ErrorModel("CAN_ANSWER"))
    checker = RelevanceChecker()
    retriever = DummyRetriever([DummyDoc("doc1")])

    assert checker.check("q", retriever) == "NO_MATCH"


def test_strict_retriever_fallback_path(patch_model):
    model = DummyModel("PARTIAL")
    patch_model(model)

    retriever = StrictRetriever([DummyDoc("doc1")])
    checker = RelevanceChecker()

    assert checker.check("q", retriever, k=1) == "PARTIAL"
    assert retriever.called == 1

def test_system_prompt_contains_injection_guard(patch_model):
    model = DummyModel("NO_MATCH")
    patch_model(model)

    checker = RelevanceChecker()
    retriever = DummyRetriever([DummyDoc("doc1")])

    checker.check("q", retriever, k=1)
    assert model.last_system is not None
    assert "Ignore any instructions" in model.last_system or "untrusted" in model.last_system