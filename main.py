# main.py
from __future__ import annotations

import os
import hashlib
from typing import List, Dict, Any, Tuple

import gradio as gr

from document_processor.file_handler import DocumentProcessor
from retriever.builder import RetrieverBuilder
from agents.workflow import AgentWorkflow
from config import constants
from config.settings import settings
from utils.logging import logger


EXAMPLES: Dict[str, Dict[str, Any]] = {
    "Google 2024 Environmental Report": {
        "question": (
            "Retrieve the data center PUE efficiency values in Singapore 2nd facility in 2019 and 2022. "
            "Also retrieve regional average CFE in Asia pacific in 2023"
        ),
        "file_paths": ["examples/google-2024-environmental-report.pdf"],
    },
    "DeepSeek-R1 Technical Report": {
        "question": "Summarize DeepSeek-R1 model's performance evaluation on all coding tasks against OpenAI o1-mini model",
        "file_paths": ["examples/DeepSeek Technical Report.pdf"],
    },
}


class _UploadStub:
    """Wrap a filesystem path into an object with `.name` attribute (to match DocumentProcessor API)."""
    def __init__(self, path: str):
        self.name = path


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_paths(uploaded_files: List[Any]) -> List[str]:
    """Gradio Files may yield str paths (examples) or objects with .name (uploads). Normalize to paths."""
    paths: List[str] = []
    for f in uploaded_files or []:
        if isinstance(f, str):
            paths.append(f)
        else:
            p = getattr(f, "name", None)
            if p:
                paths.append(p)
    return paths


def _get_file_hashes(uploaded_files: List[Any]) -> frozenset[str]:
    hashes: set[str] = set()
    for path in _normalize_paths(uploaded_files):
        if os.path.exists(path):
            hashes.add(_sha256_file(path))
        else:
            logger.warning(f"File path does not exist: {path}")
    return frozenset(hashes)


def _hash_set(file_hashes: frozenset[str]) -> str:
    """Stable upload-set id (order-independent) for optional retriever collection suffix."""
    joined = "|".join(sorted(file_hashes))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:8]


def main() -> None:
    processor = DocumentProcessor()
    retriever_builder = RetrieverBuilder()
    workflow = AgentWorkflow()

    css = """
    .title {
        font-size: 1.5em !important;
        text-align: center !important;
        color: #FFD700;
    }
    .subtitle {
        font-size: 1em !important;
        text-align: center !important;
        color: #FFD700;
    }
    .text { text-align: center; }
    """

    js = """
    function createGradioAnimation() {
        var container = document.createElement('div');
        container.id = 'gradio-animation';
        container.style.fontSize = '2em';
        container.style.fontWeight = 'bold';
        container.style.textAlign = 'center';
        container.style.marginBottom = '20px';
        container.style.color = '#eba93f';

        var text = 'Welcome to DocChat 🐥!';
        for (var i = 0; i < text.length; i++) {
            (function(i){
                setTimeout(function(){
                    var letter = document.createElement('span');
                    letter.style.opacity = '0';
                    letter.style.transition = 'opacity 0.1s';
                    letter.innerText = text[i];
                    container.appendChild(letter);
                    setTimeout(function() { letter.style.opacity = '0.9'; }, 50);
                }, i * 250);
            })(i);
        }

        var gradioContainer = document.querySelector('.gradio-container');
        if (gradioContainer) gradioContainer.insertBefore(container, gradioContainer.firstChild);
        return 'Animation created';
    }
    """

    with gr.Blocks(theme=gr.themes.Citrus(), title="DocChat 🐥", css=css, js=js) as demo:
        gr.Markdown("## DocChat: powered by Docling 🐥 and LangGraph", elem_classes="subtitle")
        gr.Markdown("# How it works ✨:", elem_classes="title")
        gr.Markdown("📤 Upload your document(s), enter your query then press Submit 📝", elem_classes="text")
        gr.Markdown(
            "Or select an example, click **Load Example**, then press **Submit** 📝",
            elem_classes="text",
        )
        gr.Markdown(
            "⚠️ **Note:** DocChat only accepts: '.pdf', '.docx', '.txt', '.md'",
            elem_classes="text",
        )

        session_state = gr.State({"file_hashes": frozenset(), "retriever": None})

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Example 📂")
                example_dropdown = gr.Dropdown(
                    label="Select an Example 🐥",
                    choices=list(EXAMPLES.keys()),
                    value=None,
                )
                load_example_btn = gr.Button("Load Example 🛠️")

                files = gr.Files(label="📄 Upload Documents", file_types=constants.ALLOWED_TYPES)
                question = gr.Textbox(label="❓ Question", lines=3)

                submit_btn = gr.Button("Submit 🚀")

            with gr.Column():
                answer_output = gr.Textbox(label="🐥 Answer", interactive=False)
                verification_output = gr.Textbox(label="✅ Verification Report", interactive=False)

        def load_example(example_key: str) -> Tuple[List[str], str]:
            if not example_key or example_key not in EXAMPLES:
                return [], ""

            ex_data = EXAMPLES[example_key]
            q = ex_data["question"]
            file_paths = ex_data["file_paths"]

            loaded_files: List[str] = []
            for path in file_paths:
                if os.path.exists(path):
                    loaded_files.append(path)  # Gradio accepts str paths
                else:
                    logger.warning(f"Example file not found: {path}")

            return loaded_files, q

        load_example_btn.click(fn=load_example, inputs=[example_dropdown], outputs=[files, question])

        def process_question(question_text: str, uploaded_files: List[Any], state: Dict[str, Any]):
            try:
                if not (question_text or "").strip():
                    raise ValueError("❌ Question cannot be empty")
                if not uploaded_files:
                    raise ValueError("❌ No documents uploaded")

                current_hashes = _get_file_hashes(uploaded_files)

                retriever = state.get("retriever", None)
                prev_hashes = state.get("file_hashes", frozenset())

                if retriever is None or current_hashes != prev_hashes:
                    logger.info("Processing new/changed documents...")

                    paths = _normalize_paths(uploaded_files)
                    file_objs = [_UploadStub(p) for p in paths]

                    chunks = processor.process(file_objs)

                    # Stable id for this upload-set; builder can use it for collection suffix if supported.
                    set_id = _hash_set(current_hashes)

                    # If your RetrieverBuilder doesn't accept these kwargs, remove them.
                    try:
                        retriever = retriever_builder.build_hybrid_retriever(
                            chunks,
                            collection_suffix=set_id,  # recommended if builder supports it
                            persist=False,             # recommended default for upload->chat
                        )
                    except TypeError:
                        # Backward compatible with your current builder signature
                        retriever = retriever_builder.build_hybrid_retriever(chunks)

                    new_state = {"file_hashes": current_hashes, "retriever": retriever}
                else:
                    new_state = state

                result = workflow.full_pipeline(question=question_text, retriever=new_state["retriever"])
                return result["draft_answer"], result["verification_report"], new_state

            except Exception as e:
                logger.error(f"Processing error: {type(e).__name__}: {e}")
                return f"❌ Error: {str(e)}", "", state

        submit_btn.click(
            fn=process_question,
            inputs=[question, files, session_state],
            outputs=[answer_output, verification_output, session_state],
        )

    share = os.getenv("GRADIO_SHARE", "0") == "1"
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "5000"))

    demo.launch(server_name=server_name, server_port=server_port, share=share)


if __name__ == "__main__":
    print("Starting DocChat 🐥...")
    main()
