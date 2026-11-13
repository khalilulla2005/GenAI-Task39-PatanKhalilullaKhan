from __future__ import annotations

import logging
from uuid import uuid4

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from src.assistant import check_ollama, run_assistant
from src.config import Settings
from src.prompts import TASK_EXAMPLES, TaskType

LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title="CodeLlama Code Assistant",
    page_icon="</>",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1180px; padding-top: 2rem; }
    .hero {
        padding: 1.6rem 1.8rem;
        border: 1px solid #dbeafe;
        border-radius: 18px;
        background: linear-gradient(120deg, #eff6ff 0%, #f8fafc 72%);
        margin-bottom: 1.4rem;
    }
    .hero h1 { margin: 0 0 .45rem 0; color: #0f172a; }
    .hero p { margin: 0; color: #475569; font-size: 1.05rem; }
    .section-label {
        color: #475569;
        font-size: .82rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "assistant_history" not in st.session_state:
    st.session_state["assistant_history"] = []
if "assistant_session_id" not in st.session_state:
    st.session_state["assistant_session_id"] = uuid4().hex


def _settings() -> Settings:
    try:
        secrets = st.secrets.to_dict()
    except (StreamlitSecretNotFoundError, FileNotFoundError):
        secrets = {}
    return Settings.from_env(secrets=secrets)


st.markdown(
    """
    <div class="hero">
      <div class="section-label">Local-first developer tool</div>
      <h1>CodeLlama Code Assistant</h1>
      <p>Generate, understand, debug, and improve Python with your Ollama model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    current_settings = _settings()
except ValueError as exc:
    st.error(f"Configuration error: {exc}")
    st.stop()

with st.sidebar:
    st.markdown("### Runtime")
    st.code(current_settings.model, language="text")
    st.caption(f"Ollama endpoint: `{current_settings.base_url}`")
    st.caption("Prompts are sent only to this configured Ollama endpoint.")

    if st.button("Check Ollama connection", use_container_width=True):
        with st.spinner("Checking Ollama and the selected model..."):
            try:
                status = check_ollama(current_settings)
            except RuntimeError as exc:
                st.error(str(exc))
            else:
                if status.model_available:
                    st.success(f"Connected. `{current_settings.model}` is available.")
                elif status.available:
                    st.warning(
                        f"Ollama is running, but `{current_settings.model}` is not "
                        "downloaded. Run `ollama pull "
                        f"{current_settings.model}`."
                    )

    st.divider()
    st.markdown("### Prompt examples")
    example_task = st.selectbox(
        "Example task",
        options=list(TaskType),
        format_func=lambda item: item.value,
        key="example_task",
    )
    example = st.selectbox(
        "Example prompt",
        options=TASK_EXAMPLES[example_task],
        key="example_prompt",
    )
    if st.button("Use this example", use_container_width=True):
        st.session_state["request_text"] = example
        st.session_state["selected_task"] = example_task.value
        st.rerun()

    st.divider()
    if st.button("Clear history", use_container_width=True):
        st.session_state["assistant_history"] = []
        st.session_state["assistant_session_id"] = uuid4().hex
        st.rerun()

task = st.selectbox(
    "What would you like help with?",
    options=list(TaskType),
    format_func=lambda item: item.value,
    key="selected_task",
)

request_text = st.text_area(
    "Describe the task or paste your Python code",
    key="request_text",
    height=210,
    max_chars=20000,
    placeholder=(
        "Example: Write a Python function that checks whether an integer is prime."
    ),
    help="Requests are limited to 20,000 characters. Submitted code is not executed.",
)

submit_col, privacy_col = st.columns([1, 3], vertical_alignment="center")
with submit_col:
    submitted = st.button(
        "Ask CodeLlama",
        type="primary",
        use_container_width=True,
        disabled=not request_text.strip(),
    )
with privacy_col:
    st.caption(
        "Review generated code and test it before use. "
        "The assistant does not execute your code."
    )

if submitted:
    try:
        status = check_ollama(current_settings)
        if not status.model_available:
            if status.available:
                st.error(
                    f"Model `{current_settings.model}` is not installed. Run "
                    f"`ollama pull {current_settings.model}` and try again."
                )
            else:
                st.error(
                    "Ollama is not available. Start it with `ollama serve` or "
                    "configure a reachable Ollama endpoint."
                )
        else:
            with st.spinner(f"{task.value} with {current_settings.model}..."):
                answer = run_assistant(task, request_text, current_settings)
            st.markdown("### Response")
            st.markdown(answer)
            st.session_state["assistant_history"].append(
                {
                    "task": task.value,
                    "request": request_text,
                    "answer": answer,
                }
            )
            st.session_state["assistant_history"] = st.session_state[
                "assistant_history"
            ][-5:]
    except (ValueError, RuntimeError) as exc:
        st.error(f"Request could not be completed: {exc}")
    except Exception as exc:
        LOGGER.exception("Unexpected CodeLlama request failure")
        st.error(
            "The Ollama request failed "
            f"({type(exc).__name__}). Check the model service and try again."
        )

history = st.session_state["assistant_history"]
if history:
    st.divider()
    st.markdown("### Recent requests")
    for index, turn in enumerate(reversed(history), start=1):
        with st.expander(f"{turn['task']} · Request {index}"):
            st.markdown("**Request**")
            st.code(turn["request"], language="text")
            st.markdown("**Response**")
            st.markdown(turn["answer"])
