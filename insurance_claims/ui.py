import os

import streamlit as st

from insurance_claims.conversation import chat
from insurance_claims.state import SessionState


st.set_page_config(
    page_title="Claims Workflow Demo",
    page_icon="💬",
    layout="wide",
)


def reset_session():
    st.session_state.claim_session = SessionState()
    st.session_state.display_messages = []


if "claim_session" not in st.session_state:
    reset_session()

st.title("Insurance Claims Support")
st.caption(
    "Demo using sample customer records. "
    "Email delivery and human transfers are simulated."
)

with st.sidebar:
    st.header("Demo controls")

    if st.button("Start new conversation", use_container_width=True):
        reset_session()
        st.rerun()

    st.caption(
        "Starting over clears this session's conversation "
        "and simulated email outbox."
    )

    status_panel = st.container()

state = st.session_state.claim_session

key_configured = bool(os.getenv("OPENAI_API_KEY"))
model_configured = bool(os.getenv("OPENAI_MODEL"))

if not key_configured or not model_configured:
    st.error(
        "Set OPENAI_API_KEY and OPENAI_MODEL in the terminal "
        "before starting the app."
    )
    st.stop()

if not st.session_state.display_messages:
    with st.chat_message("assistant"):
        st.write(
            "Hello! I can help with your insurance claim. "
            "What would you like help with today?"
        )

for message in st.session_state.display_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

message = st.chat_input(
    "Type your message",
    max_chars=8000,
    disabled=state.human_requested or state.post_process_complete,
)

if message:
    st.session_state.display_messages.append(
        {"role": "user", "content": message}
    )

    with st.chat_message("user"):
        st.write(message)

    with st.chat_message("assistant"):
        with st.spinner("Reviewing your message..."):
            response = chat(state, message)
        st.write(response)

    st.session_state.display_messages.append(
        {"role": "assistant", "content": response}
    )

    st.rerun()

with status_panel:
    st.header("SOP status")
    st.write("Phase:", state.phase.value)
    st.write("Identity verified:", state.is_verified)

    if state.is_verified:
        st.write("Selected claim:", state.selected_case_id or "None")

    st.write("Human handoff requested:", state.human_requested)
    st.write("Conversation complete:", state.post_process_complete)

    with st.expander("Remembered claim hints"):
        st.json(state.case_hints)

    st.caption(f"Model: {os.environ['OPENAI_MODEL']}")

if state.human_requested:
    st.info(
        "Automation is paused for a human representative. "
        "No real transfer occurs in this demo."
    )
elif state.post_process_complete:
    st.success(
        "This conversation is complete. "
        "Use 'Start new conversation' to test another scenario."
    )

if state.is_verified and state.outbox:
    with st.expander("Simulated email outbox", expanded=True):
        for email in state.outbox:
            st.write("To:", email["to"])
            st.write("Subject:", email["subject"])
            st.text(email["body"])
            st.caption("Simulated delivery — no real email was sent.")