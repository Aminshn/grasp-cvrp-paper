import streamlit as st


def render_exception(traceback_text: str) -> None:
    """
    Render an error message with a full traceback block.
    """
    st.error("An error occurred — full traceback below")
    st.code(traceback_text)
