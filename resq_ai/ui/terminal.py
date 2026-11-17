"""
resq_ai.ui.terminal
===================
Natural Language Dispatch Terminal component:
- Four quick-command chips.
- Text input + Run button.
- Monospace output log showing parse result and latency in ms.
"""

from __future__ import annotations
from typing import Dict, Any, Callable
import streamlit as st


QUICK_PROMPTS = [
    ("🚤 Dispatch Boat", "Dispatch rescue boat to Block C. Reports say Main Street is submerged."),
    ("🚑 Ambulance (15, 20)", "Send ambulance to (15, 20) with 2 elderly patients"),
    ("💥 Collapse Bridge", "Collapse bridge at (20, 34) due to flood waters"),
    ("📊 Query Status", "What is the current city status?"),
]


def render_terminal(
    engine: Any,
    last_response: str,
    on_command: Callable[[str], None],
) -> None:
    """Renders the bottom natural language command terminal card."""
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 16px; margin-bottom: 8px;">
            <span style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 0.92rem; color: var(--text-primary);">
                Natural Language Command Terminal
            </span>
            <span style="font-size: 0.76rem; color: var(--text-secondary);">
                Supports coordinates (x, y), landmarks, demographic entities, and dynamic closures
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Symmetrical Quick Command Chips
    chip_cols = st.columns(4)
    for i, (title, prompt) in enumerate(QUICK_PROMPTS):
        with chip_cols[i]:
            if st.button(title, key=f"chip_btn_{i}", help=f"Prompt: {prompt}", width="stretch"):
                on_command(prompt)

    # Text Input Form with Symmetrical Layout
    with st.form("terminal_input_form", clear_on_submit=True):
        col_text, col_btn = st.columns([5.2, 1.2])
        with col_text:
            user_text = st.text_input(
                "Command",
                placeholder="e.g. 'Dispatch boat to Riverfront Marina' or 'Block road at (15, 20)' or 'What is city status?'",
                label_visibility="collapsed",
                key="term_input_field",
            )
        with col_btn:
            run_btn = st.form_submit_button("Run ❯", width="stretch")

        if run_btn and user_text:
            on_command(user_text)

    # Monospace Terminal Output Log
    if last_response:
        st.markdown(
            f"""
            <div class="terminal-output">
                <span style="color: var(--accent); font-weight: 600;">❯ DISPATCH LOG:</span> {last_response}
            </div>
            """,
            unsafe_allow_html=True,
        )
