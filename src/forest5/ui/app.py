"""Streamlit application for Forest 5 UI."""

from __future__ import annotations

import streamlit as st


def _page_home() -> None:
    """Display the home page."""
    st.title("Forest 5 UI")
    st.write("Witaj w aplikacji Forest 5!")


def _page_about() -> None:
    """Display information about the application."""
    st.title("O aplikacji")
    st.write("To demo interfejsu z wykorzystaniem Streamlit.")


PAGE_MAP = {
    "Strona główna": _page_home,
    "O aplikacji": _page_about,
}


def main() -> None:
    """Run the Streamlit application."""
    st.sidebar.title("Nawigacja")
    selection = st.sidebar.radio("Przejdź do", list(PAGE_MAP))
    page = PAGE_MAP[selection]
    page()


if __name__ == "__main__":
    main()
