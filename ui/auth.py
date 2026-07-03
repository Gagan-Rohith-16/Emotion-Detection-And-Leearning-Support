"""Authentication dialogs connected to the SQLite database module."""

from __future__ import annotations

import streamlit as st

from database import DatabaseManager


AUTH_QUERY_PARAM = "auth"


def _get_auth_token() -> str:
    """Read the private auth token from the current URL."""

    value = st.query_params.get(AUTH_QUERY_PARAM, "")
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value).strip()


def _set_auth_token(token: str) -> None:
    """Persist the private auth token in the URL so refresh keeps the session."""

    st.query_params[AUTH_QUERY_PARAM] = token


def clear_auth_token(database: DatabaseManager | None = None) -> None:
    """Remove the private auth token from the URL and optionally revoke it server-side."""

    token = _get_auth_token()
    if database is not None and token:
        database.revoke_auth_token(token)
    if AUTH_QUERY_PARAM in st.query_params:
        del st.query_params[AUTH_QUERY_PARAM]

def initialize_auth_state(database: DatabaseManager) -> None:
    """Initialize authentication state."""

    st.session_state.setdefault("user", None)

    if st.session_state["user"] is None:
        token = _get_auth_token()
        if token:
            user = database.get_user_by_auth_token(token)
            if user:
                st.session_state["user"] = user
            else:
                clear_auth_token()


def render_auth_panel(database: DatabaseManager) -> None:
    """Render accessible sign-in and registration tabs."""

    sign_in, register = st.tabs(["Sign in", "Create account"])
    with sign_in:
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="student@example.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Continue", use_container_width=True)
        if submitted:
            user = database.authenticate_user(email, password)
            if user is None:
                st.error("That email and password combination was not recognized.")
            else:
                st.session_state.user = user
                _set_auth_token(database.issue_auth_token(user.user_id))
                st.success(f"Welcome back, {user.name}!")
                st.rerun()

    with register:
        with st.form("registration_form"):
            name = st.text_input("Full name", placeholder="Your name")
            new_email = st.text_input("Email address", placeholder="you@example.com")
            new_password = st.text_input(
                "Create password", type="password", help="Use at least 8 characters."
            )
            confirm_password = st.text_input("Confirm password", type="password")
            accepted = st.checkbox("I agree to responsible use of the platform")
            registered = st.form_submit_button("Create my account", use_container_width=True)
        if registered:
            if new_password != confirm_password:
                st.error("The passwords do not match.")
            elif not accepted:
                st.error("Please accept the responsible-use agreement.")
            else:
                try:
                    user = database.register_user(name, new_email, new_password)
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.session_state.user = user
                    _set_auth_token(database.issue_auth_token(user.user_id))
                    st.success("Your account is ready.")
                    st.rerun()

