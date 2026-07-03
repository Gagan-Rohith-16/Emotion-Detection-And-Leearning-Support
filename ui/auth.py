"""Authentication dialogs connected to the SQLite database module."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import extra_streamlit_components as stx
import streamlit as st

from database import DatabaseManager


AUTH_COOKIE_NAME = "emotilearn_user_id"
AUTH_COOKIE_DAYS = 30
_cookie_manager = stx.CookieManager()


def _persist_user_cookie(user_id: int) -> None:
    """Store signed-in user id in a long-lived browser cookie."""

    _cookie_manager.set(
        AUTH_COOKIE_NAME,
        str(user_id),
        expires_at=datetime.now(timezone.utc) + timedelta(days=AUTH_COOKIE_DAYS),
    )


def clear_auth_cookie() -> None:
    """Remove the persisted login cookie from the browser."""

    _cookie_manager.delete(AUTH_COOKIE_NAME)


def initialize_auth_state(database: DatabaseManager) -> None:
    """Initialize authentication state."""

    st.session_state.setdefault("user", None)

    if st.session_state["user"] is None:
        cookie_user_id = _cookie_manager.get(AUTH_COOKIE_NAME)
        if cookie_user_id and str(cookie_user_id).isdigit():
            user = database.get_user(int(cookie_user_id))
            if user:
                st.session_state["user"] = user
            else:
                clear_auth_cookie()


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
                _persist_user_cookie(user.user_id)
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
                    _persist_user_cookie(user.user_id)
                    st.success("Your account is ready.")
                    st.rerun()

