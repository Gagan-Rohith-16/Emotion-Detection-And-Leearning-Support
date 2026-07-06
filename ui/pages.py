"""Page renderers for the initial Streamlit application shell."""

from __future__ import annotations

import streamlit as st
import pandas as pd
from database import DatabaseManager
from prediction.emotion_pipeline import EmotionPredictionPipeline
from gemini.guidance import GeminiGuidanceService

from .auth import render_auth_panel
from .components import (
    render_card,
    render_metric,
    render_module_status,
    render_page_header,
)


def render_home(database: DatabaseManager) -> None:
    """Render the public landing page and authentication entry point."""

    st.markdown('<div class="eyebrow">Emotion-aware learning</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-title">Learn better when your<br>'
        '<span class="gradient-text">feelings are understood.</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-copy">Describe what is getting in the way. EmotiLearn combines '
        'deep learning, emotion fusion, and personalized AI guidance to help you find a '
        'clearer next step.</p>',
        unsafe_allow_html=True,
    )
    st.write("")
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        feature_columns = st.columns(3)
        features = (
            ("🧠", "Multi-model insight", "BiLSTM, BERT, and language rules work together."),
            ("✨", "Personal guidance", "Gemini turns emotion signals into practical support."),
            ("🔒", "Private progress", "Secure accounts keep each learner's history separate."),
        )
        for column, feature in zip(feature_columns, features, strict=True):
            with column:
                render_card(*feature)
    with right:
        if st.session_state.get("user"):
            render_card("👋", "You are ready", "Open Prediction to describe your learning challenge.")
            if st.button("Start a prediction", use_container_width=True):
                st.session_state.current_page = "Prediction"
                st.rerun()
        else:
            render_auth_panel(database)


def render_dashboard(database: DatabaseManager) -> None:
    """Render analytics dashboard."""

    render_page_header(
        "Analytics Dashboard",
        "Learning Emotion Insights",
        "Track your emotional learning patterns over time.",
    )

    user = st.session_state.get("user")

    if user is None:
        st.warning("Please login first.")
        return

    history = database.get_emotion_history(user.user_id)

    if not history:
        st.info("No prediction history found.")
        return


    rows = []

    for record in history:
        rows.append(
            {
                "Emotion": record.predicted_emotion,
                "Confidence": record.confidence_score,
                "Field": record.field,
                "Date": str(record.timestamp),
            }
        )

    df = pd.DataFrame(rows)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Predictions", len(df))

    with col2:
        st.metric(
            "Average Confidence",
            f"{df['Confidence'].mean():.2%}"
        )

    st.divider()

    st.subheader("Emotion Distribution")

    emotion_counts = df["Emotion"].value_counts()

    st.bar_chart(emotion_counts)

    st.divider()

    st.subheader("Average Confidence by Emotion")

    confidence = (
        df.groupby("Emotion")["Confidence"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(confidence)

    st.divider()

    st.subheader("Prediction History")

    st.dataframe(
        df,
        use_container_width=True,
    )

def render_prediction(database: DatabaseManager) -> None:
    """Render the prediction experience."""

    render_page_header(
        "Emotion Detection",
        "What is making learning difficult?",
        "Write naturally. Our AI will detect your emotion using BiLSTM, BERT, Keyword Rules, and Emotion Fusion.",
    )

    with st.form("prediction_form"):

        field = st.selectbox(
            "Learning Field",
            [
                "General",
                "Computer Science",
                "Mathematics",
                "Science",
                "Languages",
                "Other",
            ],
        )

        challenge = st.text_area(
            "Describe your learning challenge",
            height=180,
            placeholder="Example: I have reread recursion several times but still cannot understand it.",
        )

        submitted = st.form_submit_button(
            "Analyze My Learning State",
            use_container_width=True,
        )

    if submitted:

        if not challenge.strip():
            st.warning("Please enter your learning challenge.")
            return

        with st.spinner("Analyzing..."):

            pipeline = EmotionPredictionPipeline()
            result = pipeline.predict(challenge)

            guidance_service = GeminiGuidanceService()

            guidance = guidance_service.generate_guidance(
                student_text=challenge,
                field=field,
                primary_emotion=result["primary_emotion"],
                secondary_emotion=result["secondary_emotion"],
                confidence=result["confidence"],
                emotion_scores=result["scores"],
            )

        st.success("Analysis Completed!")

        col1, col2 = st.columns(2)

        with col1:
            emotion_icons = {
                "Confused": "😕",
                "Curious": "🤔",
                "Confident": "😎",
                "Frustrated": "😣",
                "Bored": "😴",
            }

            st.metric(
                "Primary Emotion",
                f"{emotion_icons.get(result['primary_emotion'], '😊')} {result['primary_emotion']}"
            )
            confidence = result["confidence"]

            if confidence >= 0.80:
                st.success(f"Confidence: {confidence:.2%}")
            elif confidence >= 0.60:
                st.warning(f"Confidence: {confidence:.2%}")
            else:
                st.error(f"Confidence: {confidence:.2%}")
            st.progress(result["confidence"])

        with col2:
            st.metric("Secondary Emotion", result["secondary_emotion"] or "None")
            st.metric("Mixed Emotion", "Yes" if result["mixed"] else "No")

        st.subheader("Emotion Scores")

        scores = pd.DataFrame(
            {
                "Emotion": list(result["scores"].keys()),
                "Score": list(result["scores"].values()),
            }
        ).set_index("Emotion")

        st.bar_chart(scores)

        st.subheader("Model Agreement")
        st.write(f"Agreement Score: **{result['agreement']:.2f}**")

        if result["rule_evidence"]:
            st.subheader("Keyword Evidence")
            for evidence in result["rule_evidence"]:
                st.write(f"• **{evidence.phrase}** → {evidence.emotion}")

        st.divider()

        st.markdown("---")

        st.markdown(
        """
        # 🤖 Personalized AI Learning Coach
        """
        )
        st.markdown(guidance.as_markdown())
        if st.session_state.get("user"):
            prediction_key = (
                challenge.strip(),
                result["primary_emotion"],
                result["confidence"],
            )

            if st.session_state.get("last_prediction") != prediction_key:

                database.add_emotion_record(
                    user_id=st.session_state["user"].user_id,
                    input_text=challenge,
                    field=field,
                    predicted_emotion=result["primary_emotion"],
                    secondary_emotion=result["secondary_emotion"],
                    confidence_score=result["confidence"],
                    model_used="BiLSTM + BERT + Keyword Rules + Emotion Fusion",
                    ai_response=guidance.as_markdown(),
                    emotion_scores=result["scores"],
                )

                st.session_state["last_prediction"] = prediction_key
                st.toast("✅ Prediction saved successfully!")
        if guidance.is_fallback:
            st.info(
                "Showing offline guidance. Configure a Gemini API key to receive AI-generated personalized guidance."
            )

def render_analytics(database: DatabaseManager) -> None:
    """Render detailed analytics."""

    render_page_header(
        "Analytics",
        "Learning Insights",
        "Understand your learning emotions over time.",
    )

    user = st.session_state.get("user")

    if user is None:
        st.warning("Please login first.")
        return

    history = database.get_emotion_history(user.user_id)

    if not history:
        st.info("No analytics available yet.")
        return


    rows = []

    for record in history:
        rows.append(
            {
                "Emotion": record.predicted_emotion,
                "Confidence": record.confidence_score,
                "Field": record.field,
                "Date": record.timestamp[:10],
            }
        )

    df = pd.DataFrame(rows)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Predictions", len(df))

    with col2:
        st.metric(
            "Average Confidence",
            f"{df['Confidence'].mean():.2%}"
        )

    with col3:
        st.metric(
            "Most Frequent Emotion",
            df["Emotion"].mode()[0]
        )

    st.divider()

    st.subheader("Emotion Distribution")
    st.bar_chart(df["Emotion"].value_counts())

    st.divider()

    st.subheader("Predictions by Learning Field")
    st.bar_chart(df["Field"].value_counts())

    st.divider()

    st.subheader("Average Confidence by Emotion")
    st.bar_chart(
        df.groupby("Emotion")["Confidence"].mean()
    )

    st.divider()

    st.subheader("Daily Prediction Trend")

    daily = df.groupby("Date").size()

    st.line_chart(daily)

    st.divider()

    st.subheader("Raw Analytics Data")
    st.dataframe(df, use_container_width=True)
    st.markdown("---")
    st.caption(
        "EmotiLearn AI • Google Cloud Generative AI SmartBridge Internship Project • Version 1.0"
    )

def render_history(database: DatabaseManager) -> None:
    """Render the logged-in user's prediction history."""

    render_page_header(
        "History",
        "Previous Predictions",
        "View all your previous emotion detection results.",
    )

    user = st.session_state.get("user")

    if user is None:
        st.warning("Please login first.")
        return
    history = database.get_emotion_history(user.user_id)

    if not history:
        st.info("No prediction history found.")
        return
    rows = []
    for record in history:
        rows.append(
            {
                "Date": record.timestamp,
                "Field": record.field,
                "Input": record.input_text,
                "Primary Emotion": record.predicted_emotion,
                "Secondary Emotion": record.secondary_emotion,
                "Confidence": record.confidence_score,
            }
        )
    df = pd.DataFrame(rows)
    st.download_button(
        label="📥 Download History as CSV",
        data=df.to_csv(index=False),
        file_name="emotion_history.csv",
        mime="text/csv",
    )

    for record in history:
        with st.expander(
            f"🕒 {record.timestamp} | 😊 {record.predicted_emotion} | 🎯 {record.confidence_score:.2%}"
        ):

            col1, col2 = st.columns(2)

            with col1:
                st.write("### 📚 Learning Field")
                st.info(record.field)

            with col2:
                st.write("### 🎯 Confidence")
                st.success(f"{record.confidence_score:.2%}")

            st.write("### 📝 Student Input")
            st.write(record.input_text)

            st.write("### 😊 Primary Emotion")
            st.success(record.predicted_emotion)

            st.write("### 🔄 Secondary Emotion")
            st.info(record.secondary_emotion or "None")

            st.divider()

            st.write("### 🤖 AI Learning Guidance")
            st.markdown(record.ai_response)
            if st.button(
                "🗑 Delete This Prediction",
                key=f"delete_{record.record_id}",
                use_container_width=True,
            ):
                database.delete_emotion_record(
                    record.record_id,
                    user.user_id,
                )
                st.success("Prediction deleted successfully.")
                st.rerun()
    st.markdown("---")
    st.caption(
        "EmotiLearn AI • Google Cloud Generative AI SmartBridge Internship Project • Version 1.0"
    )

def render_settings(database: DatabaseManager) -> None:
    """Render application settings."""

    render_page_header(
        "Settings",
        "Customize Your Experience",
        "Manage appearance and application preferences.",
    )
    st.subheader("👤 Account Profile")

    user = st.session_state.get("user")

    if user is None:
        st.warning("Please login first to manage your account profile.")
        render_auth_panel(database)
        return

    new_name = st.text_input(
        "Full Name",
        value=user.name,
    )

    st.text_input(
        "Email Address",
        value=user.email,
        disabled=True,
    )

    if st.button("💾 Save Profile", use_container_width=True):

        try:
            database.update_profile(
                user.user_id,
                new_name,
            )
        except ValueError as error:
            st.error(str(error))
            return

        st.success("✅ Profile updated successfully!")

        st.session_state.user = database.get_user(user.user_id)

        st.rerun()

    st.divider()

    st.subheader("🤖 AI Status")

    if st.session_state.get("user"):
        st.success("Logged in successfully.")
    else:
        st.warning("Not logged in.")

    st.divider()

    st.subheader("📊 Project Information")

    st.info(

        """
        **EmotiLearn AI**

        Version: 1.0

        Technology Stack:

        • Streamlit

        • TensorFlow BiLSTM

        • Fine-tuned BERT

        • Emotion Fusion

        • Gemini AI

        • SQLite

        • Python
        """
    )

def render_about() -> None:
    """Render project context, workflow, and responsible-AI positioning."""

    render_page_header("About", "Technology with a little more empathy", "A Google Cloud Generative AI SmartBridge Internship project designed to support—not judge—students.")
    st.markdown("**Live MVP:** https://emotion-detection-and-leearning-support.onrender.com")
    columns = st.columns(3)
    cards = (
        ("🔬", "Evidence-informed", "Multiple models contribute signals instead of relying on one brittle guess."),
        ("🧭", "Action-oriented", "Emotion detection becomes a practical learning strategy, not merely a label."),
        ("🤝", "Human-centered", "Guidance is supportive, transparent, and never a mental-health diagnosis."),
    )
    for column, card in zip(columns, cards, strict=True):
        with column:
            render_card(*card)
    with st.expander("How the platform works", expanded=True):
        st.write("Student input → preprocessing → BiLSTM → BERT → keyword rules → emotion fusion → Gemini guidance → history → analytics")
    st.markdown("---")
    st.caption(
        "EmotiLearn AI • Google Cloud Generative AI SmartBridge Internship Project • Version 1.0"
    )

def render_placeholder(page: str) -> None:
    """Render a polished placeholder for a module that has its own later step."""

    descriptions = {
        "Analytics": "Charts, model comparisons, timelines, and emotion distributions arrive in Module 10.",
        "History": "Searchable, filterable predictions and CSV export arrive in Module 11.",
        "Settings": "Theme and Gemini API-key controls arrive in Module 12.",
    }
    render_page_header(page, page, descriptions[page])
    render_module_status(f"{page} interface reserved", descriptions[page])


def render_page(page: str, database: DatabaseManager) -> None:
    """Dispatch a navigation selection to the appropriate page renderer."""

    if page == "Home":
        render_home(database)
    elif page == "Dashboard":
        render_dashboard(database)
    elif page == "Prediction":
        render_prediction(database)
    elif page == "About":
        render_about()
    elif page == "History":
        render_history(database)
    elif page == "Analytics":
        render_analytics(database)
    elif page == "Settings":
        render_settings(database)
    else:
        st.error("This page is not available.")
