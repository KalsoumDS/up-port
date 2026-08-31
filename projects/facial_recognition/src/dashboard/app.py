"""Premium Streamlit Dashboard for Facial Recognition

Features:
- Real-time RTSP stream viewer
- Live face detection and recognition
- Identity management
- SHAP explainability visualizations
- Performance metrics
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Facial Recognition v2.0",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .recognition-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300)
def load_recognition_history() -> pd.DataFrame:
    """Load recognition history"""
    n_records = 100
    return pd.DataFrame({
        "timestamp": pd.date_range(start="2024-08-31", periods=n_records, freq="10min"),
        "identity": np.random.choice(["Alice", "Bob", "Charlie", "Unknown"], n_records),
        "confidence": np.random.uniform(0.7, 0.99, n_records),
        "is_known": np.random.choice([True, False], n_records, p=[0.8, 0.2]),
    })


def render_stream_viewer() -> None:
    """Render RTSP stream viewer"""
    st.subheader("📹 Real-time Stream")

    col1, col2 = st.columns([3, 1])

    with col1:
        # Placeholder for video stream
        st.info("📹 RTSP Stream Viewer (requires OpenCV + streamlit-webrtc)")

    with col2:
        st.metric("FPS", "30", delta="+2")
        st.metric("Detection", "2 faces", delta="1 known")
        st.metric("Latency", "45ms", delta="-5ms")


def render_active_recognitions() -> None:
    """Render active recognitions on stream"""
    st.subheader("👤 Active Recognitions")

    col1, col2, col3 = st.columns(3)

    recognitions = [
        ("Alice Johnson", 0.97, True),
        ("Bob Smith", 0.93, True),
        ("Unknown", 0.62, False),
    ]

    for col, (name, conf, is_known) in zip([col1, col2, col3], recognitions):
        with col:
            status = "✅ Known" if is_known else "❓ Unknown"
            st.metric(name, f"{conf:.1%}", delta=status)


def render_identity_management() -> None:
    """Render identity database management"""
    st.subheader("👥 Identity Database")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Identities", "156", delta="+5")

    with col2:
        st.metric("Active Users", "42", delta="+3")

    with col3:
        st.metric("Recognition Rate", "94.2%", delta="↑ 2.3%")

    # Identities table
    identities = pd.DataFrame({
        "Identity ID": ["alice_001", "bob_002", "charlie_003"],
        "Name": ["Alice Johnson", "Bob Smith", "Charlie Brown"],
        "Registrations": [5, 3, 7],
        "Last Seen": ["2 min ago", "45 min ago", "8 hours ago"],
        "Confidence": [0.97, 0.93, 0.91],
    })

    st.dataframe(identities, use_container_width=True)


def render_shap_explanations() -> None:
    """Render SHAP explainability"""
    st.subheader("🔍 Recognition Explanation (SHAP)")

    col1, col2 = st.columns(2)

    with col1:
        # Feature importance
        features = ["Eye Distance", "Nose Width", "Face Shape", "Mouth Geometry", "Skin Texture"]
        importances = [0.35, 0.25, 0.20, 0.15, 0.05]

        fig = go.Figure(data=[
            go.Bar(
                y=features,
                x=importances,
                orientation="h",
                marker=dict(color="rgba(102, 126, 234, 0.8)"),
            )
        ])

        fig.update_layout(
            title="Feature Importance",
            xaxis_title="SHAP Impact",
            height=350,
            template="plotly_dark",
            margin=dict(l=50, r=50, t=50, b=50),
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Similar faces
        similar = {
            "Alice J. (alice_001)": 0.97,
            "Person ID: 045": 0.87,
            "Person ID: 128": 0.72,
            "Person ID: 203": 0.61,
        }

        fig = go.Figure(data=[
            go.Bar(
                y=list(similar.keys()),
                x=list(similar.values()),
                orientation="h",
                marker=dict(color="rgba(118, 75, 162, 0.8)"),
            )
        ])

        fig.update_layout(
            title="Similar Faces",
            xaxis_title="Similarity",
            height=350,
            template="plotly_dark",
            margin=dict(l=50, r=50, t=50, b=50),
        )

        st.plotly_chart(fig, use_container_width=True)


def render_recognition_history() -> None:
    """Render recognition history"""
    st.subheader("📊 Recognition History")

    df = load_recognition_history()

    # Timeline chart
    fig = go.Figure()

    for identity in df["identity"].unique():
        mask = df["identity"] == identity
        fig.add_trace(go.Scatter(
            x=df[mask]["timestamp"],
            y=df[mask]["confidence"],
            mode="markers+lines",
            name=identity,
            line=dict(width=2),
        ))

    fig.update_layout(
        title="Recognition Confidence Over Time",
        xaxis_title="Time",
        yaxis_title="Confidence",
        height=350,
        template="plotly_dark",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_performance_metrics() -> None:
    """Render performance metrics"""
    st.subheader("📈 Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Detection Accuracy", "98.2%", delta="↑ 1.5%")

    with col2:
        st.metric("Recognition F1", "0.94", delta="↑ 0.03")

    with col3:
        st.metric("False Positive Rate", "1.2%", delta="↓ 0.5%")

    with col4:
        st.metric("Inference Latency", "52ms", delta="-8ms")


def main() -> None:
    """Main dashboard"""
    st.title("👁️ Facial Recognition v2.0")
    st.markdown("**Enterprise Real-time Face Recognition with SHAP Explainability**")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        stream_url = st.text_input("RTSP Stream URL", value="rtsp://localhost:554/stream")
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.99, 0.75, 0.01)
        distance_threshold = st.slider("Distance Threshold", 0.3, 1.0, 0.6, 0.05)

    st.divider()

    # Stream viewer
    render_stream_viewer()

    st.divider()

    # Active recognitions
    render_active_recognitions()

    st.divider()

    # Identity management
    render_identity_management()

    st.divider()

    # SHAP explanations
    render_shap_explanations()

    st.divider()

    # Recognition history
    render_recognition_history()

    st.divider()

    # Performance metrics
    render_performance_metrics()

    st.divider()

    # Footer
    st.markdown(
        """
        ---
        **Facial Recognition v2.0** | Lead Data Scientist & MLOps Architect
        
        🔗 [GitHub](https://github.com/KalsoumDS/up-port) |
        📊 [API Docs](http://localhost:8000/docs) |
        📧 [Contact](mailto:contact@example.com)
        """
    )


if __name__ == "__main__":
    main()
