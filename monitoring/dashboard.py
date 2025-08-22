import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any

COORDINATOR_URL = "http://localhost:8000"
REFRESH_INTERVAL = 2  # seconds

st.set_page_config(
    page_title="Distributed Inference Monitor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=1)
def fetch_metrics() -> Dict:
    """Fetch metrics from coordinator"""
    try:
        response = requests.get(f"{COORDINATOR_URL}/metrics", timeout=2)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}

@st.cache_data(ttl=1)
def fetch_workers() -> Dict:
    """Fetch worker status"""
    try:
        response = requests.get(f"{COORDINATOR_URL}/workers/status", timeout=2)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {"workers": [], "total_active": 0, "total_requests": 0}

@st.cache_data(ttl=1)
def fetch_queue_status() -> Dict:
    """Fetch queue status"""
    try:
        response = requests.get(f"{COORDINATOR_URL}/queue/status", timeout=2)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {"size": 0, "high_priority": 0, "normal_priority": 0, "low_priority": 0}

@st.cache_data(ttl=1)
def fetch_health() -> Dict:
    """Fetch health status"""
    try:
        response = requests.get(f"{COORDINATOR_URL}/health", timeout=2)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {"status": "unknown", "stats": {}}

def get_status_color(status: str) -> str:
    """Get color for status"""
    colors = {
        "healthy": "🟢",
        "degraded": "🟡",
        "unhealthy": "🔴",
        "unknown": "⚫"
    }
    return colors.get(status, "⚫")

def main():
    """Main dashboard function"""
    
    # Title and header
    st.title("Distributed AI Inference System Monitor")
    st.markdown("---")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        refresh_rate = st.slider("Refresh Rate (seconds)", 1, 10, REFRESH_INTERVAL)
        
        st.markdown("---")
        st.header("Chaos Engineering")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Enable Chaos", use_container_width=True):
                for port in [8001, 8002, 8003]:
                    try:
                        requests.post(f"http://localhost:{port}/chaos/enable")
                    except:
                        pass
                st.success("Chaos enabled!")
        
        with col2:
            if st.button("Disable Chaos", use_container_width=True):
                for port in [8001, 8002, 8003]:
                    try:
                        requests.post(f"http://localhost:{port}/chaos/disable")
                    except:
                        pass
                st.success("Chaos disabled!")
        
        if st.button("Simulate Crash", use_container_width=True):
            try:
                requests.post("http://localhost:8001/simulate/crash")
            except:
                pass
            st.warning("Crash simulated on Worker 1!")
        
        st.markdown("---")
        st.header("Test Traffic")
        
        num_requests = st.number_input("Number of Requests", 1, 1000, 10)
        if st.button("Send Test Requests", use_container_width=True):
            send_test_requests(num_requests)
    
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()

    metrics = fetch_metrics()
    workers = fetch_workers()
    queue = fetch_queue_status()
    health = fetch_health()

    st.header("📈 System Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        status = health.get("status", "unknown")
        st.metric(
            "System Status",
            f"{get_status_color(status)} {status.upper()}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Active Workers",
            workers.get("total_active", 0),
            delta=f"/{len(workers.get('workers', []))}"
        )
    
    with col3:
        st.metric(
            "Total Requests",
            metrics.get("requests_total", 0),
            delta=f"+{metrics.get('requests_total', 0) - st.session_state.get('last_total', 0)}"
        )
        st.session_state['last_total'] = metrics.get("requests_total", 0)
    
    with col4:
        success_rate = metrics.get("success_rate", 0) * 100
        st.metric(
            "Success Rate",
            f"{success_rate:.1f}%",
            delta=None
        )
    
    with col5:
        st.metric(
            "Queue Size",
            queue.get("size", 0),
            delta=f"/{queue.get('max_size', 1000)}"
        )
    
    st.markdown("---")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Latency Distribution")
        
        if "latency_p50" in metrics:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=["P50", "P99", "Average"],
                y=[
                    metrics.get("latency_p50", 0),
                    metrics.get("latency_p99", 0),
                    metrics.get("latency_avg", 0)
                ],
                text=[
                    f"{metrics.get('latency_p50', 0):.1f}ms",
                    f"{metrics.get('latency_p99', 0):.1f}ms",
                    f"{metrics.get('latency_avg', 0):.1f}ms"
                ],
                textposition="outside",
                marker_color=['green', 'orange', 'blue']
            ))
            
            fig.update_layout(
                yaxis_title="Latency (ms)",
                showlegend=False,
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No latency data available yet")
    
    with col2:
        st.subheader("Request Distribution")
        
        if metrics.get("requests_total", 0) > 0:
            fig = go.Figure(data=[
                go.Pie(
                    labels=["Success", "Failed"],
                    values=[
                        metrics.get("requests_success", 0),
                        metrics.get("requests_failed", 0)
                    ],
                    hole=0.3,
                    marker_colors=['#00CC88', '#FF4444']
                )
            ])
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No requests processed yet")
    
    st.markdown("---")
    
    st.header("Worker Status")
    
    if workers.get("workers"):
        worker_df = pd.DataFrame(workers["workers"])

        cols = st.columns(len(workers["workers"]))
        
        for idx, (col, worker) in enumerate(zip(cols, workers["workers"])):
            with col:
                status_color = get_status_color(worker["status"])
                st.markdown(f"""
                <div style="
                    border: 2px solid {'#00CC88' if worker['status'] == 'healthy' else '#FF4444'};
                    border-radius: 10px;
                    padding: 15px;
                    margin: 5px;
                ">
                    <h4>{status_color} {worker['id']}</h4>
                    <p><strong>Model:</strong> {worker['model']}</p>
                    <p><strong>Load:</strong> {worker['load']*100:.1f}%</p>
                    <p><strong>Processed:</strong> {worker['processed']}</p>
                    <p><strong>Errors:</strong> {worker['errors']}</p>
                    <p><strong>Avg Latency:</strong> {worker['avg_latency']:.1f}ms</p>
                </div>
                """, unsafe_allow_html=True)

        st.subheader("Worker Performance")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Request Distribution", "Load Distribution")
        )

        fig.add_trace(
            go.Bar(
                x=[w["id"] for w in workers["workers"]],
                y=[w["processed"] for w in workers["workers"]],
                name="Processed",
                marker_color='lightblue'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                x=[w["id"] for w in workers["workers"]],
                y=[w["load"]*100 for w in workers["workers"]],
                name="Load %",
                marker_color='lightgreen'
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No workers connected")
    
    st.markdown("---")

    st.header("Queue Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queued", queue.get("size", 0))
    
    with col2:
        st.metric("High Priority", queue.get("high_priority", 0))
    
    with col3:
        st.metric("Normal Priority", queue.get("normal_priority", 0))
    
    with col4:
        st.metric("Low Priority", queue.get("low_priority", 0))

    if queue.get("size", 0) > 0:
        fig = go.Figure(data=[
            go.Bar(
                x=["High", "Normal", "Low"],
                y=[
                    queue.get("high_priority", 0),
                    queue.get("normal_priority", 0),
                    queue.get("low_priority", 0)
                ],
                marker_color=['red', 'yellow', 'green']
            )
        ])
        
        fig.update_layout(
            yaxis_title="Number of Requests",
            height=200
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")

    st.header("📜 Recent Activity")
    
    if "activity_log" not in st.session_state:
        st.session_state.activity_log = []

    if metrics.get("requests_total", 0) > st.session_state.get("last_logged_total", 0):
        st.session_state.activity_log.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "event": f"Processed {metrics.get('requests_total', 0) - st.session_state.get('last_logged_total', 0)} new requests",
            "type": "info"
        })
        st.session_state.last_logged_total = metrics.get("requests_total", 0)

    for log in st.session_state.activity_log[-5:]:
        if log["type"] == "error":
            st.error(f"[{log['timestamp']}] {log['event']}")
        elif log["type"] == "warning":
            st.warning(f"[{log['timestamp']}] {log['event']}")
        else:
            st.info(f"[{log['timestamp']}] {log['event']}")

def send_test_requests(num_requests: int):
    """Send test requests to the coordinator"""
    import base64
    import random

    dummy_image = base64.b64encode(b"dummy_image_data").decode()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    success_count = 0
    failed_count = 0
    
    for i in range(num_requests):
        try:
            priority = random.choice(["high", "normal", "low"])
            model = random.choice(["mobilenet", "resnet18", "efficientnet", "any"])
            
            response = requests.post(
                f"{COORDINATOR_URL}/inference",
                json={
                    "data": dummy_image,
                    "model_type": model,
                    "priority": priority
                },
                timeout=5
            )
            
            if response.status_code == 200:
                success_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            failed_count += 1

        progress = (i + 1) / num_requests
        progress_bar.progress(progress)
        status_text.text(f"Sent {i+1}/{num_requests} - Success: {success_count}, Failed: {failed_count}")

    st.session_state.activity_log.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "event": f"Sent {num_requests} test requests - {success_count} succeeded, {failed_count} failed",
        "type": "info" if failed_count == 0 else "warning"
    })
    
    st.success(f"Test completed! Success: {success_count}, Failed: {failed_count}")
    time.sleep(2)
    st.rerun()

if __name__ == "__main__":
    main()