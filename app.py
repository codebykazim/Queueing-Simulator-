# app.py - COMPLETE FIXED VERSION
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
from typing import Dict, Any, Optional, List
import math

# Backend URL
BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Queue Simulator",
    layout="wide",
    page_icon="🚀"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FEF3C7;
        border: 1px solid #F59E0B;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #FEE2E2;
        border: 1px solid #EF4444;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #DBEAFE;
        border: 1px solid #3B82F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .dist-params {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🚀 Queueing System Simulator</div>', unsafe_allow_html=True)
st.markdown("---")

# Initialize session state
if 'sim_data' not in st.session_state:
    st.session_state.sim_data = None
if 'sim_params' not in st.session_state:
    st.session_state.sim_params = {}

# Helper function to convert infinity values to "∞" for display
def safe_json_serialize(obj):
    """Convert infinity values to strings for display"""
    if isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_serialize(i) for i in obj]
    elif isinstance(obj, float) and (obj == float('inf') or obj == float('-inf')):
        return "∞" if obj > 0 else "-∞"
    elif isinstance(obj, float) and np.isnan(obj):
        return "NaN"
    else:
        return obj

# Helper function for API calls
def call_backend(endpoint: str, payload: Dict) -> Optional[Dict]:
    """Make API call to backend with error handling"""
    try:
        response = requests.post(f"{BACKEND_URL}/{endpoint}", json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Backend Error: {str(e)}")
        if response.status_code == 422:
            try:
                error_details = response.json()
                st.error(f"Validation Error: {error_details}")
            except:
                pass
        return None
    except Exception as e:
        st.error(f"❌ Unexpected Error: {str(e)}")
        return None

# Helper to get distribution parameters UI
def get_distribution_params(dist_type: str, prefix: str, default_params: Dict = None) -> Dict:
    """Create UI for distribution parameters"""
    if default_params is None:
        default_params = {}

    params = {}

    if dist_type == "exponential":
        # ✅ FIX: Service ke liye μ, Arrival ke liye λ
        if "service" in prefix.lower():
            label = "μ"  # Sirf mu sign
            default_value = 1.0  # Service ke liye default 1.0
        else:
            label = "λ"  # Sirf lambda sign
            default_value = 0.6  # Arrival ke liye default 0.6

        params["rate"] = st.number_input(
            label,
            min_value=0.01,
            value=default_params.get("rate", default_value),
            step=0.1,
            key=f"{prefix}_exp_rate"
        )

    elif dist_type == "uniform":
        col1, col2 = st.columns(2)
        with col1:
            params["min"] = st.number_input(
                "Min",
                min_value=0.01,
                value=default_params.get("min", 0.5),
                step=0.1,
                key=f"{prefix}_unif_min"
            )
        with col2:
            # Ensure max > min
            min_val = params["min"] + 0.01
            params["max"] = st.number_input(
                "Max",
                min_value=min_val,
                value=default_params.get("max", 2.0),
                step=0.1,
                key=f"{prefix}_unif_max"
            )

    elif dist_type == "normal":
        col1, col2 = st.columns(2)
        with col1:
            params["mean"] = st.number_input(
                "Mean",
                min_value=0.01,
                value=default_params.get("mean", 1.0),
                step=0.1,
                key=f"{prefix}_norm_mean"
            )
        with col2:
            params["std"] = st.number_input(
                "Std Dev",
                min_value=0.01,
                value=default_params.get("std", 0.2),
                step=0.1,
                key=f"{prefix}_norm_std"
            )
        if params["std"] > params["mean"] / 2:
            st.warning("⚠️ High standard deviation may cause negative times. Consider using gamma distribution.")

    elif dist_type == "gamma":
        col1, col2 = st.columns(2)
        with col1:
            params["shape"] = st.number_input(
                "Shape",
                min_value=0.01,
                value=default_params.get("shape", 2.0),
                step=0.1,
                key=f"{prefix}_gamma_shape"
            )
        with col2:
            params["scale"] = st.number_input(
                "Scale",
                min_value=0.01,
                value=default_params.get("scale", 0.5),
                step=0.1,
                key=f"{prefix}_gamma_scale"
            )

    elif dist_type == "poisson":
        params["lambda"] = st.number_input(
            "λ",  # Sirf lambda sign
            min_value=0.01,
            value=default_params.get("lambda", 4.61),
            step=0.1,
            key=f"{prefix}_poisson_lambda"
        )

    return params
# Calculate mean and variance for display
def calculate_moments(dist_type: str, params: Dict) -> tuple:
    """Calculate mean and variance for display purposes"""
    try:
        if dist_type == "exponential":
            rate = params.get("rate", 1.0)
            mean = 1.0 / rate
            var = 1.0 / (rate * rate)
            return mean, var, rate

        elif dist_type == "uniform":
            a = params.get("min", 0.5)
            b = params.get("max", 2.0)
            mean = (a + b) / 2.0
            var = ((b - a) ** 2) / 12.0
            rate = 1.0 / mean if mean > 0 else 0
            return mean, var, rate

        elif dist_type == "normal":
            mean = params.get("mean", 1.0)
            std = params.get("std", 0.2)
            var = std * std
            rate = 1.0 / mean if mean > 0 else 0
            return mean, var, rate

        elif dist_type == "gamma":
            shape = params.get("shape", 2.0)
            scale = params.get("scale", 0.5)
            mean = shape * scale
            var = shape * (scale ** 2)
            rate = 1.0 / mean if mean > 0 else 0
            return mean, var, rate

        elif dist_type == "poisson":
            lam = params.get("lambda", 4.61)
            mean = lam
            var = lam
            # For Poisson, the rate is λ (arrival rate)
            # For time between events, it's exponential with rate λ
            rate = lam
            return mean, var, rate

    except Exception as e:
        return 0, 0, 0

    return 0, 0, 0

# Sidebar navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose Mode",
        ["📊 Analytical Calculator", "⚙️ Simulation", "📈 Visualization Dashboard"],
        index=0,
        key="page_selector"
    )

    st.markdown("---")
    st.header("Backend Status")
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ Backend Connected")
        else:
            st.error(f"❌ Backend Error: {response.status_code}")
    except:
        st.error("❌ Backend Not Reachable")
        st.info("Start backend with:")
        st.code("uvicorn main:app --reload --host 0.0.0.0 --port 8000", language="bash")

    st.markdown("---")
    st.header("Model Compatibility")
    st.markdown("""
    **M/M/*:** Exponential arrivals & service
    **M/G/*:** Exponential arrivals, General service
    **G/G/*:** General arrivals & service
    **Note:** Simulation uses CP-table method
    """)

# ==============================================
# 📊 ANALYTICAL CALCULATOR PAGE
# ==============================================
if page == "📊 Analytical Calculator":
    st.markdown('<div class="sub-header">Analytical Queueing Models Calculator</div>', unsafe_allow_html=True)

    # Model selection and explanation
    col1, col2 = st.columns([1, 2])
    with col1:
        model = st.selectbox(
            "Queueing Model",
            ["M/M/1", "M/M/c", "M/G/1", "M/G/c", "G/G/1", "G/G/c"],
            help="""Select the queueing model:
            • M/M/1: Poisson arrivals, Exponential service, Single server
            • M/M/c: Poisson arrivals, Exponential service, Multiple servers
            • M/G/1: Poisson arrivals, General service, Single server
            • M/G/c: Poisson arrivals, General service, Multiple servers
            • G/G/1: General arrivals, General service, Single server
            • G/G/c: General arrivals, General service, Multiple servers""",
            key="analytical_model"
        )

    with col2:
        # Show model description
        model_descriptions = {
            "M/M/1": "Poisson arrivals (exponential interarrival times), exponential service times, single server.",
            "M/M/c": "Poisson arrivals (exponential interarrival times), exponential service times, multiple servers.",
            "M/G/1": "Poisson arrivals (exponential interarrival times), general service distribution, single server.",
            "M/G/c": "Poisson arrivals (exponential interarrival times), general service distribution, multiple servers.",
            "G/G/1": "General interarrival distribution, general service distribution, single server.",
            "G/G/c": "General interarrival distribution, general service distribution, multiple servers."
        }
        st.info(model_descriptions[model])

    # Input parameters based on model
    st.markdown("### Model Parameters")

    # Initialize parameters
    lambda_val = None
    mu_val = None
    c_val = 1
    arrival_spec = None
    service_spec = None

    # Split into columns for better layout
    col_params1, col_params2 = st.columns(2)

    with col_params1:
        st.markdown("#### Arrival Distribution")

        # Determine arrival distribution based on model
        if model.startswith("M/"):
            # M/* models require exponential arrivals
            st.info("M/* models require exponential arrivals")
            arrival_dist = "exponential"
            lambda_val = st.number_input(
                "Arrival Rate (λ)",
                min_value=0.01,
                value=0.6,
                step=0.1,
                help="Average number of arrivals per unit time",
                key="analytical_arrival_lambda"
            )
            arrival_params = {"rate": lambda_val}
        else:
            # G/* models can have general arrivals
            arrival_dist = st.selectbox(
                "Arrival Distribution Type",
                ["uniform", "normal", "gamma", "exponential"],
                key="analytical_arrival_dist"
            )
            arrival_params = get_distribution_params(arrival_dist, "analytical_arrival")

            # Calculate effective lambda for display
            if arrival_dist == "exponential":
                lambda_val = arrival_params["rate"]
            else:
                mean, var, _ = calculate_moments(arrival_dist, arrival_params)
                lambda_val = 1.0 / mean if mean > 0 else 0

        arrival_spec = {
            "dist_type": arrival_dist,
            "params": arrival_params
        }

        # Show arrival moments
        if arrival_dist != "exponential" or model.startswith("G/"):
            mean, var, _ = calculate_moments(arrival_dist, arrival_params)
            st.markdown(f"**Arrival Moments:** Mean = {mean:.4f}, Variance = {var:.4f}")
            if lambda_val:
                st.markdown(f"**Effective Arrival Rate (λ):** {lambda_val:.4f}")

    with col_params2:
        st.markdown("#### Service Distribution")

        # Determine service distribution based on model
        if model in ["M/M/1", "M/M/c"]:
            # M/M models require exponential service
            st.info("M/M models require exponential service")
            service_dist = "exponential"
            mu_val = st.number_input(
                "Service Rate (μ)",
                min_value=0.01,
                value=1.0,
                step=0.1,
                help="Average service rate per server",
                key="analytical_service_mu"
            )
            service_params = {"rate": mu_val}
        elif model in ["M/G/1", "M/G/c"]:
            # M/G models can have general service (not exponential)
            service_dist = st.selectbox(
                "Service Distribution Type",
                ["uniform", "normal", "gamma"],
                key="analytical_service_dist_mg"
            )
            service_params = get_distribution_params(service_dist, "analytical_service_mg")

            # Calculate effective mu for display
            mean, var, _ = calculate_moments(service_dist, service_params)
            mu_val = 1.0 / mean if mean > 0 else 0
        else:
            # G/G models can have general service
            service_dist = st.selectbox(
                "Service Distribution Type",
                ["uniform", "normal", "gamma", "exponential"],
                key="analytical_service_dist_gg"
            )
            service_params = get_distribution_params(service_dist, "analytical_service_gg")

            # Calculate effective mu for display
            if service_dist == "exponential":
                mu_val = service_params["rate"]
            else:
                mean, var, _ = calculate_moments(service_dist, service_params)
                mu_val = 1.0 / mean if mean > 0 else 0

        service_spec = {
            "dist_type": service_dist,
            "params": service_params
        }

        # Show service moments
        if service_dist != "exponential" or model in ["M/G/1", "M/G/c", "G/G/1", "G/G/c"]:
            mean, var, _ = calculate_moments(service_dist, service_params)
            st.markdown(f"**Service Moments:** Mean = {mean:.4f}, Variance = {var:.4f}")
            if mu_val:
                st.markdown(f"**Effective Service Rate (μ):** {mu_val:.4f}")

        # Server count (for multi-server models)
        if model in ["M/M/c", "M/G/c", "G/G/c"]:
            c_val = st.number_input(
                "Number of Servers (c)",
                min_value=1,
                value=2,
                step=1,
                help="Number of parallel servers in the system",
                key="analytical_c_val"
            )
        else:
            c_val = 1

    # Show derived parameters
    with st.expander("📊 Derived Parameters"):
        if lambda_val:
            st.write(f"**Effective Arrival Rate (λ):** {lambda_val:.4f}")
        if mu_val:
            st.write(f"**Effective Service Rate (μ):** {mu_val:.4f}")
        if lambda_val and mu_val:
            utilization = lambda_val / (c_val * mu_val) if c_val > 0 else lambda_val / mu_val
            st.write(f"**Utilization (ρ):** {utilization:.4f}")
            if utilization >= 1:
                st.error("⚠️ System is unstable (ρ ≥ 1). Queue will grow infinitely!")

    # Run Analytical Calculation
    if st.button("🔬 Calculate Analytical Results", type="primary", use_container_width=True, key="analytical_calculate"):
        with st.spinner("Computing analytical solution..."):
            # Prepare payload according to backend schema
            payload = {
                "model": model,
                "servers": int(c_val)
            }

            # Add lambda and mu based on model requirements
            if model.startswith("M"):  # M/* models need lambda
                if lambda_val is not None:
                    payload["lambda_"] = float(lambda_val)

            if model in ["M/M/1", "M/M/c"]:  # M/M models need mu
                if mu_val is not None:
                    payload["mu"] = float(mu_val)
            # For M/G and G/G models, mu will be derived from service distribution

            # Add distribution specs
            if arrival_spec:
                payload["arrival"] = arrival_spec
            if service_spec:
                payload["service"] = service_spec

            # Debug info
            with st.expander("🔧 Debug: Request Payload"):
                st.json(payload)

            # Call backend
            result = call_backend("analytical", payload)

            if result:
                # Convert any inf values for display
                safe_result = safe_json_serialize(result)

                # Display results in a nice layout
                st.markdown('<div class="sub-header">📈 Analytical Results</div>', unsafe_allow_html=True)

                # Check for unstable system
                util = safe_result.get('utilization')
                is_unstable = False
                if isinstance(util, str) or (isinstance(util, float) and util >= 1):
                    is_unstable = True
                    st.markdown("""
                    <div class="error-box">
                    ⚠️ **SYSTEM UNSTABLE** - Utilization ρ ≥ 1
                    The queue will grow infinitely over time.
                    **Recommendation:** Increase service rate (μ) or add more servers.
                    </div>
                    """, unsafe_allow_html=True)

                # Key metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    util_display = safe_result['utilization']
                    if isinstance(util_display, str) or (isinstance(util_display, float) and util_display >= 1):
                        util_color = "inverse"
                        delta_text = "Unstable"
                    elif isinstance(util_display, float) and util_display >= 0.7:
                        util_color = "inverse"
                        delta_text = "High Load"
                    else:
                        util_color = "normal"
                        delta_text = "Stable"

                    st.metric("Utilization (ρ)",
                             f"{util_display}" if isinstance(util_display, str) else f"{util_display:.3f}",
                             delta=delta_text,
                             delta_color=util_color)

                with col2:
                    lq_display = safe_result['Lq']
                    st.metric("Queue Length (Lq)",
                             f"{lq_display}" if isinstance(lq_display, str) else f"{lq_display:.3f}")

                with col3:
                    wq_display = safe_result['Wq']
                    st.metric("Wait in Queue (Wq)",
                             f"{wq_display}" if isinstance(wq_display, str) else f"{wq_display:.3f}")

                with col4:
                    l_display = safe_result['L']
                    st.metric("System Length (L)",
                             f"{l_display}" if isinstance(l_display, str) else f"{l_display:.3f}")

                # Additional metrics
                col5, col6, col7, col8 = st.columns(4)
                with col5:
                    w_display = safe_result['W']
                    st.metric("Total Wait (W)",
                             f"{w_display}" if isinstance(w_display, str) else f"{w_display:.3f}")

                with col6:
                    interarrival_rate = safe_result['interarrival_rate']
                    st.metric("Arrival Rate (λ)", f"{interarrival_rate:.3f}")

                with col7:
                    service_rate = safe_result['service_rate']
                    st.metric("Service Rate (μ)", f"{service_rate:.3f}")

                with col8:
                    var_service = safe_result['var_services']
                    st.metric("Service Variance", f"{var_service:.3f}")

                # Display any notes from backend
                if safe_result.get('note'):
                    st.markdown(f'<div class="info-box">📝 **Note:** {safe_result["note"]}</div>', unsafe_allow_html=True)

                # Show complete result
                with st.expander("📋 Complete Result Data"):
                    st.json(safe_result)

# ==============================================
# ⚙️ MONTE CARLO SIMULATION PAGE
# ==============================================
elif page == "⚙️ Simulation":
    st.markdown('<div class="sub-header">Simulation (CP-Table Method)</div>', unsafe_allow_html=True)
    st.markdown("""
    Simulate queue behavior using CP-table method.
    **Note:** Number of customers is determined by the CP-table generation (stops when CP=1).
    The n_customers parameter acts as a maximum cap.
    """)

    # Simulation parameters
    col_sim1, col_sim2, col_sim3 = st.columns(3)
    with col_sim1:
        sim_model = st.selectbox(
            "Simulation Model",
            ["M/M/c", "M/G/c", "G/G/c"],
            help="""Select model for simulation:
            • M/M/c: Exponential arrivals, Exponential service
            • M/G/c: Exponential arrivals, General service
            • G/G/c: General arrivals, General service""",
            key="sim_model"
        )
    with col_sim2:
        servers = st.number_input(
            "Number of Servers",
            min_value=1,
            value=2,
            step=1,
            key="sim_servers"
        )
    with col_sim3:
        max_customers = st.number_input(
            "Maximum Customers",
            min_value=10,
            max_value=200000,
            value=100,
            step=10,
            help="Maximum number of customers to simulate",
            key="max_customers_input"
        )

    # Random seed
    seed = st.number_input("Random Seed", min_value=1, value=123, step=1, help="Seed for reproducible results", key="sim_seed")

    # Distribution configuration
    st.markdown("### Distribution Configuration")
    col_dist1, col_dist2 = st.columns(2)

    with col_dist1:
        st.markdown("#### Arrival Distribution")
        st.markdown('<div class="dist-params">', unsafe_allow_html=True)

        # Determine arrival distribution based on model
        if sim_model.startswith("M/"):
            # M/* models use exponential arrivals (for CP-table)
            arr_dist_sim = "exponential"
            st.info("M/* models use exponential arrivals for Poisson CP-table")
        else:
            # G/G/c models can have general arrivals
            arr_dist_sim = st.selectbox(
                "Arrival Distribution Type",
                ["uniform", "normal", "gamma"],
                key="arr_dist_sim"
            )

        # Get parameters based on distribution
        default_arrival_params = {
            # "exponential": {"rate": 0.6},
            "uniform": {"min": 0.5, "max": 2.0},
            "normal": {"mean": 1.0, "std": 0.2},
            "gamma": {"shape": 2.0, "scale": 0.5}
        }

        arrival_params = get_distribution_params(
            arr_dist_sim,
            "sim_arrival",
            default_arrival_params.get(arr_dist_sim, {})
        )

        # Calculate and display moments
        # mean, var, rate = calculate_moments(arr_dist_sim, arrival_params)
        # st.markdown(f"**Mean:** {mean:.4f}")
        # st.markdown(f"**Variance:** {var:.4f}")
        # st.markdown(f"**Rate (λ):** {rate:.4f}")

        arrival_sim = {
            "dist_type": arr_dist_sim,
            "params": arrival_params
        }
        st.markdown('</div>', unsafe_allow_html=True)

    with col_dist2:
        st.markdown("#### Service Distribution")
        st.markdown('<div class="dist-params">', unsafe_allow_html=True)

        # Determine service distribution based on model
        if sim_model == "M/M/c":
            serv_dist_sim = "exponential"
            st.info("M/M/c uses exponential service")
        else:
            # M/G/c and G/G/c models can have general service
            serv_dist_sim = st.selectbox(
                "Service Distribution Type",
                ["uniform", "normal", "gamma"],
                key="serv_dist_sim"
            )

        # Get parameters based on distribution
        default_service_params = {
            # "exponential": {"rate": 1.0},
            "uniform": {"min": 0.2, "max": 1.0},
            "normal": {"mean": 0.8, "std": 0.1},
            "gamma": {"shape": 2.0, "scale": 0.5}
        }

        service_params = get_distribution_params(
            serv_dist_sim,
            "sim_service",
            default_service_params.get(serv_dist_sim, {})
        )

        # Calculate and display moments
        # mean, var, rate = calculate_moments(serv_dist_sim, service_params)
        # st.markdown(f"**Mean:** {mean:.4f}")
        # st.markdown(f"**Variance:** {var:.4f}")
        # st.markdown(f"**Rate (μ):** {rate:.4f}")

        service_sim = {
            "dist_type": serv_dist_sim,
            "params": service_params
        }
        st.markdown('</div>', unsafe_allow_html=True)

    # System utilization preview
    with st.expander("📊 System Utilization Preview"):
        # Calculate approximate utilization
        arr_mean, arr_var, arr_rate = calculate_moments(arr_dist_sim, arrival_params)
        serv_mean, serv_var, serv_rate = calculate_moments(serv_dist_sim, service_params)

        if arr_mean > 0 and serv_mean > 0:
            lambda_eff = arr_rate
            mu_eff = serv_rate
            utilization = lambda_eff / (servers * mu_eff) if mu_eff > 0 else 0

            col_util1, col_util2, col_util3 = st.columns(3)
            with col_util1:
                st.metric("Arrival Rate (λ)", f"{lambda_eff:.4f}")
            with col_util2:
                st.metric("Service Rate (μ)", f"{mu_eff:.4f}")
            with col_util3:
                st.metric("Utilization (ρ)", f"{utilization:.4f}")

            if utilization >= 1:
                st.error("⚠️ System appears unstable (ρ ≥ 1). Consider reducing arrival rate or increasing service rate.")
            elif utilization >= 0.9:
                st.warning("⚠️ System heavily loaded (ρ ≥ 0.9). Long queues expected.")
            elif utilization <= 0.3:
                st.info("✅ System lightly loaded (ρ ≤ 0.3). Good performance expected.")

    # Run Simulation button
    if st.button("🎲 Run Simulation", type="primary", use_container_width=True, key="run_simulation"):
        # Prepare payload according to backend schema
        payload = {
            "model": sim_model,
            "servers": int(servers),
            "n_customers": int(max_customers),
            "arrival": arrival_sim,
            "service": service_sim,
            "seed": int(seed) if seed else None
        }

        with st.spinner(f"Simulating {sim_model} with {servers} servers..."):
            # Debug info
            with st.expander("🔧 Debug: Simulation Request Payload"):
                st.json(payload)

            result = call_backend("simulate", payload)

            if result:
                # Store in session state for visualization page
                st.session_state.sim_data = result
                st.session_state.sim_params = {
                    "max_customers": max_customers,
                    "servers": servers,
                    "model": sim_model,
                    "arrival_dist": arr_dist_sim,
                    "service_dist": serv_dist_sim,
                    "seed": seed,
                    "arrival_params": arrival_params,
                    "service_params": service_params
                }

                # Display summary statistics
                st.markdown('<div class="sub-header">📊 Simulation Results</div>', unsafe_allow_html=True)

                # Extract data
                rows = result["rows"]
                n_customers = len(rows)
                st.success(f"✅ **Simulation completed successfully!**")
                st.info(f"**Actual customers simulated:** {n_customers}")

                if n_customers > 0:
                    # Calculate statistics
                    wait_times = result["wait_times"]
                    turnaround_times = result["turnaround_times"]
                    response_times = result["response_times"]
                    arrival_times = result["arrival_times"]

                    # Performance metrics
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    with col_stat1:
                        avg_wait = np.mean(wait_times) if wait_times else 0
                        std_wait = np.std(wait_times) if wait_times else 0
                        st.metric("Avg Wait Time", f"{avg_wait:.3f}",
                                 delta=f"±{std_wait:.3f}")
                    with col_stat2:
                        avg_turnaround = np.mean(turnaround_times) if turnaround_times else 0
                        std_turnaround = np.std(turnaround_times) if turnaround_times else 0
                        st.metric("Avg Turnaround", f"{avg_turnaround:.3f}",
                                 delta=f"±{std_turnaround:.3f}")
                    with col_stat3:
                        avg_response = np.mean(response_times) if response_times else 0
                        std_response = np.std(response_times) if response_times else 0
                        st.metric("Avg Response", f"{avg_response:.3f}",
                                 delta=f"±{std_response:.3f}")
                    with col_stat4:
                        # Calculate utilization from simulation data
                        total_service_time = sum([row['service_time'] for row in rows]) if rows else 0
                        total_time = arrival_times[-1] if arrival_times else 1
                        utilization = total_service_time / (servers * total_time) if total_time > 0 else 0
                        st.metric("Utilization", f"{utilization:.1%}")

                    # Show first few rows of data
                    st.markdown("### Simulation Data Preview")
                    df_rows = pd.DataFrame(rows)

                    # Format the dataframe
                    display_cols = ["serial_no", "cp", "cp_lookup", "inter_arrival_time",
                                   "arrival_time", "service_time", "waiting_time",
                                   "turnaround_time", "response_time", "server_time"]

                    # Filter to available columns
                    available_cols = [col for col in display_cols if col in df_rows.columns]
                    df_display = df_rows[available_cols].copy()

                    # Rename columns for display
                    column_names = {
                        "serial_no": "Customer",
                        "cp": "CP",
                        "cp_lookup": "CP Lookup",
                        "inter_arrival_time": "Interarrival",
                        "arrival_time": "Arrival Time",
                        "service_time": "Service Time",
                        "waiting_time": "Wait Time",
                        "turnaround_time": "Turnaround",
                        "response_time": "Response Time",
                        "server_time": "Server"
                    }
                    df_display = df_display.rename(columns=column_names)

                    # Show first 20 rows
                    st.dataframe(
                        df_display.head(20),
                        use_container_width=True,
                        height=300
                    )

                    if len(df_rows) > 20:
                        st.caption(f"Showing first 20 of {len(df_rows)} rows.")

                    # Download button for data
                    csv = df_display.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Simulation Data (CSV)",
                        data=csv,
                        file_name=f"simulation_{sim_model}_c{servers}_n{n_customers}.csv",
                        mime="text/csv",
                        key="download_sim_data"
                    )
                else:
                    st.error("No customers were simulated. Check your distribution parameters.")
            else:
                st.error("Simulation failed. Please check backend connection and parameters.")

# ==============================================
# 📈 VISUALIZATION DASHBOARD - FULL CODE
# Paste this entire block to replace your
# "elif page == '📈 Visualization Dashboard':" section
# ==============================================

elif page == "📈 Visualization Dashboard":
    st.markdown('<div class="sub-header">Visualization Dashboard</div>', unsafe_allow_html=True)

    if 'sim_data' not in st.session_state or st.session_state.sim_data is None:
        st.warning("⚠️ No simulation data found. Please run a simulation first from the 'Monte Carlo Simulation' page.")
        if st.button("Go to Simulation Page", key="go_to_sim_button"):
            st.rerun()
    else:
        data   = st.session_state.sim_data
        params = st.session_state.get("sim_params", {})

        # ── Simulation Summary ──────────────────────────────────────────────
        rows       = data["rows"]
        n_customers = len(rows)

        st.markdown("### Simulation Summary")
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        with col_sum1:
            st.metric("Model",     params.get('model', 'N/A'))
            st.metric("Customers", n_customers)
        with col_sum2:
            st.metric("Servers",      params.get('servers', 'N/A'))
            st.metric("Arrival Dist", params.get('arrival_dist', 'N/A'))
        with col_sum3:
            st.metric("Service Dist", params.get('service_dist', 'N/A'))
            st.metric("Seed",         params.get('seed', 'N/A'))

        # ── Tabs ────────────────────────────────────────────────────────────
        tabs = st.tabs([
            "📊 Performance Metrics",
            "📈 Time Series",
            "👥 Server Analysis",
            "📅 Gantt Chart",
            "📋 Complete Data"
        ])

        # ====================================================================
        # TAB 1 – PERFORMANCE METRICS
        # ====================================================================
        with tabs[0]:
            if rows and len(rows) > 0:
                col1, col2 = st.columns(2)

                with col1:
                    wait_times = [float(r.get('waiting_time', 0)) for r in rows]
                    if wait_times:
                        fig1 = px.histogram(
                            x=wait_times,
                            nbins=min(30, len(wait_times)),
                            title="Wait Time Distribution",
                            labels={"x": "Wait Time", "y": "Count"},
                            color_discrete_sequence=['#636EFA']
                        )
                        avg_wait = float(np.mean(wait_times))
                        fig1.add_vline(x=avg_wait, line_dash="dash",
                                       line_color="red",
                                       annotation_text=f"Mean: {avg_wait:.3f}")
                        fig1.update_layout(showlegend=False)
                        st.plotly_chart(fig1, use_container_width=True)
                    else:
                        st.info("No wait time data available")

                    turnaround_times = [float(r.get('turnaround_time', 0)) for r in rows]
                    if turnaround_times:
                        fig3 = px.histogram(
                            x=turnaround_times,
                            nbins=min(30, len(turnaround_times)),
                            title="Turnaround Time Distribution",
                            labels={"x": "Turnaround Time", "y": "Count"},
                            color_discrete_sequence=['#00CC96']
                        )
                        avg_turnaround = float(np.mean(turnaround_times))
                        fig3.add_vline(x=avg_turnaround, line_dash="dash",
                                       line_color="red",
                                       annotation_text=f"Mean: {avg_turnaround:.3f}")
                        fig3.update_layout(showlegend=False)
                        st.plotly_chart(fig3, use_container_width=True)

                with col2:
                    wait_times   = [float(r.get('waiting_time', 0))  for r in rows]
                    service_times = [float(r.get('service_time', 0)) for r in rows]
                    if wait_times and service_times:
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(
                            x=wait_times, y=service_times,
                            mode='markers', name='Data Points',
                            marker=dict(size=5, opacity=0.6, color='#636EFA')
                        ))
                        fig2.update_layout(
                            title="Wait Time vs Service Time",
                            xaxis_title="Wait Time",
                            yaxis_title="Service Time"
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.info("No data available")

                    times_data = {}
                    wt  = [float(r.get('waiting_time', 0))    for r in rows if float(r.get('waiting_time', 0)) > 0]
                    tat = [float(r.get('turnaround_time', 0)) for r in rows if float(r.get('turnaround_time', 0)) > 0]
                    rt  = [float(r.get('response_time', 0))   for r in rows if float(r.get('response_time', 0)) > 0]
                    if wt:  times_data["Wait Times"]       = wt
                    if tat: times_data["Turnaround Times"] = tat
                    if rt:  times_data["Response Times"]   = rt

                    if times_data:
                        df_list = [{"Metric": k, "Time": v}
                                   for k, vals in times_data.items() for v in vals]
                        times_df = pd.DataFrame(df_list)
                        fig4 = px.box(times_df, x="Metric", y="Time",
                                      title="Distribution of Time Metrics")
                        st.plotly_chart(fig4, use_container_width=True)
                    else:
                        st.info("No time metrics data available")
            else:
                st.info("No simulation data available")

        # ====================================================================
        # TAB 2 – TIME SERIES
        # ====================================================================
        with tabs[1]:
            if rows and len(rows) > 0:
                df_ts = pd.DataFrame({
                    "Customer":       [int(r.get('serial_no', i+1)) for i, r in enumerate(rows)],
                    "Arrival Time":   [float(r.get('arrival_time', 0))   for r in rows],
                    "Wait Time":      [float(r.get('waiting_time', 0))   for r in rows],
                    "Turnaround Time":[float(r.get('turnaround_time', 0))for r in rows],
                    "Response Time":  [float(r.get('response_time', 0))  for r in rows],
                    "Service Time":   [float(r.get('service_time', 0))   for r in rows],
                })

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_ts["Customer"], y=df_ts["Wait Time"],
                    mode="lines+markers", name="Wait Time",
                    line=dict(color="red",   width=2), marker=dict(size=4)))
                fig.add_trace(go.Scatter(x=df_ts["Customer"], y=df_ts["Turnaround Time"],
                    mode="lines+markers", name="Turnaround Time",
                    line=dict(color="blue",  width=2), marker=dict(size=4)))
                fig.add_trace(go.Scatter(x=df_ts["Customer"], y=df_ts["Response Time"],
                    mode="lines+markers", name="Response Time",
                    line=dict(color="green", width=2), marker=dict(size=4)))
                fig.update_layout(
                    title="Time Metrics Progression",
                    xaxis_title="Customer Number",
                    yaxis_title="Time",
                    hovermode="x unified", height=500
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Moving Averages")
                window_size = st.slider("Moving Average Window",
                                        min_value=2, max_value=50, value=10,
                                        key="ma_window")
                df_ts["Wait_MA"]       = df_ts["Wait Time"].rolling(window_size, min_periods=1).mean()
                df_ts["Turnaround_MA"] = df_ts["Turnaround Time"].rolling(window_size, min_periods=1).mean()
                df_ts["Service_MA"]    = df_ts["Service Time"].rolling(window_size, min_periods=1).mean()

                fig_ma = go.Figure()
                fig_ma.add_trace(go.Scatter(x=df_ts["Customer"], y=df_ts["Wait_MA"],
                    mode="lines", name=f"Wait Time MA (w={window_size})",
                    line=dict(color="red",   width=3)))
                fig_ma.add_trace(go.Scatter(x=df_ts["Customer"], y=df_ts["Turnaround_MA"],
                    mode="lines", name=f"Turnaround MA (w={window_size})",
                    line=dict(color="blue",  width=3)))
                fig_ma.add_trace(go.Scatter(x=df_ts["Customer"], y=df_ts["Service_MA"],
                    mode="lines", name=f"Service MA (w={window_size})",
                    line=dict(color="green", width=3)))
                fig_ma.update_layout(
                    title=f"Moving Averages (Window = {window_size})",
                    xaxis_title="Customer Number", yaxis_title="Time", height=400
                )
                st.plotly_chart(fig_ma, use_container_width=True)
            else:
                st.info("No time series data available")

        # ====================================================================
        # TAB 3 – SERVER ANALYSIS
        # ====================================================================
        with tabs[2]:
            if rows and len(rows) > 0:
                server_stats  = []
                servers_count = int(params.get('servers', 1))
                max_time      = max([float(r.get('arrival_time', 0)) for r in rows]) if rows else 1

                for server_id in range(1, servers_count + 1):
                    server_rows = [r for r in rows if r.get('server_time') == f"Server {server_id}"]
                    if server_rows:
                        total_busy  = sum([float(r.get('service_time', 0)) for r in server_rows])
                        utilization = total_busy / max_time if max_time > 0 else 0
                        server_stats.append({
                            "Server":          f"Server {server_id}",
                            "Busy Time":        total_busy,
                            "Utilization":      utilization,
                            "Customers Served": len(server_rows),
                            "Avg Service Time": total_busy / len(server_rows) if server_rows else 0
                        })

                if server_stats:
                    server_df = pd.DataFrame(server_stats)
                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        fig_u = px.bar(server_df, x="Server", y="Utilization",
                                       title="Server Utilization",
                                       color="Utilization",
                                       color_continuous_scale="Viridis")
                        fig_u.update_layout(yaxis_tickformat=".1%")
                        st.plotly_chart(fig_u, use_container_width=True)
                    with col_s2:
                        fig_p = px.pie(server_df, names="Server",
                                       values="Customers Served",
                                       title="Customers Served per Server",
                                       hole=0.3)
                        st.plotly_chart(fig_p, use_container_width=True)

                    st.markdown("#### Server Statistics")
                    st.dataframe(
                        server_df.style.format({
                            "Busy Time":        "{:.2f}",
                            "Utilization":      "{:.1%}",
                            "Customers Served": "{:.0f}",
                            "Avg Service Time": "{:.3f}"
                        }),
                        use_container_width=True
                    )
            else:
                st.info("No server analysis data available")

        # ====================================================================
        # TAB 4 – GANTT CHART  (image-style: flat bars, rich colors, no calculations)
        # ====================================================================
        with tabs[3]:
            # Bold heading exactly like image
            st.markdown(
                "<p style='font-size:1.25rem; font-weight:900; "
                "font-family:Arial Black,sans-serif; letter-spacing:1px; "
                "margin-bottom:12px;'>GANTT CHARTS:</p>",
                unsafe_allow_html=True
            )

            if data.get("gantt") and len(data["gantt"]) > 0:
                gantt_df = pd.DataFrame(data["gantt"])

                # Type conversions
                gantt_df['server_id']   = gantt_df['server_id'].astype(int)
                gantt_df['customer_id'] = gantt_df['customer_id'].astype(int)
                gantt_df['start']       = gantt_df['start'].astype(float)
                gantt_df['end']         = gantt_df['end'].astype(float)
                gantt_df['duration']    = gantt_df['end'] - gantt_df['start']
                gantt_df = gantt_df.sort_values(['server_id', 'start'])

                unique_servers = sorted(gantt_df['server_id'].unique())

                # ── Controls ────────────────────────────────────────────────
                col1, col2 = st.columns(2)
                with col1:
                    all_customers   = sorted(gantt_df['customer_id'].unique())
                    total_customers = len(all_customers)
                    if total_customers <= 20:
                        display_limit = total_customers
                        st.info(f"Showing all {total_customers} customers")
                    else:
                        display_limit = st.slider(
                            "Number of customers to display",
                            min_value=5,
                            max_value=min(50, total_customers),
                            value=min(20, total_customers),
                            step=1, key="gantt_customers"
                        )
                with col2:
                    selected_servers = st.multiselect(
                        "Select Servers",
                        options=unique_servers,
                        default=unique_servers,
                        format_func=lambda x: f"Server {x}"
                    )

                # ── Filter ──────────────────────────────────────────────────
                filtered_gantt = gantt_df[gantt_df['server_id'].isin(selected_servers)]
                if len(all_customers) > display_limit:
                    filtered_gantt = filtered_gantt[
                        filtered_gantt['customer_id'].isin(all_customers[:display_limit])
                    ]

                if len(filtered_gantt) == 0:
                    st.warning("No data to display. Select at least one server.")
                else:
                    # ── Color palette – 20 distinct vivid colors like image ──
                    COLORS = [
                        '#CC0000',  # C1  deep red
                        '#FF8C00',  # C2  dark orange
                        '#808000',  # C3  olive
                        '#32CD32',  # C4  lime green
                        '#006400',  # C5  dark green
                        '#008B8B',  # C6  dark teal
                        '#00CED1',  # C7  dark turquoise
                        '#1E90FF',  # C8  dodger blue
                        '#0000CD',  # C9  medium blue
                        '#191970',  # C10 midnight navy
                        '#6A0DAD',  # C11 dark purple
                        '#9400D3',  # C12 violet
                        '#DA70D6',  # C13 orchid
                        '#FF69B4',  # C14 hot pink
                        '#FF1493',  # C15 deep pink
                        '#DC143C',  # C16 crimson
                        '#FF4500',  # C17 orange-red
                        '#FFD700',  # C18 gold
                        '#7CFC00',  # C19 lawn green
                        '#00FA9A',  # C20 medium spring green
                    ]

                    # shared x-range for all servers
                    g_min = filtered_gantt['start'].min()
                    g_max = filtered_gantt['end'].max()
                    pad   = (g_max - g_min) * 0.02 if g_max > g_min else 0.5

                    # ── One flat chart per server ────────────────────────────
                    for server in sorted(filtered_gantt['server_id'].unique()):
                        server_data = (
                            filtered_gantt[filtered_gantt['server_id'] == server]
                            .sort_values('start')
                        )
                        if len(server_data) == 0:
                            continue

                        # heading – bold small text like image
                        st.markdown(
                            f"<p style='font-size:0.92rem; font-weight:700; "
                            f"font-family:Arial,sans-serif; margin:16px 0 2px 0;'>"
                            f"Gantt Chart (Server {server})</p>",
                            unsafe_allow_html=True
                        )

                        fig = go.Figure()

                        for _, row in server_data.iterrows():
                            cid   = int(row['customer_id'])
                            color = COLORS[(cid - 1) % len(COLORS)]
                            # white text on dark bars, black on light bars
                            dark  = color in {
                                '#CC0000','#006400','#008B8B','#0000CD',
                                '#191970','#6A0DAD','#9400D3','#DC143C'
                            }
                            txt_color = 'white' if dark else 'black'

                            fig.add_trace(go.Bar(
                                x=[float(row['duration'])],
                                y=[""],
                                base=[float(row['start'])],
                                orientation='h',
                                marker=dict(
                                    color=color,
                                    line=dict(color='white', width=0.6)
                                ),
                                text=f"C{cid}",
                                textposition='inside',
                                insidetextanchor='middle',
                                textfont=dict(size=10, color=txt_color,
                                              family='Arial'),
                                hovertemplate=(
                                    f"<b>Customer {cid}</b><br>"
                                    f"Start: {row['start']:.3f}<br>"
                                    f"End: {row['end']:.3f}<br>"
                                    f"Duration: {row['duration']:.3f}<br>"
                                    "<extra></extra>"
                                ),
                                showlegend=False
                            ))

                        fig.update_layout(
                            xaxis=dict(
                                range=[g_min - pad, g_max + pad],
                                showgrid=True,
                                gridcolor='rgba(180,180,180,0.35)',
                                zeroline=False,
                                showline=True,
                                linewidth=1,
                                linecolor='#bbbbbb',
                                tickfont=dict(size=9),
                                title=None,
                            ),
                            yaxis=dict(
                                showticklabels=False,
                                showgrid=False,
                                zeroline=False,
                                showline=False,
                            ),
                            barmode='stack',
                            height=72,           # flat strip like image
                            margin=dict(l=0, r=0, t=0, b=16),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            showlegend=False,
                        )

                        st.plotly_chart(fig, use_container_width=True,
                                        config={'displayModeBar': False})

                    # ── Colour legend (collapsible) ──────────────────────────
                    with st.expander("🎨 Customer Colour Legend"):
                        all_shown = sorted(filtered_gantt['customer_id'].unique())
                        leg_cols  = st.columns(6)
                        for i, cid in enumerate(all_shown):
                            color = COLORS[(int(cid) - 1) % len(COLORS)]
                            leg_cols[i % 6].markdown(
                                f"""<div style="display:flex;align-items:center;margin:3px;">
                                      <div style="width:18px;height:18px;background:{color};
                                                  border:1px solid #666;margin-right:5px;
                                                  border-radius:2px;"></div>
                                      <span style="font-size:0.8rem;">C{int(cid)}</span>
                                    </div>""",
                                unsafe_allow_html=True
                            )

                    # ── Data table (collapsible) ─────────────────────────────
                    with st.expander("📋 View Gantt Data Table"):
                        disp = filtered_gantt[
                            ['server_id','customer_id','start','end','duration']
                        ].copy()
                        disp.columns = ['Server','Customer','Start','End','Duration']
                        disp = disp.sort_values(['Server','Start'])
                        st.dataframe(disp, use_container_width=True)
                        csv = disp.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Gantt Data (CSV)",
                            data=csv, file_name="gantt_data.csv",
                            mime="text/csv"
                        )
            else:
                st.info("No Gantt chart data available. Please run a simulation first.")

        # ====================================================================
        # TAB 5 – COMPLETE DATA
        # ====================================================================
        with tabs[4]:
            if rows and len(rows) > 0:
                clean_rows = []
                for r in rows:
                    clean_row = {}
                    for key, value in r.items():
                        clean_row[key] = value.item() if hasattr(value, 'item') else value
                    clean_rows.append(clean_row)

                df_complete = pd.DataFrame(clean_rows)
                st.info(f"Dataset contains {len(df_complete)} rows and "
                        f"{len(df_complete.columns)} columns")

                page_size   = st.selectbox("Rows per page",
                                           [20, 50, 100, 200, 500],
                                           index=0, key="page_size_complete")
                total_pages = (len(df_complete) + page_size - 1) // page_size

                if 'data_page_complete' not in st.session_state:
                    st.session_state.data_page_complete = 1

                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    if (st.button("◀ Prev", key="prev_complete")
                            and st.session_state.data_page_complete > 1):
                        st.session_state.data_page_complete -= 1
                        st.rerun()
                with col2:
                    st.session_state.data_page_complete = st.number_input(
                        "Page", min_value=1, max_value=total_pages,
                        value=st.session_state.data_page_complete,
                        key="page_input_complete"
                    )
                with col3:
                    if (st.button("Next ▶", key="next_complete")
                            and st.session_state.data_page_complete < total_pages):
                        st.session_state.data_page_complete += 1
                        st.rerun()

                start = (st.session_state.data_page_complete - 1) * page_size
                end   = min(start + page_size, len(df_complete))

                st.dataframe(df_complete.iloc[start:end],
                             use_container_width=True, height=500)
                st.caption(f"Showing rows {start+1} to {end} of {len(df_complete)}")

                csv = df_complete.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download All Data (CSV)",
                    data=csv,
                    file_name=f"simulation_data_{params.get('model', 'unknown')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No simulation data available")

#Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6B7280; font-size: 0.9rem;'>
    Queue Simulator Frontend | Powered by Streamlit | Backend: FastAPI<br>
    <small>Note: ∞ indicates infinite values (unstable system)</small>
    </div>
    """,
    unsafe_allow_html=True
)