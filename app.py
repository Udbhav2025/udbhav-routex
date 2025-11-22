import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from ml_model import MLModel
from oracle_engine import SupplyChainOracle
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Shipment Delay Predictor", 
    page_icon="🚚", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .delay-card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .time-display {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'oracle_initialized' not in st.session_state:
    st.session_state.oracle_initialized = False
    st.session_state.ml_model = None
    st.session_state.oracle = None
    st.session_state.last_submission_time = None

def initialize_system():
    """Initialize the ML models and oracle system"""
    try:
        with st.spinner("🔄 Loading Hyderabad shipment data..."):
            ml_model = MLModel()
            df = ml_model.load_and_preprocess_data('hyderabad_shipments_500_balanced.csv')
            
            with st.spinner("🤖 Training AI models..."):
                ml_model.train_models(df)
            
            oracle = SupplyChainOracle(ml_model)
            
            st.session_state.ml_model = ml_model
            st.session_state.oracle = oracle
            st.session_state.oracle_initialized = True
            st.session_state.historical_data = df
            
            return True
    except Exception as e:
        st.error(f"❌ Error initializing system: {str(e)}")
        return False

def main():
    # Header - UPDATED TITLE
    st.markdown('<h1 class="main-header">🚚 Shipment Delay Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center;">AI-Powered Delivery Time Prediction for Hyderabad</h3>', unsafe_allow_html=True)
    
    # Real-time clock
    current_time = datetime.now()
    st.markdown(f"""
    <div class="time-display">
        <h4>🕐 Current Hyderabad Time</h4>
        <h3>{current_time.strftime('%Y-%m-%d %I:%M:%S %p')}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system if not done
    if not st.session_state.oracle_initialized:
        st.info("🔧 System initialization required. Click below to start.")
        if st.button("🚀 Initialize Prediction System", type="primary"):
            if initialize_system():
                st.success("✅ System initialized successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to initialize system. Please check your data file.")
        return
    
    # Sidebar Navigation
    with st.sidebar:
        st.header("📊 Navigation")
        page = st.radio("Go to:", ["🏠 Delivery Predictor", "📈 Analytics Dashboard"])
        
        st.header("📦 Quick Stats")
        if st.session_state.oracle_initialized:
            insights = st.session_state.oracle.analyze_historical_shipments(
                st.session_state.historical_data
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Shipments", insights['total_shipments'])
                st.metric("Delay Rate", f"{insights['delay_rate']*100:.1f}%")
            with col2:
                st.metric("Avg Delay", f"{insights['avg_delay_hours']:.1f}h")
                st.metric("Max Delay", f"{insights['max_delay_hours']:.1f}h")
    
    # Main content based on page selection
    if page == "🏠 Delivery Predictor":
        render_delivery_predictor(current_time)
    else:
        render_analytics_dashboard()

def render_delivery_predictor(current_time):
    """Render the delivery prediction interface"""
    st.header("🔮 Delivery Time Predictor")
    
    with st.form("delivery_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📍 Route Details")
            origin = st.selectbox("Pickup Location", 
                                ["Secunderabad", "Banjara Hills", "Kukatpally", "Uppal", 
                                 "Gachibowli", "Madhapur", "Shamshabad", "HITEC City", "Kondapur"])
            destination = st.selectbox("Delivery Location", 
                                     ["Secunderabad", "Banjara Hills", "Kukatpally", "Uppal", 
                                      "Gachibowli", "Madhapur", "Shamshabad", "HITEC City", "Kondapur"])
            
            # Auto-fill distance
            if origin and destination:
                distance = st.session_state.ml_model.get_distance(origin, destination)
                st.info(f"📍 Distance: {distance} km (auto-calculated)")
            else:
                distance = 15.0
            
            carrier = st.selectbox("Delivery Partner", ["A", "B", "C", "D"])
            priority = st.selectbox("Delivery Priority", ["express", "high", "medium", "low"])
            
        with col2:
            st.subheader("📊 Delivery Parameters")
            
            # Use auto-calculated distance (hidden from user)
            distance_km = distance
            
            hub_count = st.slider("Number of Transfer Points", 0, 5, 1)
            est_hours = st.slider("Estimated Transit Hours", 0.0, 30.0, 10.0, 0.1)
            
            st.subheader("📝 Additional Factors")
            weather_risk = st.slider("Weather Conditions", 0.0, 1.0, 0.3, 0.01)
            traffic_risk = st.slider("Traffic Conditions", 0.0, 1.0, 0.4, 0.01)
            customs_risk = st.slider("Documentation Factors", 0.0, 1.0, 0.2, 0.01)
            
            weekend = st.checkbox("Weekend Delivery")
            holiday = st.checkbox("Holiday Delivery")
        
        # Submit button
        submitted = st.form_submit_button("🔮 Predict Delivery Outcome", type="primary")
    
    if submitted:
        # Store submission time
        st.session_state.last_submission_time = current_time
        
        # Check if origin and destination are same
        if origin == destination:
            st.error("🚚 DELIVERY ALREADY ARRIVED")
            st.info("Origin and destination are the same location. No delivery needed!")
            return
        
        # Create shipment data
        shipment_data = {
            'origin': origin, 'destination': destination, 'carrier': carrier,
            'priority': priority, 'distance_km': distance_km, 'hub_count': hub_count,
            'weather_risk': weather_risk, 'traffic_risk': traffic_risk, 'customs_risk': customs_risk,
            'weekend_shipment': int(weekend), 'holiday_shipment': int(holiday), 
            'estimated_transit_hours': est_hours
        }
        
        with st.spinner("🤖 Analyzing delivery..."):
            try:
                prediction = st.session_state.oracle.predict_shipment(shipment_data)
                time_info = st.session_state.oracle.calculate_delivery_times(prediction, current_time)
                explanations = st.session_state.oracle.generate_explanation(prediction, shipment_data, time_info)
                recommendations = st.session_state.oracle.generate_recommendations(prediction, shipment_data)
                
                # Display results
                st.success("✅ Analysis Complete!")
                
                # Results in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Prediction Results")
                    render_prediction_cards(prediction, time_info)
                
                with col2:
                    st.subheader("📋 Delivery Details")
                    render_explanations(explanations)
                
                # Recommendations
                st.subheader("💡 Recommendations")
                render_recommendations(recommendations)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")

def render_prediction_cards(prediction, time_info):
    """Render prediction results as cards"""
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction['will_be_delayed']:
            st.error("📦 DELAY PREDICTED")
            st.metric("Delay Probability", f"{prediction['delay_probability']*100:.1f}%")
            st.metric("Expected Delay", f"{prediction['predicted_delay_hours']}h")
        else:
            st.success("✅ ON-TIME DELIVERY")
            st.metric("On-time Probability", f"{prediction['on_time_probability']*100:.1f}%")
            st.metric("Expected Delay", "0h")
    
    with col2:
        st.metric("Original Estimate", f"{prediction['original_estimate_hours']}h")
        st.metric("Total Duration", f"{prediction['estimated_arrival_hours']}h")
        
        # Time display
        st.info(f"🕐 Delivery by: {time_info['delivery_time'].strftime('%I:%M %p')}")

def render_explanations(explanations):
    """Render explanations"""
    for explanation in explanations:
        if "DELIVERY ALREADY ARRIVED" in explanation:
            st.error(explanation)
        elif "DELAY PREDICTED" in explanation:
            st.error(explanation)
        elif "ON-TIME DELIVERY" in explanation:
            st.success(explanation)
        elif "Current time" in explanation or "Predicted delivery" in explanation:
            st.info(explanation)
        else:
            st.write(explanation)

def render_recommendations(recommendations):
    """Render recommendations"""
    for rec in recommendations:
        if "RECOMMENDED ACTIONS" in rec:
            st.warning(rec)
        elif "CURRENT PLAN" in rec:
            st.success(rec)
        else:
            st.info(rec)

def render_analytics_dashboard():
    """Render analytics dashboard"""
    st.header("📈 Delivery Analytics Dashboard")
    
    if not st.session_state.oracle_initialized:
        st.warning("Please initialize the system first.")
        return
    
    df = st.session_state.historical_data
    insights = st.session_state.oracle.analyze_historical_shipments(df)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Deliveries", insights['total_shipments'])
    with col2:
        st.metric("Delay Rate", f"{insights['delay_rate']*100:.1f}%")
    with col3:
        st.metric("Avg Delay", f"{insights['avg_delay_hours']:.1f}h")
    with col4:
        st.metric("Max Delay", f"{insights['max_delay_hours']:.1f}h")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Carrier performance
        st.subheader("🚚 Carrier Performance")
        carrier_perf = insights['carrier_performance']
        fig = px.bar(carrier_perf, y='is_delayed', 
                    title='Delay Rate by Carrier',
                    labels={'is_delayed': 'Delay Rate', 'carrier': 'Carrier'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Priority performance
        st.subheader("⚡ Priority Level Performance")
        priority_perf = insights['priority_performance']
        fig = px.bar(priority_perf, y='is_delayed',
                    title='Delay Rate by Priority',
                    labels={'is_delayed': 'Delay Rate'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Worst performing routes
    st.subheader("📊 Route Performance")
    st.dataframe(insights['worst_routes'], use_container_width=True)

if __name__ == "__main__":
    main()