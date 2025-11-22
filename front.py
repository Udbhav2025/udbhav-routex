import streamlit as st
import pandas as pd

def main():
    st.set_page_config(page_title="Supply Chain Oracle", page_icon="🚚", layout="wide")
    
    st.title("🚚 Supply Chain Oracle")
    st.subheader("AI-Powered Shipment Delay Predictor")
    
    # Sidebar for input
    with st.sidebar:
        st.header("📦 Shipment Details")
        origin = st.selectbox("Origin", ["CHENNAI", "MUMBAI", "DELHI", "HYD", "BGLR"])
        destination = st.selectbox("Destination", ["BOS", "SFO", "MSP", "CLT", "HOU", "LAS", "ORD", "PHX"])
        carrier = st.selectbox("Carrier", ["A", "B", "C", "D"])
        route_type = st.selectbox("Route Type", ["road", "air", "rail", "sea"])
        priority = st.selectbox("Priority", ["low", "medium", "high", "express"])
        
        col1, col2 = st.columns(2)
        with col1:
            distance_km = st.number_input("Distance (km)", min_value=0.0, value=1000.0)
            hub_count = st.number_input("Hub Count", min_value=0, max_value=10, value=1)
            weekend = st.checkbox("Weekend Shipment")
        with col2:
            weather_risk = st.slider("Weather Risk", 0.0, 1.0, 0.3)
            traffic_risk = st.slider("Traffic Risk", 0.0, 1.0, 0.3)
            customs_risk = st.slider("Customs Risk", 0.0, 1.0, 0.3)
        
        holiday = st.checkbox("Holiday Shipment")
        est_hours = st.number_input("Estimated Hours", min_value=0.0, value=24.0)
    
    # Create shipment data
    shipment_data = {
        'origin': origin, 'destination': destination, 'carrier': carrier,
        'route_type': route_type, 'priority': priority, 'distance_km': distance_km,
        'hub_count': hub_count, 'weather_risk': weather_risk, 'traffic_risk': traffic_risk,
        'customs_risk': customs_risk, 'weekend_shipment': int(weekend),
        'holiday_shipment': int(holiday), 'estimated_transit_hours': est_hours
    }
    
    # Predict button
    if st.button("🔮 Predict Shipment Outcome", type="primary"):
        with st.spinner("Analyzing shipment risks..."):
            prediction = oracle.predict_shipment(shipment_data)
            explanations = oracle.generate_explanation(prediction, shipment_data)
            recommendations = oracle.generate_recommendations(prediction, shipment_data)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Prediction Results")
            
            # Status card
            if prediction['will_be_delayed']:
                st.error(f"🚨 DELAY PREDICTED: {prediction['delay_probability']*100:.1f}%")
            else:
                st.success(f"✅ ON-TIME DELIVERY: {(1-prediction['delay_probability'])*100:.1f}%")
            
            # Metrics
            st.metric("Predicted Delay Hours", f"{prediction['predicted_delay_hours']}h")
            st.metric("Estimated Arrival", f"{prediction['estimated_arrival_hours']}h")
            
            if prediction['is_high_risk']:
                st.error(f"🔥 HIGH RISK: {prediction['high_risk_probability']*100:.1f}%")
            else:
                st.info(f"🟢 LOW RISK: {(1-prediction['high_risk_probability'])*100:.1f}%")
        
        with col2:
            st.subheader("🔍 Risk Analysis")
            for explanation in explanations:
                if "HIGH PROBABILITY" in explanation or "HIGH-RISK" in explanation:
                    st.error(explanation)
                elif "ON-TIME" in explanation:
                    st.success(explanation)
                else:
                    st.info(explanation)
        
        # Recommendations
        st.subheader("💡 Recommended Actions")
        for rec in recommendations:
            if "ALTERNATIVES" in rec:
                st.warning(rec)
            else:
                st.info(rec)
        
        # Risk factors visualization
        st.subheader("📈 Risk Factors Breakdown")
        risk_data = {
            'Weather': shipment_data['weather_risk'],
            'Traffic': shipment_data['traffic_risk'], 
            'Customs': shipment_data['customs_risk'],
            'Total': shipment_data['weather_risk'] + shipment_data['traffic_risk'] + shipment_data['customs_risk']
        }
        st.bar_chart(risk_data)

if __name__ == "__main__":
    main()
