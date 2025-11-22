import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SupplyChainOracle:
    def __init__(self, ml_model):
        self.ml_model = ml_model
        
        # Hyderabad areas
        self.hyderabad_areas = [
            'Secunderabad', 'Banjara Hills', 'Kukatpally', 'Uppal', 
            'Gachibowli', 'Madhapur', 'Shamshabad', 'HITEC City', 'Kondapur'
        ]
        
        # Carrier performance
        self.carrier_performance = {
            'A': {'reliability': 0.75, 'speed': 0.8},
            'B': {'reliability': 0.65, 'speed': 0.7},
            'C': {'reliability': 0.80, 'speed': 0.6},
            'D': {'reliability': 0.70, 'speed': 0.9}
        }
    
    def predict_shipment(self, shipment_data):
        """Make prediction for a shipment"""
        return self.ml_model.predict(shipment_data)
    
    def calculate_delivery_times(self, prediction, submission_time):
        """Calculate real delivery times - FIXED VERSION"""
        current_time = submission_time
        
        # Convert numpy types to standard Python float to avoid timedelta issues
        estimated_hours = float(prediction['estimated_arrival_hours'])  # Convert to standard float
        
        # Calculate delivery datetime
        delivery_time = current_time + timedelta(hours=estimated_hours)
        
        return {
            'current_time': current_time,
            'delivery_time': delivery_time,
            'time_until_delivery': estimated_hours * 3600,  # in seconds
            'estimated_hours': estimated_hours
        }
    
    def generate_explanation(self, prediction, shipment_data, time_info):
        """Generate explanation for the prediction"""
        explanations = []
        
        # Check if origin and destination are same
        if shipment_data['origin'] == shipment_data['destination']:
            explanations.append("🚚 DELIVERY ALREADY ARRIVED")
            explanations.append("Origin and destination are the same location")
            return explanations
        
        # Delay prediction only
        if prediction['will_be_delayed']:
            explanations.append("📦 DELAY PREDICTED")
            explanations.append(f"Delay probability: {prediction['delay_probability']*100:.1f}%")
            explanations.append(f"Expected delay: {prediction['predicted_delay_hours']} hours")
            explanations.append(f"Original estimate: {prediction['original_estimate_hours']} hours")
            explanations.append(f"Estimated arrival: {prediction['estimated_arrival_hours']} hours")
        else:
            explanations.append("✅ ON-TIME DELIVERY EXPECTED")
            explanations.append(f"On-time probability: {prediction['on_time_probability']*100:.1f}%")
            explanations.append(f"Estimated arrival: {prediction['estimated_arrival_hours']} hours")
        
        # Route information
        origin = shipment_data['origin']
        destination = shipment_data['destination']
        explanations.append(f"Route: {origin} → {destination}")
        explanations.append(f"Distance: {shipment_data['distance_km']} km")
        explanations.append(f"Transfer points: {shipment_data['hub_count']}")
        
        # Time information
        explanations.append(f"Current time (Hyderabad): {time_info['current_time'].strftime('%Y-%m-%d %I:%M %p')}")
        explanations.append(f"Predicted delivery: {time_info['delivery_time'].strftime('%Y-%m-%d %I:%M %p')}")
        
        return explanations
    
    def generate_recommendations(self, prediction, shipment_data):
        """Generate recommendations"""
        recommendations = []
        current_carrier = shipment_data['carrier']
        current_priority = shipment_data['priority']
        
        if prediction['will_be_delayed']:
            recommendations.append("🚀 RECOMMENDED ACTIONS:")
            
            # Priority upgrade suggestion
            if current_priority != 'express':
                priority_upgrade = {
                    'low': 'medium', 'medium': 'high', 'high': 'express'
                }
                if current_priority in priority_upgrade:
                    recommendations.append(f"• Upgrade priority from {current_priority} to {priority_upgrade[current_priority]}")
            
            # Carrier recommendations
            reliable_carriers = []
            for carrier, perf in self.carrier_performance.items():
                if carrier != current_carrier and perf['reliability'] > 0.75:
                    reliable_carriers.append((carrier, perf['reliability']))
            
            if reliable_carriers:
                best_carrier = max(reliable_carriers, key=lambda x: x[1])
                recommendations.append(f"• Switch to Carrier {best_carrier[0]} ({best_carrier[1]*100:.0f}% reliability)")
            
            # Operational recommendations
            if shipment_data['hub_count'] > 1:
                recommendations.append("• Request direct routing to reduce transfers")
            
            if shipment_data.get('weekend_shipment', 0) or shipment_data.get('holiday_shipment', 0):
                recommendations.append("• Consider weekday delivery for better reliability")
            
            if shipment_data['distance_km'] > 25:
                recommendations.append("• Consider splitting into multiple deliveries")
        
        else:
            recommendations.append("🎉 CURRENT PLAN LOOKS GOOD")
            recommendations.append("• High probability of on-time delivery")
        
        return recommendations
    
    def analyze_historical_shipments(self, df):
        """Analyze historical shipment data"""
        delayed_shipments = df[df['is_delayed'] == 1]
        
        insights = {
            'total_shipments': len(df),
            'delayed_shipments': len(delayed_shipments),
            'delay_rate': len(delayed_shipments) / len(df),
            'avg_delay_hours': delayed_shipments['delay_hours'].mean(),
            'max_delay_hours': df['delay_hours'].max(),
            
            # Performance by carrier
            'carrier_performance': df.groupby('carrier').agg({
                'is_delayed': 'mean',
                'delay_hours': 'mean'
            }).round(3),
            
            # Performance by priority
            'priority_performance': df.groupby('priority').agg({
                'is_delayed': 'mean',
                'delay_hours': 'mean'
            }).round(3),
            
            # Worst performing routes
            'worst_routes': df.groupby(['origin', 'destination']).agg({
                'is_delayed': 'mean',
                'delay_hours': 'mean'
            }).round(3).sort_values('delay_hours', ascending=False).head(10)
        }
        
        return insights