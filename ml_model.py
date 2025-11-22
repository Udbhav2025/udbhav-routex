import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class MLModel:
    def __init__(self):
        self.models_trained = False
        self.label_encoders = {}
        self.features = []
        self.scaler = StandardScaler()
        
        # Real distance data for Hyderabad areas (in km)
        self.distance_matrix = {
            'Secunderabad': {'Secunderabad': 0, 'Banjara Hills': 8, 'Kukatpally': 12, 'Uppal': 10, 
                           'Gachibowli': 18, 'Madhapur': 16, 'Shamshabad': 35, 'HITEC City': 15, 'Kondapur': 17},
            'Banjara Hills': {'Secunderabad': 8, 'Banjara Hills': 0, 'Kukatpally': 14, 'Uppal': 12, 
                            'Gachibowli': 12, 'Madhapur': 10, 'Shamshabad': 30, 'HITEC City': 8, 'Kondapur': 10},
            'Kukatpally': {'Secunderabad': 12, 'Banjara Hills': 14, 'Kukatpally': 0, 'Uppal': 15, 
                          'Gachibowli': 20, 'Madhapur': 18, 'Shamshabad': 40, 'HITEC City': 17, 'Kondapur': 19},
            'Uppal': {'Secunderabad': 10, 'Banjara Hills': 12, 'Kukatpally': 15, 'Uppal': 0, 
                     'Gachibowli': 22, 'Madhapur': 20, 'Shamshabad': 38, 'HITEC City': 19, 'Kondapur': 21},
            'Gachibowli': {'Secunderabad': 18, 'Banjara Hills': 12, 'Kukatpally': 20, 'Uppal': 22, 
                          'Gachibowli': 0, 'Madhapur': 4, 'Shamshabad': 25, 'HITEC City': 3, 'Kondapur': 2},
            'Madhapur': {'Secunderabad': 16, 'Banjara Hills': 10, 'Kukatpally': 18, 'Uppal': 20, 
                        'Gachibowli': 4, 'Madhapur': 0, 'Shamshabad': 28, 'HITEC City': 2, 'Kondapur': 3},
            'Shamshabad': {'Secunderabad': 35, 'Banjara Hills': 30, 'Kukatpally': 40, 'Uppal': 38, 
                          'Gachibowli': 25, 'Madhapur': 28, 'Shamshabad': 0, 'HITEC City': 26, 'Kondapur': 24},
            'HITEC City': {'Secunderabad': 15, 'Banjara Hills': 8, 'Kukatpally': 17, 'Uppal': 19, 
                          'Gachibowli': 3, 'Madhapur': 2, 'Shamshabad': 26, 'HITEC City': 0, 'Kondapur': 1},
            'Kondapur': {'Secunderabad': 17, 'Banjara Hills': 10, 'Kukatpally': 19, 'Uppal': 21, 
                        'Gachibowli': 2, 'Madhapur': 3, 'Shamshabad': 24, 'HITEC City': 1, 'Kondapur': 0}
        }
        
    def get_distance(self, origin, destination):
        """Get real distance between Hyderabad areas"""
        if origin in self.distance_matrix and destination in self.distance_matrix[origin]:
            return self.distance_matrix[origin][destination]
        return 15.0  # Default distance
    
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the Hyderabad shipment data"""
        df = pd.read_csv(file_path)
        
        print(f"📊 Dataset loaded: {df.shape[0]} shipments, {df.shape[1]} features")
        print(f"📈 Delay rate: {df['is_delayed'].mean():.2%}")
        
        # Create additional features
        df['actual_transit_hours'] = df['estimated_transit_hours'] + df['delay_hours']
        df['delay_ratio'] = df['delay_hours'] / (df['estimated_transit_hours'] + 0.001)
        df['total_risk'] = df['weather_risk'] + df['traffic_risk'] + df['customs_risk']
        df['weekend_holiday'] = df['weekend_shipment'] | df['holiday_shipment']
        df['risk_intensity'] = df['total_risk'] * df['distance_km']
        df['efficiency_score'] = df['distance_km'] / (df['estimated_transit_hours'] + 0.001)
        
        # Encode categorical variables
        categorical_cols = ['origin', 'destination', 'carrier', 'priority']
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        # Feature selection
        self.features = [
            'origin_encoded', 'destination_encoded', 'carrier_encoded', 'priority_encoded',
            'distance_km', 'hub_count', 'weather_risk', 'traffic_risk', 'customs_risk',
            'weekend_shipment', 'holiday_shipment', 'weekend_holiday', 'estimated_transit_hours',
            'total_risk', 'risk_intensity', 'efficiency_score'
        ]
        
        return df
    
    def train_models(self, df):
        """Train ML models"""
        X = df[self.features]
        y_delay = df['is_delayed']
        y_delay_hours = df['delay_hours']
        
        # Split data
        X_train, X_test, y_delay_train, y_delay_test = train_test_split(
            X, y_delay, test_size=0.2, random_state=42, stratify=y_delay
        )
        
        X_train_dh, X_test_dh, y_delay_hours_train, y_delay_hours_test = train_test_split(
            X, y_delay_hours, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_train_dh_scaled = self.scaler.fit_transform(X_train_dh)
        X_test_dh_scaled = self.scaler.transform(X_test_dh)
        
        # Train Delay Classification Model
        self.delay_model = xgb.XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42
        )
        self.delay_model.fit(X_train_scaled, y_delay_train)
        
        # Train Delay Hours Regression Model
        self.delay_hours_model = xgb.XGBRegressor(
            n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42
        )
        self.delay_hours_model.fit(X_train_dh_scaled, y_delay_hours_train)
        
        # Evaluate models
        delay_accuracy = accuracy_score(y_delay_test, self.delay_model.predict(X_test_scaled))
        dh_mae = mean_absolute_error(y_delay_hours_test, self.delay_hours_model.predict(X_test_dh_scaled))
        
        print(f"✅ Delay Classification Accuracy: {delay_accuracy:.3f}")
        print(f"⏱️ Delay Hours MAE: {dh_mae:.3f} hours")
        
        self.models_trained = True
        return True
    
    def prepare_features(self, shipment_data):
        """Prepare features for prediction"""
        features_df = pd.DataFrame([shipment_data])
        
        # Encode categorical variables
        for col in ['origin', 'destination', 'carrier', 'priority']:
            if col in features_df.columns:
                try:
                    features_df[col + '_encoded'] = self.label_encoders[col].transform(
                        [features_df[col].iloc[0]]
                    )
                except ValueError:
                    features_df[col + '_encoded'] = 0
        
        # Create derived features
        features_df['total_risk'] = (
            features_df['weather_risk'] + 
            features_df['traffic_risk'] + 
            features_df['customs_risk']
        )
        features_df['weekend_holiday'] = (
            features_df['weekend_shipment'].astype(bool) | 
            features_df['holiday_shipment'].astype(bool)
        ).astype(int)
        
        features_df['risk_intensity'] = features_df['total_risk'] * features_df['distance_km']
        features_df['efficiency_score'] = features_df['distance_km'] / (features_df['estimated_transit_hours'] + 0.001)
        
        # Ensure all required features are present
        for feature in self.features:
            if feature not in features_df.columns:
                features_df[feature] = 0
        
        # Scale features
        features_scaled = self.scaler.transform(features_df[self.features])
        return features_scaled
    
    def predict(self, shipment_data):
        """Make predictions for a shipment - UPDATED WITH FLOAT CONVERSION"""
        if not self.models_trained:
            raise Exception("Models not trained yet! Call train_models() first.")
        
        X = self.prepare_features(shipment_data)
        
        # Make predictions
        delay_prob = self.delay_model.predict_proba(X)[0][1]
        delay_pred = self.delay_model.predict(X)[0]
        delay_hours_pred = max(0, self.delay_hours_model.predict(X)[0])
        
        # Convert all values to standard Python types to avoid numpy issues with timedelta
        estimated_arrival = float(shipment_data['estimated_transit_hours']) + float(delay_hours_pred)
        
        return {
            'will_be_delayed': bool(delay_pred),
            'delay_probability': float(round(delay_prob, 3)),  # Convert to float
            'predicted_delay_hours': float(round(delay_hours_pred, 2)),  # Convert to float
            'estimated_arrival_hours': float(round(estimated_arrival, 2)),  # Convert to float
            'original_estimate_hours': float(shipment_data['estimated_transit_hours']),  # Convert to float
            'on_time_probability': float(round(1 - delay_prob, 3))  # Convert to float
        }