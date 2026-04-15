"""
Integration Example: How the Deep Learning Model Works with the Pipeline
This demonstrates the seamless integration of DL with the existing ML pipeline
"""

# ============================================================================
# EXAMPLE 1: Using the Neural Network Model Standalone
# ============================================================================

from models.neural_network_model import ChurnPredictor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv("data/dataset_without_preprocessing.csv")

# Prepare features and target
features = ["total_orders", "festival_orders", "inactive_days", 
            "total_spent", "avg_order_value", "app_sessions_per_month"]
X = df[features]
y = df["Churn"] if "Churn" in df.columns else (df["festival_orders"] / df["total_orders"] > 0.6).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
predictor = ChurnPredictor(input_dim=len(features))
predictor.build_model()
history = predictor.train(X_train, y_train, X_test, y_test, epochs=50, verbose=1)

# Make predictions
test_customer = X_test.iloc[0:1]
prediction = predictor.predict(test_customer)           # Binary: 0 or 1
probability = predictor.predict_proba(test_customer)    # Confidence: 0.0-1.0

print(f"Prediction: {prediction[0]}")
print(f"Churn Probability: {probability[0]:.2%}")

# Evaluate model
metrics = predictor.evaluate(X_test, y_test)
print(f"Model Accuracy: {metrics['Accuracy']:.4f}")
print(f"Model AUC: {metrics['AUC']:.4f}")

# Save model for later use
predictor.save_model('saved_models/my_neural_net', 'saved_models/my_scaler.pkl')


# ============================================================================
# EXAMPLE 2: Loading a Saved Model
# ============================================================================

saved_predictor = ChurnPredictor(input_dim=len(features))
saved_predictor.load_model('saved_models/my_neural_net', 'saved_models/my_scaler.pkl')

new_predictions = saved_predictor.predict_proba(X_test)


# ============================================================================
# EXAMPLE 3: Integrated with Random Forest for Comparison
# ============================================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Train both models on same data
nn_predictor = ChurnPredictor(input_dim=len(features))
nn_predictor.build_model()
nn_predictor.train(X_train, y_train, epochs=50, verbose=0)

# Random Forest baseline
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_scaled, y_train)

# Compare predictions
nn_pred = nn_predictor.predict_proba(X_test)
X_test_scaled = scaler.transform(X_test)
rf_pred = rf_model.predict_proba(X_test_scaled)[:, 1]

# Ensemble prediction (average both models)
ensemble_pred = (nn_pred + rf_pred) / 2

comparison_df = pd.DataFrame({
    'Neural_Network': nn_pred,
    'Random_Forest': rf_pred,
    'Ensemble': ensemble_pred,
    'Actual': y_test.values
})

print(comparison_df.head(10))


# ============================================================================
# EXAMPLE 4: Batch Prediction for Multiple Customers
# ============================================================================

# Create a batch of new customers
new_customers = pd.DataFrame([
    {"total_orders": 50, "festival_orders": 10, "inactive_days": 30,
     "total_spent": 5000, "avg_order_value": 200, "app_sessions_per_month": 20},
    {"total_orders": 200, "festival_orders": 5, "inactive_days": 5,
     "total_spent": 40000, "avg_order_value": 800, "app_sessions_per_month": 60},
    {"total_orders": 15, "festival_orders": 14, "inactive_days": 200,
     "total_spent": 1000, "avg_order_value": 100, "app_sessions_per_month": 2},
])

# Get predictions for all
batch_probabilities = predictor.predict_proba(new_customers)
batch_predictions = predictor.predict(new_customers)

for idx, (prob, pred) in enumerate(zip(batch_probabilities, batch_predictions)):
    status = "⚠️ CHURN RISK" if pred == 1 else "✅ RETAIN"
    print(f"Customer {idx+1}: {prob:.2%} churn risk - {status}")


# ============================================================================
# EXAMPLE 5: Custom Training with Different Architectures
# ============================================================================

# You can easily extend the ChurnPredictor class for custom architectures
from tensorflow import keras
from tensorflow.keras import layers

class CustomChurnPredictor(ChurnPredictor):
    def build_model(self):
        """Override with custom architecture"""
        self.model = keras.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
        )

# Use custom predictor
custom_predictor = CustomChurnPredictor(input_dim=len(features))
custom_predictor.train(X_train, y_train, epochs=50, verbose=0)


# ============================================================================
# EXAMPLE 6: Integration with Streamlit (see app.py for full implementation)
# ============================================================================

"""
# In app.py:

import streamlit as st
from models.neural_network_model import ChurnPredictor
from sklearn.ensemble import RandomForestClassifier

# Load pre-trained models
nn_predictor = ChurnPredictor(input_dim=6)
nn_predictor.load_model()

# Get user input
customer_features = st.sidebar.slider('Features...', ...)

# Make predictions
nn_prob = nn_predictor.predict_proba(customer_features)
rf_prob = rf_model.predict_proba(customer_features)

# Display results
st.metric('Neural Network Risk', f'{nn_prob:.1%}')
st.metric('Random Forest Risk', f'{rf_prob:.1%}')
"""


# ============================================================================
# EXAMPLE 7: Performance Monitoring and Logging
# ============================================================================

import matplotlib.pyplot as plt

# Plot training history
history = predictor.train(X_train, y_train, X_test, y_test, epochs=50, verbose=0)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss Over Time')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy Over Time')

plt.tight_layout()
plt.savefig('model_performance.png')


# ============================================================================
# KEY INTEGRATION POINTS
# ============================================================================

"""
1. DATA PREPROCESSING:
   - Both Neural Network and Random Forest use same StandardScaler
   - Same train-test split and feature engineering
   
2. EVALUATION:
   - Both models use same evaluation metrics
   - Common scoring function for fair comparison
   
3. PREDICTION:
   - Neural Network: predict_proba() returns probability [0, 1]
   - Random Forest: predict_proba() returns probability [0, 1]
   - Both can be averaged for ensemble predictions
   
4. MODEL SERIALIZATION:
   - Neural Network: Saved as TensorFlow SavedModel
   - Scaler: Saved as joblib pickle
   - Can be loaded independently for inference
   
5. STREAMLIT INTEGRATION:
   - Both models are cached with @st.cache_resource
   - Same input features
   - Side-by-side display of predictions
   
6. MONITORING:
   - Training history for Neural Network
   - Feature importance from Random Forest
   - Metrics comparison dashboard
"""

if __name__ == "__main__":
    print("Integration examples loaded successfully!")
    print("Run 'python train_model.py' to train models")
    print("Run 'streamlit run app.py' to launch interactive UI")
