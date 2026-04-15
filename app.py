import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from models.neural_network_model import ChurnPredictor
import tensorflow as tf

# Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')

# Set page config
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Home", "Model Comparison & Stats"]
)

# Use sidebar for model selection on home page
if page == "Home":
    model_choice = st.sidebar.radio(
        "Select Model Type",
        ["Neural Network (DL)", "Random Forest (ML)", "Compare Both"]
    )
else:
    model_choice = None

# Main Title
st.title("Customer Churn Prediction System")
st.markdown("Powered by Machine Learning & Deep Learning Models")

# -------------------------------
# LOAD DATA & PREPROCESSING
# -------------------------------
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    df = pd.read_csv("data/dataset_without_preprocessing.csv")
    
    # Feature engineering
    df["festival_ratio"] = df["festival_orders"] / df["total_orders"]
    df["recency_ratio"] = df["inactive_days"]
    
    # Create churn label
    df["Churn"] = df.apply(
        lambda x: 1 if (x["festival_ratio"] > 0.6 or x["recency_ratio"] > 180) else 0,
        axis=1
    )
    
    # Add noise (reduce overfitting)
    noise = np.random.rand(len(df)) < 0.1
    df.loc[noise, "Churn"] = 1 - df.loc[noise, "Churn"]
    
    return df

@st.cache_resource
def train_models():
    """Train both ML and DL models"""
    df = load_and_preprocess_data()
    
    features = [
        "total_orders",
        "festival_orders",
        "inactive_days",
        "total_spent",
        "avg_order_value",
        "app_sessions_per_month"
    ]
    
    X = df[features]
    y = df["Churn"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # ===== Train Random Forest =====
    scaler_rf = StandardScaler()
    X_train_rf = scaler_rf.fit_transform(X_train)
    X_test_rf = scaler_rf.transform(X_test)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_rf, y_train)
    
    # ===== Train Neural Network =====
    nn_predictor = ChurnPredictor(input_dim=len(features))
    nn_predictor.build_model()
    nn_predictor.train(X_train, y_train, X_test, y_test, epochs=100, verbose=0)
    
    return {
        'rf_model': rf_model,
        'rf_scaler': scaler_rf,
        'nn_predictor': nn_predictor,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'features': features
    }

# Load models
models_data = train_models()
rf_model = models_data['rf_model']
rf_scaler = models_data['rf_scaler']
nn_predictor = models_data['nn_predictor']
X_test = models_data['X_test']
y_test = models_data['y_test']
features = models_data['features']

# ========================
# PAGE 1: HOME PAGE
# ========================
if page == "Home":
    st.markdown("Enter customer details below to predict churn risk using machine learning and deep learning models.")
    st.markdown("")
    
    # USER INPUT SECTION
    st.header("Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        total_orders = st.number_input("Total Orders", min_value=1, value=50)
        inactive_days = st.number_input("Inactive Days", min_value=0, value=30)
        avg_order_value = st.number_input("Average Order Value", min_value=0.0, value=200.0)

    with col2:
        festival_orders = st.number_input("Festival Orders", min_value=0, value=10)
        total_spent = st.number_input("Total Spent", min_value=0.0, value=5000.0)
        app_sessions = st.number_input("App Sessions per Month", min_value=0, value=20)

    with col3:
        st.write("")  # Spacing
        st.write("")
        predict_btn = st.button("Predict Churn", use_container_width=True)

    # PREDICTION & RESULTS
    if predict_btn:
        input_data = pd.DataFrame([{
            "total_orders": total_orders,
            "festival_orders": festival_orders,
            "inactive_days": inactive_days,
            "total_spent": total_spent,
            "avg_order_value": avg_order_value,
            "app_sessions_per_month": app_sessions
        }])
        
        st.markdown("---")
        st.subheader("Prediction Results")
        
        if model_choice in ["Neural Network (DL)", "Compare Both"]:
            # Neural Network Prediction
            nn_prob = nn_predictor.predict_proba(input_data)[0]
            nn_pred = 1 if nn_prob > 0.5 else 0
            
            col_nn, _ = st.columns([1, 1])
            with col_nn:
                st.info("**Deep Learning Model (Neural Network)**")
                st.metric("Churn Risk", f"{nn_prob*100:.1f}%", 
                         delta="HIGH RISK" if nn_pred == 1 else "LOW RISK")
                st.write(f"Model Prediction: {('CHURN' if nn_pred == 1 else 'RETAIN')}")
        
        if model_choice in ["Random Forest (ML)", "Compare Both"]:
            # Random Forest Prediction
            input_scaled = rf_scaler.transform(input_data)
            rf_pred = rf_model.predict(input_scaled)[0]
            rf_prob = rf_model.predict_proba(input_scaled)[0][1]
            
            col_rf, _ = st.columns([1, 1])
            with col_rf:
                st.info("**Machine Learning Model (Random Forest)**")
                st.metric("Churn Risk", f"{rf_prob*100:.1f}%",
                         delta="HIGH RISK" if rf_pred == 1 else "LOW RISK")
                st.write(f"Model Prediction: {('CHURN' if rf_pred == 1 else 'RETAIN')}")
        
        if model_choice == "Compare Both":
            st.markdown("---")
            agreement = "Models Agree" if nn_pred == rf_pred else "Models Disagree"
            st.success(f"**Model Consensus: {agreement}**")

# ========================
# PAGE 2: MODEL COMPARISON & STATS
# ========================
elif page == "Model Comparison & Stats":
    st.markdown("Compare performance metrics and detailed statistics between the two models.")
    st.markdown("")
    
    st.header("Model Performance Evaluation")

    eval_tab1, eval_tab2, eval_tab3 = st.tabs(["Neural Network Performance", "Random Forest Performance", "Comparison"])

    with eval_tab1:
        st.subheader("Deep Learning Model Metrics")
        nn_metrics = nn_predictor.evaluate(X_test, y_test)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{nn_metrics['Accuracy']:.4f}")
        col2.metric("Precision", f"{nn_metrics['Precision']:.4f}")
        col3.metric("Recall", f"{nn_metrics['Recall']:.4f}")
        col4.metric("AUC", f"{nn_metrics['AUC']:.4f}")
        col5.metric("Loss", f"{nn_metrics['Loss']:.4f}")

    with eval_tab2:
        st.subheader("Machine Learning Model Metrics")
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        X_test_rf = rf_scaler.transform(X_test)
        rf_pred_test = rf_model.predict(X_test_rf)
        rf_prob_test = rf_model.predict_proba(X_test_rf)[:, 1]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy_score(y_test, rf_pred_test):.4f}")
        col2.metric("Precision", f"{precision_score(y_test, rf_pred_test):.4f}")
        col3.metric("Recall", f"{recall_score(y_test, rf_pred_test):.4f}")
        col4.metric("AUC", f"{roc_auc_score(y_test, rf_prob_test):.4f}")

    with eval_tab3:
        st.subheader("Models Comparison")
        
        # Create comparison dataframe
        nn_metrics = nn_predictor.evaluate(X_test, y_test)
        X_test_rf = rf_scaler.transform(X_test)
        rf_pred_test = rf_model.predict(X_test_rf)
        rf_prob_test = rf_model.predict_proba(X_test_rf)[:, 1]
        
        comparison_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'AUC'],
            'Neural Network': [
                nn_metrics['Accuracy'],
                nn_metrics['Precision'],
                nn_metrics['Recall'],
                nn_metrics['AUC']
            ],
            'Random Forest': [
                accuracy_score(y_test, rf_pred_test),
                precision_score(y_test, rf_pred_test),
                recall_score(y_test, rf_pred_test),
                roc_auc_score(y_test, rf_prob_test)
            ]
        }
        comparison_df = pd.DataFrame(comparison_data)
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Bar chart comparison
        st.bar_chart(comparison_df.set_index('Metric'))
    
    st.markdown("---")
    st.header("Dataset Information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", len(X_test) + len(models_data['X_train']))
    col2.metric("Features", len(features))
    col3.metric("Test Set Size", len(X_test))

    with st.expander("View Feature Importance"):
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.bar_chart(importance_df.set_index('Feature'))