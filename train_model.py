"""
Comprehensive Training and Evaluation Script
Demonstrates the Deep Learning Neural Network model for customer churn prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
from models.neural_network_model import ChurnPredictor, train_neural_network
import os

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_and_prepare_data():
    """Load and prepare the dataset"""
    print("=" * 80)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("=" * 80)
    
    df = pd.read_csv("data/dataset_without_preprocessing.csv")
    print(f"✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"\nFirst few rows:\n{df.head()}")
    
    # Feature engineering
    df["festival_ratio"] = df["festival_orders"] / df["total_orders"]
    df["recency_ratio"] = df["inactive_days"]
    
    # Create churn label
    df["Churn"] = df.apply(
        lambda x: 1 if (x["festival_ratio"] > 0.6 or x["recency_ratio"] > 180) else 0,
        axis=1
    )
    
    # Add noise
    noise = np.random.rand(len(df)) < 0.1
    df.loc[noise, "Churn"] = 1 - df.loc[noise, "Churn"]
    
    print(f"✓ Churn label created")
    print(f"  - Churn cases: {df['Churn'].sum()} ({df['Churn'].mean()*100:.1f}%)")
    print(f"  - Non-churn cases: {(1-df['Churn']).sum()} ({(1-df['Churn']).mean()*100:.1f}%)")
    
    return df

def split_and_scale_data(df):
    """Split and scale the dataset"""
    print("\n" + "=" * 80)
    print("STEP 2: SPLITTING AND SCALING DATA")
    print("=" * 80)
    
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
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Data split: {len(X_train)} train, {len(X_test)} test")
    print(f"  - Train churn rate: {y_train.mean():.2%}")
    print(f"  - Test churn rate: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test, features

def train_neural_network_model(X_train, X_test, y_train, y_test):
    """Train the Deep Learning Neural Network"""
    print("\n" + "=" * 80)
    print("STEP 3: TRAINING DEEP LEARNING MODEL (Neural Network)")
    print("=" * 80)
    
    print(f"\n📊 Neural Network Architecture:")
    print(f"  Input Layer: {X_train.shape[1]} features")
    print(f"  Hidden Layer 1: 64 neurons (ReLU) + BatchNorm + Dropout")
    print(f"  Hidden Layer 2: 32 neurons (ReLU) + BatchNorm + Dropout")
    print(f"  Hidden Layer 3: 16 neurons (ReLU) + Dropout")
    print(f"  Output Layer: 1 neuron (Sigmoid) - Binary classification")
    
    # Create and train model
    nn_predictor = ChurnPredictor(input_dim=X_train.shape[1])
    nn_predictor.build_model()
    
    print(f"\n⏳ Training Neural Network...")
    history = nn_predictor.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=100,
        batch_size=32,
        verbose=0
    )
    
    print(f"✓ Neural Network training completed!")
    print(f"  - Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"  - Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print(f"  - Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    return nn_predictor, history

def train_random_forest_model(X_train, X_test, y_train, y_test):
    """Train the ML baseline model"""
    print("\n" + "=" * 80)
    print("STEP 4: TRAINING MACHINE LEARNING BASELINE (Random Forest)")
    print("=" * 80)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    print(f"✓ Random Forest training completed!")
    print(f"  - Number of trees: 100")
    print(f"  - Training accuracy: {rf_model.score(X_train_scaled, y_train):.4f}")
    
    return rf_model, scaler

def evaluate_neural_network(nn_predictor, X_test, y_test):
    """Evaluate Neural Network performance"""
    print("\n" + "=" * 80)
    print("STEP 5: EVALUATING NEURAL NETWORK")
    print("=" * 80)
    
    nn_metrics = nn_predictor.evaluate(X_test, y_test)
    
    print(f"\n🧠 Neural Network Performance Metrics:")
    print(f"  - Accuracy:  {nn_metrics['Accuracy']:.4f}")
    print(f"  - Precision: {nn_metrics['Precision']:.4f}")
    print(f"  - Recall:    {nn_metrics['Recall']:.4f}")
    print(f"  - AUC:       {nn_metrics['AUC']:.4f}")
    print(f"  - Loss:      {nn_metrics['Loss']:.4f}")
    
    # Get predictions
    y_pred_nn = nn_predictor.predict(X_test)
    y_pred_proba_nn = nn_predictor.predict_proba(X_test)
    
    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_nn)
    print(f"    True Negatives:  {cm[0, 0]}")
    print(f"    False Positives: {cm[0, 1]}")
    print(f"    False Negatives: {cm[1, 0]}")
    print(f"    True Positives:  {cm[1, 1]}")
    
    return nn_metrics, y_pred_nn, y_pred_proba_nn, cm

def evaluate_random_forest(rf_model, scaler, X_test, y_test):
    """Evaluate Random Forest performance"""
    print("\n" + "=" * 80)
    print("STEP 6: EVALUATING RANDOM FOREST")
    print("=" * 80)
    
    X_test_scaled = scaler.transform(X_test)
    y_pred_rf = rf_model.predict(X_test_scaled)
    y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"\n🌲 Random Forest Performance Metrics:")
    print(f"  - Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"  - Precision: {precision_score(y_test, y_pred_rf):.4f}")
    print(f"  - Recall:    {recall_score(y_test, y_pred_rf):.4f}")
    print(f"  - AUC:       {roc_auc_score(y_test, y_pred_proba_rf):.4f}")
    print(f"  - F1 Score:  {f1_score(y_test, y_pred_rf):.4f}")
    
    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_rf)
    print(f"    True Negatives:  {cm[0, 0]}")
    print(f"    False Positives: {cm[0, 1]}")
    print(f"    False Negatives: {cm[1, 0]}")
    print(f"    True Positives:  {cm[1, 1]}")
    
    return y_pred_rf, y_pred_proba_rf, cm

def compare_models(nn_metrics, y_pred_nn, y_pred_rf, y_test):
    """Compare both models"""
    print("\n" + "=" * 80)
    print("STEP 7: MODEL COMPARISON")
    print("=" * 80)
    
    rf_acc = accuracy_score(y_test, y_pred_rf)
    nn_acc = nn_metrics['Accuracy']
    
    print(f"\n📊 Model Performance Comparison:")
    print(f"  Neural Network Accuracy: {nn_acc:.4f} {'✓ Better' if nn_acc > rf_acc else ''}")
    print(f"  Random Forest Accuracy:  {rf_acc:.4f} {'✓ Better' if rf_acc > nn_acc else ''}")
    
    # Agreement
    agreement = np.sum(y_pred_nn == y_pred_rf) / len(y_test)
    print(f"\n  Model Agreement: {agreement:.2%}")
    print(f"  Models Disagree: {(1-agreement):.2%}")

def demo_predictions(nn_predictor, rf_model, scaler):
    """Demonstrate predictions on sample customers"""
    print("\n" + "=" * 80)
    print("STEP 8: SAMPLE PREDICTIONS")
    print("=" * 80)
    
    # Create sample customers
    samples = pd.DataFrame([
        {
            "total_orders": 100,
            "festival_orders": 5,
            "inactive_days": 10,
            "total_spent": 20000,
            "avg_order_value": 500,
            "app_sessions_per_month": 45
        },
        {
            "total_orders": 20,
            "festival_orders": 18,
            "inactive_days": 250,
            "total_spent": 2000,
            "avg_order_value": 100,
            "app_sessions_per_month": 2
        },
        {
            "total_orders": 150,
            "festival_orders": 90,
            "inactive_days": 5,
            "total_spent": 50000,
            "avg_order_value": 800,
            "app_sessions_per_month": 60
        }
    ])
    
    print(f"\n🔮 Making predictions on 3 sample customers:\n")
    
    for idx, sample in samples.iterrows():
        print(f"Customer {idx + 1}:")
        print(f"  Orders: {sample['total_orders']}, Festival: {sample['festival_orders']}, "
              f"Inactive: {sample['inactive_days']} days")
        
        # Neural Network prediction
        nn_prob = nn_predictor.predict_proba(sample.values.reshape(1, -1))[0]
        nn_pred = 1 if nn_prob > 0.5 else 0
        
        # Random Forest prediction
        sample_scaled = scaler.transform(sample.values.reshape(1, -1))
        rf_prob = rf_model.predict_proba(sample_scaled)[0][1]
        rf_pred = rf_model.predict(sample_scaled)[0]
        
        print(f"  Neural Network: {nn_prob:.2%} churn risk {'⚠️ CHURN' if nn_pred == 1 else '✅ RETAIN'}")
        print(f"  Random Forest:  {rf_prob:.2%} churn risk {'⚠️ CHURN' if rf_pred == 1 else '✅ RETAIN'}")
        print()

def main():
    """Main execution"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  CUSTOMER CHURN PREDICTION: DEEP LEARNING INTEGRATION".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Step 1: Load data
    df = load_and_prepare_data()
    
    # Step 2: Prepare data
    X_train, X_test, y_train, y_test, features = split_and_scale_data(df)
    
    # Step 3: Train Neural Network
    nn_predictor, history = train_neural_network_model(X_train, X_test, y_train, y_test)
    
    # Step 4: Train Random Forest baseline
    rf_model, scaler = train_random_forest_model(X_train, X_test, y_train, y_test)
    
    # Step 5: Evaluate Neural Network
    nn_metrics, y_pred_nn, y_pred_proba_nn, cm_nn = evaluate_neural_network(
        nn_predictor, X_test, y_test
    )
    
    # Step 6: Evaluate Random Forest
    y_pred_rf, y_pred_proba_rf, cm_rf = evaluate_random_forest(
        rf_model, scaler, X_test, y_test
    )
    
    # Step 7: Compare models
    compare_models(nn_metrics, y_pred_nn, y_pred_rf, y_test)
    
    # Step 8: Demo predictions
    demo_predictions(nn_predictor, rf_model, scaler)
    
    # Save models
    print("=" * 80)
    print("SAVING MODELS")
    print("=" * 80)
    os.makedirs('saved_models', exist_ok=True)
    nn_predictor.save_model()
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print("\n🚀 Next Steps:")
    print("  1. Run the Streamlit app: streamlit run app.py")
    print("  2. Try different customer profiles")
    print("  3. Compare Neural Network vs Random Forest predictions")
    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()
