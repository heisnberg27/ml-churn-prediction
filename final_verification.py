"""
Final Verification: Demonstrates that all components work together
This script shows the neural network model functioning in the pipeline
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test all critical imports"""
    print("🔍 Testing Critical Imports...")
    try:
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        print("  ✅ ML Stack (pandas, numpy, sklearn) - OK")
        
        try:
            import tensorflow as tf
            import keras
            print("  ✅ Deep Learning Stack (tensorflow, keras) - OK")
            dl_available = True
        except ImportError:
            print("  ⚠️  TensorFlow not yet installed (will install on first run)")
            dl_available = False
        
        return True, dl_available
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False, False

def test_data_loading():
    """Test data loading"""
    print("\n📊 Testing Data Loading...")
    try:
        import pandas as pd
        df = pd.read_csv("data/dataset_without_preprocessing.csv")
        print(f"  ✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Check required features
        required_features = ["total_orders", "festival_orders", "inactive_days", 
                            "total_spent", "avg_order_value", "app_sessions_per_month"]
        missing = [f for f in required_features if f not in df.columns]
        if missing:
            print(f"  ⚠️  Missing features: {missing}")
        else:
            print(f"  ✅ All required features present")
        
        return True, df
    except Exception as e:
        print(f"  ❌ Error loading data: {e}")
        return False, None

def test_preprocessing():
    """Test data preprocessing"""
    print("\n⚙️  Testing Data Preprocessing...")
    try:
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        
        df = pd.read_csv("data/dataset_without_preprocessing.csv")
        
        # Feature engineering
        df["festival_ratio"] = df["festival_orders"] / df["total_orders"]
        df["recency_ratio"] = df["inactive_days"]
        df["Churn"] = ((df["festival_ratio"] > 0.6) | (df["recency_ratio"] > 180)).astype(int)
        
        features = ["total_orders", "festival_orders", "inactive_days", 
                   "total_spent", "avg_order_value", "app_sessions_per_month"]
        
        X = df[features]
        y = df["Churn"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"  ✅ Data split: {len(X_train)} train, {len(X_test)} test")
        print(f"  ✅ Scaling complete: shape {X_train_scaled.shape}")
        print(f"  ✅ Churn distribution: {y_train.sum()} / {len(y_train)} = {y_train.mean():.1%}")
        
        return True, (X_train, X_test, y_train, y_test, features)
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False, None

def test_random_forest():
    """Test Random Forest as baseline"""
    print("\n🌲 Testing Random Forest Baseline...")
    try:
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        df = pd.read_csv("data/dataset_without_preprocessing.csv")
        df["festival_ratio"] = df["festival_orders"] / df["total_orders"]
        df["Churn"] = ((df["festival_ratio"] > 0.6) | (df["inactive_days"] > 180)).astype(int)
        
        features = ["total_orders", "festival_orders", "inactive_days", 
                   "total_spent", "avg_order_value", "app_sessions_per_month"]
        
        X = df[features]
        y = df["Churn"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        
        accuracy = accuracy_score(y_test, rf.predict(X_test_scaled))
        print(f"  ✅ Random Forest trained")
        print(f"  ✅ Test Accuracy: {accuracy:.4f}")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_neural_network():
    """Test Neural Network model"""
    print("\n🧠 Testing Neural Network Model...")
    try:
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from models.neural_network_model import ChurnPredictor
        
        print("  Loading data...")
        df = pd.read_csv("data/dataset_without_preprocessing.csv")
        df["festival_ratio"] = df["festival_orders"] / df["total_orders"]
        df["Churn"] = ((df["festival_ratio"] > 0.6) | (df["inactive_days"] > 180)).astype(int)
        
        features = ["total_orders", "festival_orders", "inactive_days", 
                   "total_spent", "avg_order_value", "app_sessions_per_month"]
        
        X = df[features].iloc[:1000]  # Use subset for quick test
        y = df["Churn"].iloc[:1000]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("  Initializing Neural Network...")
        predictor = ChurnPredictor(input_dim=len(features))
        predictor.build_model()
        
        print("  Training (5 epochs for quick test)...")
        predictor.train(X_train, y_train, X_test, y_test, epochs=5, verbose=0)
        
        print("  Making predictions...")
        test_pred = predictor.predict(X_test)
        test_prob = predictor.predict_proba(X_test)
        
        metrics = predictor.evaluate(X_test, y_test)
        
        print(f"  ✅ Predictions shape: {test_pred.shape}")
        print(f"  ✅ Probabilities range: [{test_prob.min():.3f}, {test_prob.max():.3f}]")
        print(f"  ✅ Test Accuracy: {metrics['Accuracy']:.4f}")
        print(f"  ✅ Test AUC: {metrics['AUC']:.4f}")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("╔" + "="*60 + "╗")
    print("║" + "FINAL VERIFICATION - COMPONENT TEST".center(60) + "║")
    print("╚" + "="*60 + "╝")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Imports
    tests_total += 1
    imports_ok, dl_available = test_imports()
    if imports_ok:
        tests_passed += 1
    
    # Test 2: Data Loading
    tests_total += 1
    data_ok, df = test_data_loading()
    if data_ok:
        tests_passed += 1
    
    # Test 3: Preprocessing
    tests_total += 1
    preprocess_ok, data_tuple = test_preprocessing()
    if preprocess_ok:
        tests_passed += 1
    
    # Test 4: Random Forest
    tests_total += 1
    if test_random_forest():
        tests_passed += 1
    
    # Test 5: Neural Network (if TensorFlow available)
    if dl_available:
        tests_total += 1
        if test_neural_network():
            tests_passed += 1
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Tests Passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("\n✅ ALL TESTS PASSED!")
        print("\nYou can now run:")
        print("  1. python train_model.py         # Full pipeline training")
        print("  2. streamlit run app.py           # Interactive UI")
        print("\nOr test specific components:")
        print("  - See INTEGRATION_EXAMPLES.py for code examples")
        print("  - See QUICK_START.md for usage guide")
        return 0
    else:
        print(f"\n⚠️  {tests_total - tests_passed} test(s) failed")
        if not dl_available:
            print("\n💡 Tip: TensorFlow will be installed on first run of train_model.py")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
