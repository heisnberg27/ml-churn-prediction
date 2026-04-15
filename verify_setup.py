"""
Quick verification script to ensure all components are properly integrated
Run this to validate the Deep Learning pipeline before using the app
"""

import sys
import os

def check_imports():
    """Verify all required packages are installed"""
    print("✓ Checking imports...")
    required_packages = {
        'tensorflow': 'tensorflow',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'streamlit': 'streamlit',
    }
    
    missing = []
    for module, package_name in required_packages.items():
        try:
            __import__(module)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - MISSING")
            missing.append(package_name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    return True

def check_data():
    """Verify dataset exists"""
    print("\n✓ Checking dataset...")
    if os.path.exists("data/dataset_without_preprocessing.csv"):
        print("  ✓ Dataset found")
        import pandas as pd
        df = pd.read_csv("data/dataset_without_preprocessing.csv")
        print(f"  ✓ Samples: {len(df)}, Features: {len(df.columns)}")
        return True
    else:
        print("  ✗ Dataset not found at data/dataset_without_preprocessing.csv")
        return False

def check_models():
    """Verify model files exist"""
    print("\n✓ Checking model files...")
    model_files = [
        'models/neural_network_model.py',
        'app.py',
        'train_model.py'
    ]
    
    for file in model_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - MISSING")
            return False
    return True

def test_neural_network():
    """Quick test of neural network model"""
    print("\n✓ Testing Neural Network model...")
    try:
        import numpy as np
        from models.neural_network_model import ChurnPredictor
        
        # Create dummy data
        X_dummy = np.random.randn(10, 6)  # 10 samples, 6 features
        y_dummy = np.random.randint(0, 2, 10)
        
        # Initialize and train
        predictor = ChurnPredictor(input_dim=6)
        predictor.build_model()
        predictor.train(X_dummy, y_dummy, epochs=2, verbose=0)
        
        # Test prediction
        test_sample = np.random.randn(1, 6)
        pred = predictor.predict(test_sample)
        prob = predictor.predict_proba(test_sample)
        
        print(f"  ✓ Model initialized successfully")
        print(f"  ✓ Training works")
        print(f"  ✓ Predictions: {pred[0]} (prob: {prob[0]:.2%})")
        return True
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return False

def main():
    """Run all checks"""
    print("=" * 60)
    print("DEEP LEARNING PIPELINE VERIFICATION")
    print("=" * 60)
    
    checks = [
        ("Imports", check_imports),
        ("Dataset", check_data),
        ("Model Files", check_models),
        ("Neural Network", test_neural_network),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"Error during {name} check: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✅ All checks passed! You can now run:")
        print("   1. python train_model.py     # Train models")
        print("   2. streamlit run app.py       # Launch interactive app")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
