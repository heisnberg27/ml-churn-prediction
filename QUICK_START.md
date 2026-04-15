# 🚀 Quick Reference Guide

## Installation (One-Time Setup)

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install specific packages
pip install tensorflow keras scikit-learn streamlit pandas numpy seaborn matplotlib joblib
```

## Usage

### Option A: Train & Evaluate Everything
```bash
python train_model.py
```
Takes a few minutes. Shows:
- Model training progress
- Performance metrics
- Sample predictions
- Saves models to `saved_models/`

### Option B: Launch Interactive App
```bash
streamlit run app.py
```
Opens browser with:
- Customer input form
- Real-time predictions
- Model comparison
- Performance dashboard

### Option C: Run Verification
```bash
python verify_setup.py
```
Checks all dependencies and shows status.

---

## Using the Model in Python

```python
from models.neural_network_model import ChurnPredictor
import pandas as pd

# Create predictor
predictor = ChurnPredictor(input_dim=6)
predictor.build_model()

# Train on your data
predictor.train(X_train, y_train, epochs=50)

# Make predictions
new_customer = pd.DataFrame([{
    'total_orders': 50,
    'festival_orders': 10,
    'inactive_days': 30,
    'total_spent': 5000,
    'avg_order_value': 200,
    'app_sessions_per_month': 20
}])

# Get prediction (0 = Retain, 1 = Churn)
prediction = predictor.predict(new_customer)

# Get probability (0.0 - 1.0)
probability = predictor.predict_proba(new_customer)
print(f"Churn Risk: {probability[0]:.1%}")

# Save for later
predictor.save_model('my_model', 'my_scaler.pkl')
```

---

## API Reference

### ChurnPredictor Class

```python
predictor = ChurnPredictor(input_dim=6)

# Build the model
predictor.build_model()

# Train
history = predictor.train(
    X_train, y_train,           # Training data
    X_val=X_test, y_val=y_test, # Validation (optional)
    epochs=100,                 # Number of iterations
    batch_size=32,              # Batch size
    verbose=0                   # 0=silent, 1=progress, 2=minimal
)

# Predict
predictions = predictor.predict(X)        # Returns 0 or 1
probabilities = predictor.predict_proba(X) # Returns 0.0-1.0

# Evaluate
metrics = predictor.evaluate(X_test, y_test)
# Returns: {Accuracy, Precision, Recall, AUC, Loss}

# Save/Load
predictor.save_model('path/to/model', 'path/to/scaler.pkl')
predictor.load_model('path/to/model', 'path/to/scaler.pkl')
```

---

## Model Details

| Aspect | Details |
|--------|---------|
| **Framework** | TensorFlow/Keras |
| **Architecture** | 3-layer neural network |
| **Input Dim** | 6 customer features |
| **Layers** | 64 → 32 → 16 → 1 |
| **Activation** | ReLU (hidden), Sigmoid (output) |
| **Regularization** | Batch Norm + Dropout |
| **Optimizer** | Adam (lr=0.001) |
| **Loss** | Binary Crossentropy |
| **Metrics** | Accuracy, Precision, Recall, AUC |
| **Early Stopping** | Yes (patience=15) |

---

## Input Features

```
1. total_orders          - Total number of orders (integer)
2. festival_orders       - Orders during festivals (integer)
3. inactive_days         - Days since last activity (integer)
4. total_spent           - Total amount spent (float in $)
5. avg_order_value       - Average order value (float in $)
6. app_sessions_per_month - App usage frequency (integer)
```

---

## Output

**Binary Classification Output**:
- `0` = Customer will NOT churn (Retain)
- `1` = Customer WILL churn (At Risk)

**Probability Output**:
- `0.0 - 0.5` = Low churn risk
- `0.5 - 1.0` = High churn risk
- `0.75+` = Very high risk

---

## File Locations

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web UI |
| `train_model.py` | Training pipeline |
| `verify_setup.py` | Setup verification |
| `models/neural_network_model.py` | DL model code |
| `saved_models/neural_network_model/` | Trained NN |
| `saved_models/scaler.pkl` | Data scaler |
| `data/dataset_without_preprocessing.csv` | Training data |

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError: tensorflow | `pip install tensorflow` |
| ModuleNotFoundError: streamlit | `pip install streamlit` |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| Dataset not found | Ensure `data/dataset_without_preprocessing.csv` exists |
| Models not found | Run `python train_model.py` first |
| Slow training | Reduce epochs or batch size |
| Out of memory | Reduce batch size from 32 to 16 |

---

## Performance Benchmarks

Expected performance on test data (20% of 20K samples):

**Neural Network**:
- Accuracy: ~82-85%
- Precision: ~81-84%
- Recall: ~82-86%
- AUC: ~90-91%

**Random Forest** (Baseline):
- Accuracy: ~80-82%
- Precision: ~79-82%
- Recall: ~80-84%
- AUC: ~88-90%

*Note: Actual performance may vary based on random splits*

---

## Tips & Tricks

### 1. Improve Performance
```python
# Try deeper network
layers.Dense(128, activation='relu'),
layers.Dense(64, activation='relu'),
layers.Dense(32, activation='relu'),

# More epochs
epochs=200,  # instead of 100

# Lower learning rate
optimizer=keras.optimizers.Adam(learning_rate=0.0005)
```

### 2. Prevent Overfitting
```python
# Increase dropout
layers.Dropout(0.5),  # instead of 0.3

# More regularization
optimizer=keras.optimizers.Adam(
    learning_rate=0.001,
    weight_decay=0.01
)
```

### 3. Faster Training
```python
# Reduce network size
layers.Dense(32, activation='relu'),

# Batch size
batch_size=64,  # increase from 32

# Fewer epochs
epochs=50,  # instead of 100
```

---

## Integration with CI/CD

```bash
#!/bin/bash
# train.sh - Automated model training

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running verification..."
python verify_setup.py

echo "Training models..."
python train_model.py

echo "Models saved to saved_models/"
```

---

## Docker Support

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

---

## Got Questions?

1. **Model not training?** → Check `verify_setup.py`
2. **Want faster predictions?** → Use batch processing
3. **Want better accuracy?** → Tune hyperparameters (see INTEGRATION_EXAMPLES.py)
4. **Want to deploy?** → Use saved models in production
5. **Need help?** → Check README.md or IMPLEMENTATION_SUMMARY.md

---

**Last Updated**: 2024
**Deep Learning Framework**: TensorFlow 2.14 with Keras 2.14
