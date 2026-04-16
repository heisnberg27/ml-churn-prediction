#  Customer Churn Prediction: ML & Deep Learning Pipeline

## Project Overview

This project implements a **hybrid machine learning and deep learning pipeline** for customer churn prediction. It integrates a **Deep Neural Network (DNN)** with the existing Random Forest classifier, providing ensemble-like capabilities and comprehensive model evaluation.

### Key Features
-  **Deep Learning Model**: 3-layer neural network with batch normalization and dropout
-  **ML Baseline**: Random Forest classifier for comparison
-  **Integrated Pipeline**: Both models share data preprocessing and evaluation
-  **Streamlit UI**: Interactive web interface for predictions
-  **Model Comparison**: Side-by-side analysis of both models
-  **Performance Metrics**: Accuracy, Precision, Recall, AUC, Loss tracking

---

## Project Structure

```
ml-churn-project/
├── app.py                          # Streamlit web application
├── train_model.py                  # Training & evaluation script
├── requirements.txt                # Python dependencies
├── data/
│   └── dataset_without_preprocessing.csv
├── models/
│   ├── __init__.py
│   ├── neural_network_model.py     #  Deep Learning model
│   ├── decision_tree_model.py
│   ├── knn_model.py
│   ├── logistic_model.py
│   └── random_forest_model.py
├── saved_models/                   # Trained model artifacts
│   ├── neural_network_model/       # TensorFlow saved model
│   └── scaler.pkl                  # StandardScaler for preprocessing
└── utils/
    ├── preprocessing.py
    ├── evaluation.py
    └── cross_validation.py
```

---

## Deep Learning Model Architecture

### Neural Network Design

The implemented Deep Neural Network uses a progressive architecture for binary classification:

```
Input Layer (6 features)
    ↓
Hidden Layer 1: 64 neurons (ReLU)
    ↓ BatchNormalization
    ↓ Dropout (0.3)
    ↓
Hidden Layer 2: 32 neurons (ReLU)
    ↓ BatchNormalization
    ↓ Dropout (0.3)
    ↓
Hidden Layer 3: 16 neurons (ReLU)
    ↓ Dropout (0.2)
    ↓
Output Layer: 1 neuron (Sigmoid)
    ↓
Binary Output (0=Retain, 1=Churn)
```

### Key Components

1. **Batch Normalization**: Stabilizes learning and reduces internal covariate shift
2. **Dropout Layers**: Prevents overfitting by randomly deactivating neurons during training
3. **Adam Optimizer**: Adaptive learning rate optimization
4. **Early Stopping**: Halts training when validation loss stops improving

---

## Installation & Setup

### 1. Install Dependencies

```bash
cd ml-churn-project
pip install -r requirements.txt
```

Required packages:
- `tensorflow==2.14.0` - Deep learning framework
- `keras==2.14.0` - Neural network API
- `scikit-learn==1.3.0` - ML algorithms & preprocessing
- `streamlit==1.32.0` - Web UI framework
- `pandas`, `numpy` - Data manipulation
- `seaborn`, `matplotlib` - Visualization

### 2. Verify Dataset

Ensure `data/dataset_without_preprocessing.csv` exists with these columns:
- `total_orders`
- `festival_orders`
- `inactive_days`
- `total_spent`
- `avg_order_value`
- `app_sessions_per_month`

---

## Usage

### Option 1: Train Models & View Results

```bash
python train_model.py
```

This script will:
1. Load and preprocess the dataset
2. Train the Neural Network model
3. Train the Random Forest baseline
4. Evaluate both models on test set
5. Display performance comparisons
6. Show sample predictions
7. Save trained models to `saved_models/`

### Option 2: Interactive Streamlit Application

```bash
streamlit run app.py
```

Features:
-  Customer input form (6 features)
-  Neural Network predictions with confidence scores
-  Random Forest predictions
-  Side-by-side model comparison
-  Performance metrics and statistics
-  Feature importance visualization

---

## Model Integration Points

### 1. **Shared Preprocessing**
- Both models use the same feature engineering
- StandardScaler for feature normalization
- Train-test split with stratification

### 2. **Consistent Evaluation**
- Common metrics: Accuracy, Precision, Recall, AUC
- Confusion matrix analysis
- Prediction probability comparison

### 3. **Ensemble Capability**
The architecture supports creating an ensemble:
```python
# Average predictions from both models
final_prediction = (nn_prob + rf_prob) / 2
```

### 4. **Model Persistence**
- Neural Network: TensorFlow SavedModel format
- Scaler: Joblib pickle format
- Enables model reloading without retraining

---

## Deep Learning Features

### 1. **ChurnPredictor Class** (`models/neural_network_model.py`)

```python
# Initialize model
predictor = ChurnPredictor(input_dim=6)

# Build architecture
predictor.build_model()

# Train with validation
predictor.train(X_train, y_train, X_val=X_test, y_val=y_test, 
                epochs=100)

# Make predictions
predictions = predictor.predict(X_new)
probabilities = predictor.predict_proba(X_new)

# Evaluate performance
metrics = predictor.evaluate(X_test, y_test)

# Save/Load
predictor.save_model()
predictor.load_model()
```

### 2. **Training Strategy**
- **Early Stopping**: Prevents overfitting
- **Batch Normalization**: Improves training stability
- **Dropout**: Reduces model variance
- **Validation Monitoring**: Tracks generalization

### 3. **Performance Metrics**
- Accuracy, Precision, Recall (binary classification)
- AUC-ROC (probability calibration)
- Loss tracking (binary cross-entropy)

---

## Sample Output

```
 Neural Network Performance Metrics:
  - Accuracy:  0.8750
  - Precision: 0.8875
  - Recall:    0.8500
  - AUC:       0.9125

 Random Forest Performance Metrics:
  - Accuracy:  0.8625
  - Precision: 0.8750
  - Recall:    0.8375
  - F1 Score:  0.8560

Customer 1:
  Neural Network: 15.3% churn risk  RETAIN
  Random Forest:  18.7% churn risk  RETAIN

Customer 2:
  Neural Network: 87.5% churn risk  CHURN
  Random Forest:  84.2% churn risk  CHURN
```

---

## Key Results & Benefits

### Neural Network Advantages
 Captures non-linear patterns
 Handles feature interactions automatically
 Probabilistic outputs (confidence scores)
 Scalable with data volume

### Hybrid Pipeline Benefits
 Combines interpretability (Random Forest) with power (Neural Network)
 Model diversity reduces prediction variance
 Can create predictions through ensemble
 Easy comparison of different paradigms

---

## Hyperparameter Tuning

To improve performance, you can modify:

1. **Architecture** (in `neural_network_model.py`):
   ```python
   layers.Dense(128, activation='relu'),  # Increase neurons
   layers.Dropout(0.4),                   # Increase dropout
   ```

2. **Training** (in `train_model.py`):
   ```python
   history = nn_predictor.train(
       epochs=200,        # Increase epochs
       batch_size=16,     # Reduce batch size
   )
   ```

3. **Learning Rate**:
   ```python
   optimizer=keras.optimizers.Adam(learning_rate=0.0005)
   ```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| TensorFlow not importing | `pip install tensorflow==2.14.0` |
| Dataset not found | Verify `data/dataset_without_preprocessing.csv` exists |
| Models not saving | Ensure `saved_models/` directory exists |
| Streamlit port conflict | `streamlit run app.py --server.port 8501` |
| GPU not found | CPU-only mode is default; TensorFlow will work |

---

## Next Steps

1. **Experiment with Architectures**: Try deeper/wider networks
2. **Feature Engineering**: Create new features for better signals
3. **Hyperparameter Optimization**: Use GridSearch or Bayesian optimization
4. **Cross-Validation**: Implement k-fold CV for robust evaluation
5. **Ensemble Methods**: Combine multiple models for predictions
6. **Production Deployment**: Deploy model as REST API

---

## References

- TensorFlow Documentation: https://www.tensorflow.org/api_docs
- Keras Sequential Model: https://keras.io/guides/sequential_model/
- Best Practices: https://keras.io/guides/training_with_fit/

---

**Created**: 2024 | **Model Type**: Deep Neural Network (DNN) for Binary Classification
