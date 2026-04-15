"""
Deep Neural Network Model for Customer Churn Prediction
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import joblib
import json
from pathlib import Path


class ChurnPredictor:
    """Neural Network model for customer churn prediction"""
    
    def __init__(self, input_dim=6):
        """
        Initialize the neural network
        
        Args:
            input_dim: Number of input features (default: 6 for churn dataset)
        """
        self.input_dim = input_dim
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def build_model(self):
        """Build the neural network architecture"""
        self.model = keras.Sequential([
            layers.Input(shape=(self.input_dim,)),
            
            # First hidden layer
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Second hidden layer
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Third hidden layer
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(),
                    keras.metrics.Recall(),
                    keras.metrics.AUC()]
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, verbose=1):
        """
        Train the neural network
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: 0=silent, 1=progress bar, 2=one line per epoch
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Fit the scaler and scale training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Prepare validation data if provided
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
        
        # Define early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        )
        
        # Train the model
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        self.is_trained = True
        return history
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled, verbose=0)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """
        Get probability predictions
        
        Args:
            X: Input features
            
        Returns:
            Probability of churn (0-1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0).flatten()
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        X_test_scaled = self.scaler.transform(X_test)
        loss, accuracy, precision, recall, auc = self.model.evaluate(
            X_test_scaled, y_test, verbose=0
        )
        
        return {
            'Loss': float(loss),
            'Accuracy': float(accuracy),
            'Precision': float(precision),
            'Recall': float(recall),
            'AUC': float(auc)
        }
    
    def save_model(self, model_path='saved_models/neural_network_model',
                   scaler_path='saved_models/scaler.pkl'):
        """Save the trained model and scaler"""
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save neural network
        self.model.save(model_path)
        
        # Save scaler
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path='saved_models/neural_network_model',
                   scaler_path='saved_models/scaler.pkl'):
        """Load a trained model and scaler"""
        self.model = keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.is_trained = True
        
        print(f"Model loaded from {model_path}")
        print(f"Scaler loaded from {scaler_path}")


def train_neural_network(X_train, y_train, X_val=None, y_val=None, 
                        epochs=100):
    """
    Convenience function to train a neural network
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        epochs: Number of epochs
        
    Returns:
        Trained model
    """
    predictor = ChurnPredictor(input_dim=X_train.shape[1])
    predictor.build_model()
    predictor.train(X_train, y_train, X_val, y_val, epochs=epochs, verbose=0)
    return predictor
