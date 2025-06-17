import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump, load
import os

class KrillHerdOptimizer:
    def __init__(self, n_krill=50, max_iter=100, n_features=None):
        self.n_krill = n_krill
        self.max_iter = max_iter
        self.n_features = n_features
        self.best_solution = None
        self.best_fitness = float('inf')

    def optimize(self, X, y, model):
        # Initialize krill positions
        krill = np.random.rand(self.n_krill, self.n_features)
        
        for _ in range(self.max_iter):
            # Evaluate fitness for each krill
            fitness = []
            for k in krill:
                # Select features based on krill position
                selected_features = k > 0.5
                if np.sum(selected_features) == 0:
                    continue
                
                # Train model with selected features
                model.fit(X[:, selected_features], y)
                score = -model.score(X[:, selected_features], y)
                fitness.append(score)
                
                if score < self.best_fitness:
                    self.best_fitness = score
                    self.best_solution = selected_features
            
            # Update krill positions
            krill = self._update_positions(krill, fitness)
        
        return self.best_solution

    def _update_positions(self, krill, fitness):
        # Simple position update rule
        best_idx = np.argmin(fitness)
        best_krill = krill[best_idx]
        
        for i in range(self.n_krill):
            if i != best_idx:
                krill[i] += 0.1 * (best_krill - krill[i])
                krill[i] = np.clip(krill[i], 0, 1)
        
        return krill

def load_and_preprocess_data():
    # Load the dataset
    df = pd.read_csv('/mnt/data/Chennai housing sale.csv')
    
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Separate features and target
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Create the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and run KHA
    kha = KrillHerdOptimizer(n_features=X.shape[1])
    best_features = kha.optimize(X_train.values, y_train.values, model)
    
    # Train final model with best features
    model.fit(X_train, y_train)
    
    # Save the model
    os.makedirs('ml_model', exist_ok=True)
    dump(model, 'ml_model/model.pkl')
    
    return model, X.columns

def preprocess_input(input_data):
    # Convert input data to DataFrame
    df = pd.DataFrame([input_data])
    
    # Load the preprocessing pipeline
    model = load('ml_model/model.pkl')
    
    # Preprocess the input data
    processed_data = model.named_steps['preprocessor'].transform(df)
    
    return processed_data 