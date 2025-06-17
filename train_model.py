import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import joblib
import os

# Create ml_model directory if it doesn't exist
os.makedirs('ml_model', exist_ok=True)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('Book_PREDICT_with_coords.csv')

# Define features and target
features = ['INT_SQFT', 'DISTANT FROM THE MAINROAD', 'NUMBER OF BEDROOM', 
           'NUMBER OF BATHROOM', 'NUMBER OF ROOM', 'REGISTRATION FEE', 
           'COMMISSSION PRICE', 'AREA']
target = 'SALES PRICE'

# Prepare feature information
feature_info = {
    'feature_names': features,
    'categorical_features': ['AREA'],
    'numerical_features': [f for f in features if f != 'AREA']
}

# Initialize label encoders for categorical variables
label_encoders = {}
for cat_feature in feature_info['categorical_features']:
    le = LabelEncoder()
    df[cat_feature] = le.fit_transform(df[cat_feature].astype(str))
    label_encoders[cat_feature] = le

# Split features and target
X = df[features]
y = df[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values for numerical features
print("Handling missing values...")
numerical_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Split features into numerical and categorical
X_train_num = X_train[feature_info['numerical_features']]
X_test_num = X_test[feature_info['numerical_features']]
X_train_cat = X_train[feature_info['categorical_features']]
X_test_cat = X_test[feature_info['categorical_features']]

# Impute missing values
X_train_num_imputed = numerical_imputer.fit_transform(X_train_num)
X_test_num_imputed = numerical_imputer.transform(X_test_num)
X_train_cat_imputed = categorical_imputer.fit_transform(X_train_cat)
X_test_cat_imputed = categorical_imputer.transform(X_test_cat)

# Scale numerical features
print("Scaling features...")
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num_imputed)
X_test_num_scaled = scaler.transform(X_test_num_imputed)

# Combine numerical and categorical features
X_train_processed = np.hstack((X_train_num_scaled, X_train_cat_imputed))
X_test_processed = np.hstack((X_test_num_scaled, X_test_cat_imputed))

# Train the model
print("Training model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_processed, y_train)

# Evaluate the model
train_score = model.score(X_train_processed, y_train)
test_score = model.score(X_test_processed, y_test)
print(f"Train R² score: {train_score:.4f}")
print(f"Test R² score: {test_score:.4f}")

# Save the model and related files
print("Saving model files...")
joblib.dump(model, 'ml_model/model.joblib')
joblib.dump(scaler, 'ml_model/scaler.joblib')
joblib.dump(label_encoders, 'ml_model/label_encoders.joblib')
joblib.dump(feature_info, 'ml_model/feature_info.pkl')
joblib.dump(numerical_imputer, 'ml_model/numerical_imputer.joblib')
joblib.dump(categorical_imputer, 'ml_model/categorical_imputer.joblib')

print("Model training and saving completed successfully!") 