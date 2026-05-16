import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Synthesise a banking credit dataset with structural risk patterns
np.random.seed(42)
num_customers = 2000

data = {
    'age': np.random.randint(18, 70, num_customers),
    'annual_income': np.random.randint(15000, 140000, num_customers),
    'credit_score': np.random.randint(300, 850, num_customers),
    'debt_to_income_ratio': np.random.uniform(0.05, 0.65, num_customers),
    'missed_payments_last_2_years': np.random.choice([0, 1, 2, 3], num_customers, p=[0.7, 0.18, 0.08, 0.04])
}

df = pd.DataFrame(data)

# Define empirical risk rules for the classification target (1 = High Risk, 0 = Low Risk)
risk_score = (
    (df['credit_score'] < 580) * 0.4 + 
    (df['debt_to_income_ratio'] > 0.45) * 0.3 + 
    (df['missed_payments_last_2_years'] > 1) * 0.3 +
    np.random.normal(0, 0.15, num_customers)
)
df['is_high_risk'] = (risk_score > 0.35).astype(int)

# 2. Split features and target vector
X = df.drop(columns=['is_high_risk'])
y = df['is_high_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Feature Scaling (Data Engineering)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Model Training via Random Forest Ensemble
print("Training predictive credit classification model...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# 5. Production Performance Evaluation
y_pred = model.predict(X_test_scaled)
print("\n--- Model Evaluation Matrix ---")
print(confusion_matrix(y_test, y_pred))
print("\n--- Detailed Classification Performance ---")
print(classification_report(y_test, y_pred))

# 6. Serialise production artifacts for deployment
joblib.dump(model, 'credit_risk_model.pkl')
joblib.dump(scaler, 'credit_feature_scaler.pkl')
print("\nProduction artifacts successfully saved ('credit_risk_model.pkl', 'credit_feature_scaler.pkl')")