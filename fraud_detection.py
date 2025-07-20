import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("fraud_transactions.csv")

print("📄 First 5 rows:")
print(df.head())

print("\n📊 Dataset Info:")
print(df.info())

print("\n🔍 Fraud Distribution:")
print(df['is_fraud'].value_counts())

df_encoded = pd.get_dummies(df, columns=["location", "device_type"])

X = df_encoded.drop(["transaction_id", "is_fraud"], axis=1)
y = df_encoded["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("✅ Data Prepared!")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("✅Model Trained!")

y_pred =model.predict(X_test)

print("🔍 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))