import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------- Select dataset ----------
# Usage:
#   python3 scripts/train_model.py moderate
#   python3 scripts/train_model.py realistic

mode = "moderate"  # default

if len(sys.argv) > 1:
    mode = sys.argv[1].strip().lower()

if mode == "realistic":
    DATASET_PATH = "data/dataset_realistic.csv"
elif mode == "moderate":
    DATASET_PATH = "data/dataset_moderate.csv"
else:
    print("Unknown mode. Use 'moderate' or 'realistic'.")
    sys.exit(1)

# ---------- Check file exists ----------
if not os.path.exists(DATASET_PATH):
    print(f"Dataset not found: {DATASET_PATH}")
    sys.exit(1)

print(f"\nUsing dataset: {DATASET_PATH}")

# ---------- Load data ----------
df = pd.read_csv(DATASET_PATH)

if "zone" not in df.columns:
    print("Error: 'zone' column not found in dataset.")
    sys.exit(1)

X = df.drop(columns=["zone"])
y = df["zone"]

# ---------- Train-test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- Models ----------
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)

# ---------- Train ----------
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# ---------- Predict ----------
dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)

# ---------- Metrics ----------
dt_acc = accuracy_score(y_test, dt_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print("\n--- Results ---")
print(f"Decision Tree Accuracy: {dt_acc:.3f}")
print(f"Random Forest Accuracy: {rf_acc:.3f}")

print("\n--- Confusion Matrix (Random Forest) ---")
print(confusion_matrix(y_test, rf_pred))

print("\n--- Classification Report (Random Forest) ---")
print(classification_report(y_test, rf_pred))