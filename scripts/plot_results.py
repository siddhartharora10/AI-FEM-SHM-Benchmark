import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

# -------- SELECT MODE --------
mode = "moderate"  # default

if len(sys.argv) > 1:
    mode = sys.argv[1]

if mode == "realistic":
    DATASET_PATH = "data/dataset_realistic.csv"
    fig_name = "confusion_realistic.png"
elif mode == "moderate":
    DATASET_PATH = "data/dataset_moderate.csv"
    fig_name = "confusion_moderate.png"
else:
    print("Use: moderate or realistic")
    sys.exit()

# -------- LOAD DATA --------
df = pd.read_csv(DATASET_PATH)

X = df.drop(columns=["zone"])
y = df["zone"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------- TRAIN MODEL --------
model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -------- CREATE RESULTS FOLDER --------
os.makedirs("results", exist_ok=True)

# -------- CONFUSION MATRIX --------
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title(f"Confusion Matrix ({mode})")
plt.savefig(f"results/{fig_name}")
plt.close()

print(f"Saved: results/{fig_name}")

# -------- ACCURACY BAR PLOT --------
# Hardcoded based on your results
accuracy_data = {
    "Moderate": 0.995,
    "Realistic": 0.44
}

plt.figure()
plt.bar(accuracy_data.keys(), accuracy_data.values())
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.savefig("results/accuracy_plot.png")
plt.close()

print("Saved: results/accuracy_plot.png")