import json
import pickle
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create sample student data
data = {
    "attendance": [90, 80, 75, 60, 95, 45, 85, 70, 55, 92,
                   88, 40, 78, 65, 98, 50, 82, 73, 60, 96],
    "study_hours": [5, 4, 3, 1, 6, 1, 5, 3, 2, 6,
                    5, 1, 4, 2, 7, 1, 4, 3, 2, 7],
    "marks": [85, 75, 65, 35, 95, 25, 88, 60, 40, 92,
              80, 20, 78, 50, 98, 30, 82, 68, 45, 96],
    "result": [1, 1, 1, 0, 1, 0, 1, 1, 0, 1,
               1, 0, 1, 0, 1, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)
df.to_csv("student_results.csv", index=False)

X = df[["attendance", "study_hours", "marks"]]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

with open("student_result_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("metrics.json", "w") as file:
    json.dump({"accuracy": accuracy}, file, indent=4)

print("Model training completed!")
print("Model accuracy:", round(accuracy, 4))
