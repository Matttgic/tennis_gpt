# models/train_model.py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data.preprocess import load_atp_data

data = load_atp_data()
X = data.drop(columns=["label", "p1_rank", "p2_rank"])
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

import joblib
joblib.dump(model, "models/tennis_model.pkl")
