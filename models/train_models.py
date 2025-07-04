from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from data.preprocess import load_data

def train_and_save_model(tour="ATP"):
    data = load_data(tour=tour)
    X = data.drop(columns=["label", "p1_rank", "p2_rank"])
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{tour} model accuracy: {acc:.2%}")

    filename = f"models/tennis_model_{tour.lower()}.pkl"
    joblib.dump(model, filename)
    print(f"✔️ Model saved to {filename}")

if __name__ == "__main__":
    train_and_save_model("ATP")
    train_and_save_model("WTA")
