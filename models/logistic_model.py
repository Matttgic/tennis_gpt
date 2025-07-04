# models/logistic_model.py
from sklearn.linear_model import LogisticRegression
import numpy as np

class TennisPredictor:
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_proba(self, match_features):
        return self.model.predict_proba([match_features])[0]
