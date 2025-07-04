from data.stats_fetcher import calculate_winrate, get_head_to_head
from utils.helpers import implied_probability, is_value_bet
from models.logistic_model import TennisPredictor
import numpy as np

# Exemples fictifs pour tester
player1_id = "1234"
player2_id = "5678"

winrate1 = calculate_winrate(player1_id)
winrate2 = calculate_winrate(player2_id)
h2h = get_head_to_head(player1_id, player2_id)
h2h_score = h2h.get("wins_player1", 0) / (h2h.get("total_matches", 1))

# Features = [winrate1, winrate2, h2h_score]
features = np.array([winrate1, winrate2, h2h_score])

# On suppose que le modèle est déjà entraîné
predictor = TennisPredictor()
predictor.train(X_train=[[0.6, 0.4, 0.75], [0.5, 0.5, 0.5]], y_train=[1, 0])  # Exemple de pré-training

proba = predictor.predict_proba(features)
proba_p1 = proba[1]  # Proba victoire joueur 1

cote = 2.0
if is_value_bet(proba_p1, cote):
    print(f"🎯 PARI CONSEILLÉ : Parier sur Joueur 1 avec une probabilité estimée de {proba_p1:.2%} et une cote de {cote}")
