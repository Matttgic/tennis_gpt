import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("🎾 Tennis GPT — Value Bet Finder")
st.write("Prédictions ML sur historique ATP/WTA")

tour = st.radio("Circuit", ["ATP", "WTA"])
model_path = f"models/tennis_model_{tour.lower()}.pkl"
model = joblib.load(model_path)

# Saisie utilisateur
rank1 = st.number_input("Classement joueur 1", min_value=1, max_value=1000, value=25)
rank2 = st.number_input("Classement joueur 2", min_value=1, max_value=1000, value=50)
surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])
level = st.selectbox("Tournoi", ["G", "M", "A", "F"])
cote = st.number_input("Cote proposée pour joueur 1", value=2.0)

# Toutes les features possibles dans le modèle
all_features = [
    "rank_diff",
    "surface_Clay", "surface_Grass", "surface_Hard",
    "level_A", "level_F", "level_G", "level_M"
]

# Construction manuelle avec remplissage des colonnes manquantes
features = {
    "rank_diff": rank2 - rank1,
    "surface_Clay": 0,
    "surface_Grass": 0,
    "surface_Hard": 0,
    "level_A": 0,
    "level_F": 0,
    "level_G": 0,
    "level_M": 0
}
features[f"surface_{surface}"] = 1
features[f"level_{level}"] = 1

X = pd.DataFrame([features])[all_features]  # On s’assure que l’ordre est bon

proba = model.predict_proba(X)[0][1]
implied = 1 / cote

st.metric("🔮 Probabilité victoire Joueur 1", f"{proba*100:.2f}%")

if proba > implied + 0.05:
    st.success("🎯 VALUE BET DÉTECTÉ !")
else:
    st.warning("⛔️ Pas de value ici.") 
