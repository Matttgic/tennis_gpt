import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("🎾 Tennis GPT — Value Bet Finder")
st.write("Prédictions ML sur historique ATP/WTA")

# Sélection du circuit
tour = st.radio("Circuit", ["ATP", "WTA"])
model_path = f"models/tennis_model_{tour.lower()}.pkl"
model = joblib.load(model_path)

# Saisie des entrées
rank1 = st.number_input("Classement joueur 1", min_value=1, max_value=1000, value=25)
rank2 = st.number_input("Classement joueur 2", min_value=1, max_value=1000, value=50)
surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])
level = st.selectbox("Tournoi", ["G", "M", "A", "F"])
cote = st.number_input("Cote proposée pour joueur 1", value=2.0)

# Construction des features
features = {
    "rank_diff": rank2 - rank1,
    "surface_Clay": int(surface == "Clay"),
    "surface_Grass": int(surface == "Grass"),
    "surface_Hard": int(surface == "Hard"),
    "level_A": int(level == "A"),
    "level_F": int(level == "F"),
    "level_G": int(level == "G"),
    "level_M": int(level == "M")
}

X = pd.DataFrame([features])
proba = model.predict_proba(X)[0][1]
implied = 1 / cote

# Affichage résultats
st.metric("🔮 Probabilité victoire Joueur 1", f"{proba*100:.2f}%")

if proba > implied + 0.05:
    st.success("🎯 VALUE BET DÉTECTÉ !")
else:
    st.warning("⛔️ Pas de value ici.")
