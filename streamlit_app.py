import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("🎾 Tennis GPT — Value Bet Finder")
st.write("Prédictions ML sur historique ATP/WTA")

# Choix du circuit
tour = st.radio("Circuit", ["ATP", "WTA"])
model_path = f"models/tennis_model_{tour.lower()}.pkl"
model = joblib.load(model_path)

# 🔧 Récupérer dynamiquement les colonnes utilisées à l'entraînement
if hasattr(model, "feature_names_in_"):
    all_features = list(model.feature_names_in_)
else:
    st.error("Le modèle ne contient pas d'informations sur les colonnes d'entraînement.")
    st.stop()

# Inputs utilisateur
rank1 = st.number_input("Classement joueur 1", min_value=1, max_value=1000, value=25)
rank2 = st.number_input("Classement joueur 2", min_value=1, max_value=1000, value=50)
surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])
level = st.selectbox("Tournoi", ["G", "M", "A", "F"])
cote = st.number_input("Cote proposée pour joueur 1", value=2.0)

# Création du vecteur features
features = {name: 0 for name in all_features}
features["rank_diff"] = rank2 - rank1
features[f"surface_{surface}"] = 1
features[f"level_{level}"] = 1

X = pd.DataFrame([features])[all_features]
proba = model.predict_proba(X.to_numpy())[0][1]
implied = 1 / cote

st.metric("🔮 Probabilité victoire Joueur 1", f"{proba*100:.2f}%")
if proba > implied + 0.05:
    st.success("🎯 VALUE BET DÉTECTÉ !")
else:
    st.warning("⛔️ Pas de value ici.") 
