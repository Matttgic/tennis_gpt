# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load("models/tennis_model.pkl")

st.title("🎾 Tennis GPT — Value Bet Finder")
st.write("Base ML sur historiques ATP/WTA")

rank1 = st.number_input("Classement joueur 1", min_value=1, max_value=1000, value=25)
rank2 = st.number_input("Classement joueur 2", min_value=1, max_value=1000, value=50)
surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])
level = st.selectbox("Tournoi", ["G", "M", "A", "F"])

# Dummy encoding
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
prob = model.predict_proba(X)[0][1]

st.metric("🔮 Probabilité victoire Joueur 1", f"{prob*100:.2f}%")

cote = st.number_input("Cote proposée pour joueur 1", value=2.0)
implied = 1 / cote
if prob > implied + 0.05:
    st.success("🎯 VALUE BET DÉTECTÉ")
else:
    st.warning("⛔️ Pas de value ici")
