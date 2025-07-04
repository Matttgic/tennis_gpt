# data/preprocess.py
import pandas as pd
import glob

def load_atp_data(path="data/"):
    files = glob.glob(path + "atp_matches_*.csv")
    df = pd.concat([pd.read_csv(f) for f in files])
    df = df[["winner_rank", "loser_rank", "surface", "tourney_level"]]
    df = df.dropna()
    df["winner"] = 1
    df["loser"] = 0

    winner_df = df[["winner_rank", "loser_rank", "surface", "tourney_level", "winner"]]
    loser_df = df[["loser_rank", "winner_rank", "surface", "tourney_level", "loser"]]
    winner_df.columns = ["p1_rank", "p2_rank", "surface", "level", "label"]
    loser_df.columns = ["p1_rank", "p2_rank", "surface", "level", "label"]

    data = pd.concat([winner_df, loser_df])
    data["rank_diff"] = data["p2_rank"] - data["p1_rank"]
    data = pd.get_dummies(data, columns=["surface", "level"])
    return data
