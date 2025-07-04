# data/stats_fetcher.py
import requests

HEADERS = {
    "x-rapidapi-host": "tennisapi1.p.rapidapi.com",
    "x-rapidapi-key": "1df93a4239msh5776d5f2c3b3a91p147a3ejsnea4c93adaca3"
}

def get_player_last_matches(player_id, n=10):
    url = f"https://tennisapi1.p.rapidapi.com/api/tennis/players/{player_id}/matches"
    response = requests.get(url, headers=HEADERS)
    matches = response.json().get("matches", [])
    return matches[:n]

def calculate_winrate(player_id):
    matches = get_player_last_matches(player_id)
    wins = sum(1 for m in matches if m.get("outcome") == "win")
    return wins / len(matches) if matches else 0.5

def get_head_to_head(player1_id, player2_id):
    url = f"https://tennisapi1.p.rapidapi.com/api/tennis/head-to-head/{player1_id}/{player2_id}"
    response = requests.get(url, headers=HEADERS)
    return response.json().get("summary", {})
