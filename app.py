from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)


df = pd.read_csv("dataset/quizdata.csv")


le_difficulty = LabelEncoder()
le_multiplayer = LabelEncoder()
le_platform = LabelEncoder()
le_time = LabelEncoder()
le_genre = LabelEncoder()

df["difficulty"] = le_difficulty.fit_transform(df["difficulty"])
df["multiplayer"] = le_multiplayer.fit_transform(df["multiplayer"])
df["platform"] = le_platform.fit_transform(df["platform"])
df["time"] = le_time.fit_transform(df["time"])
df["genre"] = le_genre.fit_transform(df["genre"])

X = df.drop("genre", axis=1)
y = df["genre"]


model = DecisionTreeClassifier()
model.fit(X, y)

games_by_genre = {
    "Action": [
        {
            "name": "GTA V",
            "image": "gta.jpg",
            "trailer": "https://www.youtube.com/results?search_query=GTA+V+trailer"
        },
        {
            "name": "Call of Duty",
            "image": "cod.jpg",
            "trailer": "https://www.youtube.com/results?search_query=Call+of+Duty+trailer"
        },
        {
            "name": "God of War",
            "image": "gow.jpg",
            "trailer": "https://www.youtube.com/results?search_query=God+of+War+trailer"
        },
        {
            "name": "Assassin's Creed",
            "image": "ac.jpg",
            "trailer": "https://www.youtube.com/results?search_query=Assassins+Creed+trailer"
        }
    ],

    "casual": [
        {
            "name": "Fall Guys",
            "image": "fg.jpg",
            "trailer": "https://www.youtube.com/results?search_query=Fall+Guys+trailer"
        },
        {
            "name": "Subway Surfers",
            "image": "sb.jpg",
            "trailer": "https://www.youtube.com/results?search_query=Subway+Surfers+trailer"
        }
    ],

    "strategy": [
        {
            "name": "Clash of Clan",
            "image": "coc.jpg",
            "trailer": "https://www.youtube.com/results?search_query=Clash+ofClan+trailer"
        },
        {
            "name": "Clash Royale",
            "image": "cr.jpg",
            "trailer": "https://www.youtube.com/results?search_query=Clash+Royale+trailer"
        }
    ],

    "Sports": [
        {
            "name": "FIFA",
            "image": "fifa.jpg",
            "trailer": "https://www.youtube.com/results?search_query=FIFA+game+trailer"
        },
        {
            "name": "F1 25",
            "image": "f1.jpg",
            "trailer": "https://www.youtube.com/results?search_query=F1+25+trailer"
        }
    ]
}



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    difficulty = request.form["difficulty"]
    multiplayer = request.form["multiplayer"]
    platform = request.form["platform"]
    time = request.form["time"]

    input_data = [[
        le_difficulty.transform([difficulty])[0],
        le_multiplayer.transform([multiplayer])[0],
        le_platform.transform([platform])[0],
        le_time.transform([time])[0]
    ]]

    prediction = model.predict(input_data)
    genre = le_genre.inverse_transform(prediction)[0]

    games = games_by_genre.get(genre, [])
    return render_template("result.html", genre=genre, games=games)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)