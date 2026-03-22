import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("dataset/quizdata.csv")

# Encode categorical data
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

# Features and label
X = df.drop("genre", axis=1)
y = df["genre"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Test prediction (sample input)
sample_input = [[
    le_difficulty.transform(["Hard"])[0],
    le_multiplayer.transform(["No"])[0],
    le_platform.transform(["PC"])[0],
    le_time.transform([">1hr"])[0]
]]

prediction = model.predict(sample_input)
predicted_genre = le_genre.inverse_transform(prediction)

print("Predicted Genre:", predicted_genre[0])
