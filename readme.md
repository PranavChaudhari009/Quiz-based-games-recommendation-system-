# Quiz Based Games Recommendation System

Quiz-based Flask web app that recommends games from a predicted genre using a small scikit-learn decision tree model.

## Features

- Quiz-based input form
- Genre prediction with scikit-learn
- Multiple game recommendations per result
- Game cards with images and trailer links
- Try-again flow to retake the quiz

## Tech Stack

- Python
- Flask
- pandas
- scikit-learn
- HTML/CSS

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

The app runs on `http://127.0.0.1:5000` by default.

## Deploy on Render

This repo includes `render.yaml`, so Render can create the service with the correct commands automatically.

1. Push this project to GitHub.
2. In Render, select `New +` and then `Blueprint`.
3. Connect the GitHub repository.
4. Render will detect `render.yaml`.
5. Click deploy.

Render uses:

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn --bind 0.0.0.0:$PORT app:app`

## Notes

- The dataset is loaded from `dataset/quizdata.csv`.
- The app supports the `PORT` environment variable for hosting platforms like Render.
