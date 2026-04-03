<div align="center">

# Movie Recommendation System

<p>
	A movie recommendation project built with Python, pandas, scikit-learn, and Streamlit.
	It supports data cleaning, content-based filtering, collaborative filtering, and an interactive web app.
</p>

<p>
	<img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python badge" />
	<img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit badge" />
	<img src="https://img.shields.io/badge/scikit--learn-Recommender-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn badge" />
	<img src="https://img.shields.io/badge/Status-Ready-2EA44F?style=for-the-badge" alt="Status badge" />
</p>

</div>

---

## Overview

This project builds a complete movie recommendation workflow from the MovieLens-style CSV dataset stored in `dataset/`.

It includes:

- Data loading and cleaning
- Genre and tag feature engineering
- Content-based recommendations using TF-IDF and cosine similarity
- Collaborative filtering with SVD
- Automatic fallback to `TruncatedSVD` when `scikit-surprise` is unavailable
- Streamlit app for interactive movie recommendations
- Exported CSV artifacts and generated plots

## Key Note

> The Streamlit app does **not** require any `.pkl` file to run.
> It loads the CSV files from `dataset/` and builds the recommender automatically at startup.

## Tech Stack

<table>
	<tr>
		<td><b>Language</b></td>
		<td>Python</td>
	</tr>
	<tr>
		<td><b>Data Processing</b></td>
		<td>pandas, numpy</td>
	</tr>
	<tr>
		<td><b>Machine Learning</b></td>
		<td>scikit-learn, optional scikit-surprise</td>
	</tr>
	<tr>
		<td><b>Visualization</b></td>
		<td>matplotlib, seaborn</td>
	</tr>
	<tr>
		<td><b>UI</b></td>
		<td>Streamlit</td>
	</tr>
</table>

## Project Structure

```text
Movie data/
|-- app.py
|-- run_project.py
|-- requirements.txt
|-- README.md
|-- movie_model.ipynb
|-- dataset/
|   |-- movies.csv
|   |-- ratings.csv
|   |-- tags.csv
|   |-- links.csv
|-- src/
|   |-- content_recommender.py
|   |-- collaborative_recommender.py
|   |-- data_pipeline.py
|   |-- visualize.py
|-- artifacts/
|-- plots/
```

## Features

### Content-Based Recommendation

- Builds movie content from genres and user tags
- Converts text features into TF-IDF vectors
- Uses cosine similarity to recommend similar movies
- Supports flexible title matching for user input

### Collaborative Filtering

- Trains an SVD-based recommender from user ratings
- Uses `Surprise SVD` when available
- Falls back to `TruncatedSVD` from scikit-learn on unsupported environments
- Predicts user-movie ratings and generates unseen movie recommendations

### Streamlit App

- Loads movie and rating data directly from `dataset/`
- Builds the content recommender at runtime
- Lets users select a movie title and request recommendations
- Shows recommended titles with genre, similarity score, and average rating

### Analysis Outputs

- Top popular movies CSV
- Top rated movies CSV
- Collaborative recommendation CSV
- Rating distribution plot
- Top genres plot
- User-movie heatmap plot

## Installation

```bash
pip install -r requirements.txt
```

## Run the Full Pipeline

```bash
python run_project.py
```

This script will:

- Load and clean the dataset
- Generate top popular and top rated movie reports
- Train the content-based recommender
- Train the collaborative filtering model
- Save CSV outputs in `artifacts/`
- Save visualizations in `plots/`

## Run the Streamlit App

```bash
streamlit run app.py
```

After startup, open the local URL shown in the terminal.

## Python Dependencies

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
joblib
```

## Output Files

Generated files may include:

- `artifacts/top_popular_movies.csv`
- `artifacts/top_rated_movies.csv`
- `artifacts/collaborative_recommendations.csv`
- `artifacts/content_model.pkl`
- `artifacts/svd_model.pkl`
- `plots/rating_distribution.png`
- `plots/top_genres.png`
- `plots/user_movie_heatmap.png`

## Notes

- The app itself does not depend on saved `.pkl` files.
- The `.pkl` files are optional pipeline outputs created by `run_project.py`.
- If `scikit-surprise` is not installed or does not build on your Python version, the collaborative recommender still works with the scikit-learn fallback.

## Future Improvements

- Add poster thumbnails and richer movie metadata in the UI
- Add search-as-you-type movie lookup
- Add filtering by genre, year, or rating threshold
- Deploy the Streamlit app online

## Author

<div align="center">
	Built for a movie recommendation final project.
</div>
