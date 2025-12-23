# 🎵 Song Popularity Prediction using Machine Learning

A complete **end-to-end machine learning project** that predicts a song’s popularity based on artist information, release year, and album characteristics. The project demonstrates **real-world ML practices** such as stratified splitting, preprocessing pipelines, and model persistence.

---

## 🚀 Project Overview

Predicting song popularity is a challenging regression problem due to the influence of external factors like marketing, virality, and listener behavior. This project focuses on learning patterns **purely from structured metadata** using a robust ML pipeline.

Key highlights:

* Cleaned and engineered real-world Spotify-like data
* Used **StratifiedShuffleSplit** for time-aware sampling
* Built reusable **Scikit-learn Pipelines**
* Trained a **RandomForestRegressor**
* Saved trained model and pipeline for reuse

---

## 🧠 Features Used

### Numerical Features

* Track Popularity
* Artist Popularity
* Arist followers
* Liveness
* album total tracks
* Release year
* Explicit content (binary encoded)

### Categorical Features

* `artist_genres` (Top 30 genres + `other`)
* `album_type`

---

## 🛠️ Tech Stack

* **Python**
* **Pandas & NumPy** – Data manipulation
* **Scikit-learn** – ML models & pipelines
* **Joblib** – Model persistence

---

## ⚙️ Machine Learning Pipeline

The preprocessing pipeline ensures consistency between training and inference:

* **Numerical Pipeline**

  * Median imputation
  * Standard scaling

* **Categorical Pipeline**

  * OneHotEncoding with `handle_unknown="ignore"`

Both pipelines are combined using `ColumnTransformer`.

---

## 📊 Model

* **Algorithm:** RandomForestRegressor
* **Why Random Forest?**

  * Handles non-linearity well
  * Robust to outliers
  * Strong baseline for tabular data

The model predicts popularity scores (0–100). Minor prediction deviations (±1–5) are expected and acceptable due to the noisy nature of popularity metrics.

---

## 🗂️ Project Structure

```
├── Songs.csv                 # Raw dataset
├── testing_data.csv          # Stratified test split
├── Predictions.csv           # Final predictions
├── model.py                  # Main training & inference script
└── Process_with_explaination.ipynb                 # Complete Process explained and visualized
```

---

## ▶️ How to Run

### 1️⃣ Install dependencies

```bash
pip install pandas numpy scikit-learn joblib
```

### 2️⃣ Train the model

```bash
python model.py
```

This will:

* Clean the data
* Split it using stratified sampling
* Train the model
* Save the model and pipeline

### 3️⃣ Generate predictions

Re-running the script will:

* Load saved artifacts
* Predict popularity on test data
* Save results to `Predictions.csv`

---

## 📈 Results

* Predictions are mostly close to actual values
* Small decimal differences (±0.1–0.2) indicate strong learning
* Larger differences (±5) are expected due to unobserved real-world factors

This performance is **solid for a metadata-only popularity model**.

---

## 📌 Learning Outcomes

* Real-world data cleaning
* Handling high-cardinality categorical features
* Proper ML pipeline design
* Model persistence and reuse

---

## 🙌 Acknowledgements

This project was built as part of a hands-on learning journey into **Machine Learning and Data Science**.

---

## Dataset Credits
The data used in this project is publicly available on kaggle - https://www.kaggle.com/datasets/wardabilal/spotify-global-music-dataset-20092025/data

---

⭐ If you found this project helpful, consider giving it a star!
