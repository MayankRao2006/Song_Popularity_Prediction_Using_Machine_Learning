import pandas as pd
import numpy as np
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "Model.pkl"
PIPELINE = "Pipeline.pkl"

def build_pipeline(num_attrs, cat_attrs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    full_pipeline = ColumnTransformer([
    ("nums", num_pipeline, num_attrs),
    ("cats", cat_pipeline, cat_attrs)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    df = pd.read_csv("Songs.csv")
    
    # Adding Strata
    df["album_release_date"] = df["album_release_date"].astype(str)
    df["album_release_date"] = pd.to_datetime(df["album_release_date"], dayfirst=True, errors="coerce")
    df["release_year"] = df["album_release_date"].dt.year
    
    # Changing explicit column's True and False to numericals
    df["explicit"] = df["explicit"].map({False:0, True:1})

    # Preparing the data
    df.drop(["track_id", "track_name", "artist_name", "album_id", "album_name"], axis=1, inplace=True)

    # Removing data which only comes once
    df.dropna(subset=["release_year"], inplace=True)
    df = df[df["release_year"] != 1963.0]
    df = df.reset_index(drop=True)

    # Handling missing values in artist_genres column
    df["artist_genres"] = df["artist_genres"].apply(lambda x: "unknown" if x == '[]' else x)
    top_genres = df["artist_genres"].value_counts().nlargest(30).index
    df["artist_genres"] = df["artist_genres"].apply(lambda x:x if x in top_genres else 'other')

    # Splitting the data
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df["release_year"]):
        train_set = df.loc[train_index]
        test_set = df.loc[test_index]

    test_set.to_csv("testing_data.csv",index=False)

    # Training the model
    features = train_set.drop(["track_popularity", "album_release_date"], axis=1) # album_release_date is only used to extract release_year
    labels = train_set["track_popularity"].copy()

    num_attrs = features.drop(["artist_genres", "album_type"], axis=1).columns.to_list()
    cat_attrs = ["artist_genres", "album_type"]

    pipeline = build_pipeline(num_attrs, cat_attrs)
    prepared_data = pipeline.fit_transform(features)

    model = RandomForestRegressor()
    model.fit(prepared_data, labels)
    
    # Save the Model and the pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE)

    print("Model and pipeline have been successfully trained and saved.")

else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE)

    df = pd.read_csv("testing_data.csv")
    features = df.drop(["track_popularity", "album_release_date"], axis=1)
    labels = df["track_popularity"].copy()

    prepared_data = pipeline.transform(features)
    predictions = model.predict(prepared_data)

    df["Predicted_track_popularity"] = predictions
    df.to_csv("Predictions.csv", index=False)
    print("Predictions have been made and saved to predictions.csv")