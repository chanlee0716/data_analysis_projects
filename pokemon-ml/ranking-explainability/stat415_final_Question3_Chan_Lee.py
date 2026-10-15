"""

Question 3: Predictive Modeling

For this question, we will use the datasets
    question_3_4_5_pokemon_rankings_train and question_3_4_5_pokemon_rankings_test

In these datasets, 10 of my friends were shown hundreds of pokemon photos and asked to rank them.
Note that only the first 151 pokemon were used in this experiment.

This dataset has two potential outcome variables of interest.

    raw_ranking: an average score across the 10 friends. Ranges from 0.5-4.5
    tier_ranking: The raw ranking is binned to the closest integer. This integer is then converted to a grade.
        0.5 < raw_ranking < 1.49 --> 1.0 = C
        1.49 < raw_ranking < 2.49  --> 2.0 = B
        2.50 < raw_ranking < 3.49 --> 3.0 = A
        raw_ranking > 3.49 --> 4.0 = S


Your goal is to use the data features to predict a pokemon's ranking.

Dataset description:

    Pokemon_names: index variable. Not a feature. Don't use this to help with predictions.
    raw_rankings: outcome variable. Continuous. Average over 10 rankings of a photo specific photo.
    poke_tiers: outcome variable. discretized version of raw ranking. Treat as categorical. Ordinal.
    famous_artist: predictor variable. was the artist that made this picture well known?
    photo_brightness_level: predictor variable. how bright was the photo.
    photo_depth_of_field: predictor variable. what is the scale of the objects in the photo.
    is_pokemon_larger_than_a_burrito: predictor variable. returns true if the pokemon is larger than a reasonably large burrito and false else.
    cute_classification_conv_net_feature_14: predictor variable. I trained a conv net to identify cute pictures of pokemon. This is one of the features learned by the conv net.
    cute_classification_conv_net_feature_26: predictor variable. See above.
    photo_night_or_day: predictor variable. Does the photo happen at night or during the day.
    is_pokemon_moving: preditor variable. Is the shot dynamic? Is the pokemon likely moving in the frame.

"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score


def import_data(dataset_path):
    # anything else you want to do, do it here.
    # I would suggest normalizing the data appropiately.
    df = pd.read_csv(dataset_path)
    if "Pokemon_names" in df.columns:
        df = df.set_index("Pokemon_names")
    return df

def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Return a ColumnTransformer that imputes & encodes automatically."""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")),
         ("scaler", StandardScaler())]
    )

    categorical_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")),
         ("ohe", OneHotEncoder(handle_unknown="ignore"))]
    )

    return ColumnTransformer(
        [("num", numeric_pipe, numeric_cols),
         ("cat", categorical_pipe, categorical_cols)]
    )

######## Ridge Regression ########
def train_predictive_model_regression(train_df):
    y = train_df["raw_rankings"].values
    X = train_df.drop(columns=["raw_rankings", "poke_tiers"])

    # creates a pipeline that imputes, scales, and encodes the data
    model = Pipeline(
        [("prep", build_preprocessor(X)),
         ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 7)))]
    )
    model.fit(X, y)

    train_mse = mean_squared_error(y, model.predict(X))
    return model, train_mse


def test_predictive_model_regression(test_df, model):
    y = test_df["raw_rankings"].values
    X = test_df.drop(columns=["raw_rankings", "poke_tiers"])
    return mean_squared_error(y, model.predict(X))



######## Random Forest Classifier ########

def train_predictive_model_classification(train_df):
    """
    Train RandomForestClassifier on 'poke_tiers'.
    Returns
    model              : fitted Pipeline
    train_set_error    : 1 – accuracy  (mis-classification rate) on training data
    """
    y = train_df["poke_tiers"].values
    X = train_df.drop(columns=["poke_tiers", "raw_rankings"])

    model = Pipeline(
        [("prep", build_preprocessor(X)),
         ("rf", RandomForestClassifier(
                 n_estimators=300,
                 random_state=42,
                 class_weight="balanced"))]
    )
    model.fit(X, y)

    train_acc   = accuracy_score(y, model.predict(X))
    train_error = 1.0 - train_acc           # <-- error, not accuracy
    return model, train_error


def test_predictive_model_classification(test_df, model):
    """
    Evaluate on held-out data.
    Returns:
    error   : 1 – accuracy  (mis-classification rate) on test data
    """
    y = test_df["poke_tiers"].values
    X = test_df.drop(columns=["poke_tiers", "raw_rankings"])

    test_acc = accuracy_score(y, model.predict(X))
    return 1.0 - test_acc                   # <-- error, not accuracy


# Letter-to-number mapping per the prompt
TIER_TO_NUM = {"C": 1.0, "B": 2.0, "A": 3.0, "S": 4.0}

def _to_numeric(arr):
    """
    Helper: convert an array of tier labels or ints to numeric floats.
    Handles three cases:
      • already numeric (int/float)           → pass through as float
      • single letters 'C','B','A','S'        → map via TIER_TO_NUM
      • pandas Categorical with those levels → map as well
    """
    # If the dtype is numeric, just cast to float and return
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(float)

    # Otherwise, map each entry via TIER_TO_NUM
    return np.vectorize(TIER_TO_NUM.get)(arr)


def get_mse_of_classifier(test_set_x, test_set_y, predictive_model):
    """
    Parameters
    ----------
    test_set_x : pd.DataFrame or np.ndarray
        Feature matrix (NO raw_rankings column).
    test_set_y : array-like
        True numeric labels (1.0,2.0,3.0,4.0) *or* letter grades.
    predictive_model : fitted sklearn classifier
        Must have a .predict() method that outputs letter grades or ints.

    Returns
    -------
    mse : float
        Mean-squared error between numeric predictions and numeric targets.
    """
    # 1. Classifier predictions
    y_pred = predictive_model.predict(test_set_x)

    # 2. Convert both y_pred and y_true to numeric floats
    y_pred_num = _to_numeric(np.asarray(y_pred))
    y_true_num = _to_numeric(np.asarray(test_set_y))

    # 3. Mean-squared error
    mse = np.mean((y_true_num - y_pred_num) ** 2)
    return mse


######## Main ########

def main():
    dataset = import_data('question_3_4_5_pokemon_rankings_train.csv')
    test_dataset = import_data('question_3_4_5_pokemon_rankings_test.csv')

    """
    Part A:
        Using the MSE loss function, train a Ridge regression model
        that uses the dataset's features to predict 'raw_ranking.' 
        Evaluate this predictive model on the test set by reporting the 
        test set MSE. 
        
        Note 1: Be careful you don's use 'tier_ranking' in your predictor variables! 
        
    """
    ridge_model, ridge_train_mse = train_predictive_model_regression(dataset)
    ridge_test_mse = test_predictive_model_regression(test_dataset, ridge_model)


    """
        
    Part B: 
        For part B, we will treat predicting 'tier_ranking' as a classification problem. 
        Begin by converting 'tier_ranking' into a format that can be consumed by sci kit learn. 
        
        Using a random forrest classifier, train a predictive model 
        that uses the dataset features to predict 'tier_ranking' 
        Evaluate the predictive model on the test set by reporting the 
        test set Accuracy. 
        
        Note 1: Be careful you don's use 'raw_ranking' in your predictor variables! 
        
    
    """

    # predictive_model = train_predictive_model_classification(dataset)
    # test_error = test_predictive_model_classification(dataset.test_data_x, dataset.test_data_y,
    #                                                   predictive_model=predictive_model)
    rf_model, rf_train_error = train_predictive_model_classification(dataset)
    rf_test_error            = test_predictive_model_classification(test_dataset, rf_model)

    rf_train_acc = 1.0 - rf_train_error
    rf_test_acc  = 1.0 - rf_test_error

    """
    
    Part C: 
        We want to compare classification and regression on this problem. To do this, compute the test set 
        MSE for your trained random forest classifier model. 
        How does the model MSE compare to the MSE from Part A? 
        
        
        Note 1: You may assume the pokemon's letter grades map directly back onto the integer that was used
                to generate the ranking. For example, C would map back to 1.0 and S would map back to 4.0
                                
    """

    # predictive_model = train_predictive_model_classification(dataset)
    # test_error = get_mse_of_classifier(dataset.test_data_x, dataset.test_data_y, predictive_model=predictive_model)

    X_test = test_dataset.drop(columns=["poke_tiers", "raw_rankings"])
    y_test_raw = test_dataset["raw_rankings"].values

    rf_numeric_mse = get_mse_of_classifier(
        test_set_x=X_test,
        test_set_y=y_test_raw,
        predictive_model=rf_model
    )

    print("══════════════════════════════════════════════════════════════")
    print("PART A  (Ridge Regression on raw_rankings)")
    print(f"• train MSE = {ridge_train_mse:.3f}")
    print(f"• test  MSE = {ridge_test_mse:.3f}\n")

    print("PART B  – Random-Forest Classifier (poke_tiers)")
    print(f"• Train Accuracy : {rf_train_acc:.3f}   (error {rf_train_error:.3f})")
    print(f"• Test  Accuracy : {rf_test_acc:.3f}   (error {rf_test_error:.3f})\n")

    print("PART C  – RF classifier mapped back to numeric grade")
    print(f"• Test MSE : {rf_numeric_mse:.3f}\n")

    print("Comparison → Ridge MSE (Part A) vs. RF-numeric MSE (Part C)")
    print(f"• Ridge MSE : {ridge_test_mse:.3f}")
    print(f"• RF MSE    : {rf_numeric_mse:.3f}")
    print("════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
