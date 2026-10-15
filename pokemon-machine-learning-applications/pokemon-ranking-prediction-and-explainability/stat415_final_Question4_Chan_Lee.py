"""

Question 4: Explainability

For this question, we will use the datasets
    question_3_4_5_pokemon_rankings_train and question_3_4_5_pokemon_rankings_test

See question 3 for a dataset description.

Your goal is to find the important features that determine a pokemon's tier_ranking.
You will also need to explain the mistakes your predictive model is making.

"""

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def import_data(dataset_path):
    """ Same as the one for question 3"""
    # anything else you want to do, do it here.
    # I would suggest normalizing the data appropiately.
    df = pd.read_csv(dataset_path)
    if "Pokemon_names" in df.columns:
        df = df.set_index("Pokemon_names")
    return df

def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Same as the one for question 3

    Return a ColumnTransformer that imputes & encodes automatically.
    """
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


# def train_predictive_model(train_dataset):
#     # Train a model to take pokemon features and predict the ranking.
#     trained_model = None
#     train_set_error = None
#     return trained_model, train_set_error

def train_predictive_model(train_df):
    """
    Same as the one for question 3

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





################################################################################
########### Everything below this line is new ##################################
################################################################################


# def get_important_features(preditive_model):
#     important_features = predictive_model.important_features
#     ranked_features = important_features.rank
#     return ranked_features

def get_important_features(predictive_model):
    # 1. Extract the fitted preprocessor (ColumnTransformer) and the fitted RF
    prep = predictive_model.named_steps["prep"]   # ColumnTransformer
    rfc  = predictive_model.named_steps["rf"]     # RandomForestClassifier

    # transformers = prep.transformers_
    # numeric_cols = []
    # categorical_cols = []
    # for name, transformer, cols in transformers:
    #     if name == "num":
    #         numeric_cols = list(cols)
    #     elif name == "cat":
    #         categorical_cols = list(cols)


    # 2. Ask the preprocessor for the post‐transform feature names:
    feature_names = prep.get_feature_names_out()

    # 3. Pull out the importances from the RF
    importances = rfc.feature_importances_

    # 4. Build a Series, sort descending, return top_n
    feat_series = pd.Series(importances, index=feature_names)
    feat_series = feat_series.sort_values(ascending=False)
    return feat_series.iloc[:5]


def plt_cm_matrix(labels_test, y_hat, forest):
    # 1. Get the classifier’s known class ordering (original string labels)
    classes = list(forest.classes_)  # e.g. ['C', 'B', 'A', 'S']

    # 2. Compute the confusion matrix, specifying labels=classes to preserve ordering
    cm = confusion_matrix(labels_test, y_hat, labels=classes)

    # 3. Convert to a DataFrame so the heatmap shows readable row/col labels
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)

    # 4. Plot as a heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        df_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False
    )
    plt.xlabel("Predicted Tier")
    plt.ylabel("True Tier")
    plt.title("Confusion Matrix of Pokémon Tier Predictions")
    plt.show()

    return cm


def main():
    train_dataset = import_data('question_3_4_5_pokemon_rankings_train.csv')
    test_dataset = import_data('question_3_4_5_pokemon_rankings_test.csv')

    """
    Part A:
        Using your random forest classifier from question 3, 
        what are the two most important features that the model is selecting on
        to make its ranking? 
        
    """
    trained_model, _ = train_predictive_model(train_dataset)
    important_features = get_important_features(trained_model)
    # Top feature:
    top_name  = important_features.index[0]
    top_imp   = important_features.iloc[0]
    print(f"Top feature: {top_name}   (importance = {top_imp:.6f})")

    # Second feature:
    second_name = important_features.index[1]
    second_imp  = important_features.iloc[1]
    print(f"Second  feature: {second_name}   (importance = {second_imp:.6f})")

    # Top feature:
    top_name  = important_features.index[2]
    top_imp   = important_features.iloc[2]
    print(f"Third feature: {top_name}   (importance = {top_imp:.6f})")

    # Second feature:
    second_name = important_features.index[3]
    second_imp  = important_features.iloc[3]
    print(f"Fourth feature: {second_name}   (importance = {second_imp:.6f})")

    # Top feature:
    top_name  = important_features.index[4]
    top_imp   = important_features.iloc[4]
    print(f"Fifth feature: {top_name}   (importance = {top_imp:.6f})")



    """
    
    Part B: 
        Using the same classifier from part A, look at the classifier confusion matrix. 
        What rankings is the classifier getting wrong? 
        Your answer should have the form: 
        `The classifier commonly mistakes pokemon with ranking XXX as having ranking YYY.' 
        Please use the original tier_ranking labels from the dataset in your answer.  
        Do not give your answer in terms of a numeric scale or in terms of basis vectors.  
        
    """
    # Prepare test features and labels
    X_test = test_dataset.drop(columns=["poke_tiers", "raw_rankings"])
    y_true = test_dataset["poke_tiers"].values

    # Predict on the test set
    y_hat = trained_model.predict(X_test)

    # Extract the raw RandomForestClassifier to get classes_
    forest = trained_model.named_steps["rf"]

    # Plot confusion matrix (function defined earlier) and get the array
    cm = plt_cm_matrix(y_true, y_hat, forest)

    # Identify the largest off-diagonal entry in cm
    classes = list(forest.classes_)  # e.g. ['C','B','A','S']
    n = len(classes)
    max_count = 0
    mistaken_true = None
    mistaken_pred = None

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if cm[i, j] > max_count:
                max_count = cm[i, j]
                mistaken_true = classes[i]
                mistaken_pred = classes[j]

    # Print the most common misclassification
    if max_count > 0:
        print(
            f"\nThe classifier commonly mistakes Pokémon with ranking "
            f"{mistaken_true!r} as having ranking {mistaken_pred!r} "
            f"({max_count} cases)."
        )
    else:
        print("\nThe classifier made no mistakes on the test set.")

    print(f"Number of Pokemon misclassified as {mistaken_pred!r} when they were actually {mistaken_true!r}: {max_count}")



if __name__ == "__main__":
    main()

    # # ---------------------------------------
    # # 1. Load the training dataset
    # # ---------------------------------------
    # df = import_data('question_3_4_5_pokemon_rankings_train.csv')

    # # ---------------------------------------
    # # 2. Ensure feature_26 is numeric
    # # ---------------------------------------
    # print(df['cute_classification_conv_net_feature_26'].unique())
    # # Example output might be: array(['zero', 'non-zero'], dtype=object)

    # # ---------------------------------------
    # # 3. Plot distribution of feature_26
    # # ---------------------------------------
    # plt.figure(figsize=(8, 4))
    # sns.histplot(df['cute_classification_conv_net_feature_26'], bins=30, kde=True)
    # plt.title('Distribution of conv_net_feature_26')
    # plt.xlabel('Feature 26 Value')
    # plt.ylabel('Count')
    # plt.show()

    # # ---------------------------------------
    # # 4. Boxplot: feature_26 by tier
    # # ---------------------------------------
    # plt.figure(figsize=(8, 5))
    # sns.boxplot(
    #     x='poke_tiers',
    #     y='cute_classification_conv_net_feature_26',
    #     data=df,
    #     order=['C', 'B', 'A', 'S']
    # )
    # plt.title('Feature 26 by Pokemon Tier')
    # plt.xlabel('Tier')
    # plt.ylabel('Feature 26 Value')
    # plt.show()

    # # ---------------------------------------
    # # 5. Scatter: feature_26 vs raw_rankings
    # # ---------------------------------------
    # plt.figure(figsize=(8, 5))
    # sns.scatterplot(
    #     x='cute_classification_conv_net_feature_26',
    #     y='raw_rankings',
    #     data=df
    # )
    # plt.title('Feature 26 vs Raw Ranking')
    # plt.xlabel('Feature 26 Value')
    # plt.ylabel('Raw Rankings')
    # plt.show()

    # # ---------------------------------------
    # # 6. Compute correlation
    # # ---------------------------------------
    # corr_value = df['cute_classification_conv_net_feature_26'].corr(df['raw_rankings'])
    # print(f"Correlation between feature_26 and raw_rankings: {corr_value:.3f}")

