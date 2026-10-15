"""

Question 1: EDA

For this question, we will use the dataset question_1_pokemon_cuteness_data.csv
In this dataset, people were shown pictures of pokemon and asked if the pokemon looks cute.
The outcome variable is the column ‘is_cute,’ which is a binary.

For all questions in this exam, treat the pokemon's name as an index.
Do not treat it as a feature.


Dataset description:

    pokemon_names: index variable. Not a feature. Don’t use this to help with predictions.
    pokemon_cute: outcome variable. Binary. Whether or not this pokemon picture was voted cute.
    multiple_pokes_in_pic: predictor variable. Binary. Whether or not there are multiple pokemon in the picture.
    photo_contrast_level: predictor variable. Ordinal. A measure of the contrast level of the photo, ranging from low to high.
    pokes_size_in_frame: predictor variable. Ordinal. The size of the pokemon in the frame, measured in pixels.
    looks_fluffy: predictor variable. Binary. Whether or not the pokemon in the picture looks fluffy.
    original_image_source: predictor variable. Nominal. The original source of the image.


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def import_data(dataset_path):
    # read the csv file using pd
    df = pd.read_csv(dataset_path)

    # make sure that pokemon_names is the index
    if "pokemon_names" in df.columns:
        df = df.set_index("pokemon_names")
    
    # Replace blanks (and optionally NaNs) in photo_contrast_level
    df["photo_contrast_level"] = (
        df["photo_contrast_level"]
        .astype(str)            # make sure it's string
        .str.strip()            # remove whitespace
        .replace({'': 'Missing'})
    )
    print(f"length or original df: {len(df)}")

    # drop rows where photo_contrast_level is missing
    df = df[df["photo_contrast_level"] != "Missing"]

    print(f"length of df after dropping missing values: {len(df)}")
    return df


def find_cutest_mon_by_quantity(df, target_col="pokemon_cute"):
    """Return name of Pokémon with the most ‘cute’ votes and that vote count."""
    # Boolean mask for cute rows (works for 1/0 or True/False)
    cute_mask = df[target_col] == 1

    # value_counts on the index of cute rows → count per Pokémon
    cute_counts = df.loc[cute_mask].index.value_counts()

    if cute_counts.empty:
        return None, 0

    cutest_mon = cute_counts.idxmax()
    num_votes  = int(cute_counts.max())
    return cutest_mon, num_votes



def find_cutest_feature_by_ratio(df, target_col="pokemon_cute"):
    """
    Determine which individual feature *value* is most strongly associated
    with a Pokémon being labeled cute.

    For every predictor column (binary, ordinal, or nominal) and every
    distinct value that appears in that column:

        1. Form the subset of rows in which the column equals that value.
        2. Compute the conditional probability
               P(cute | feature == value)  =  (# cute in subset) / (size of subset)
        3. Track the feature-value pair with the highest conditional probability.

    Returns
    -------
    best_feature : str
        Name of the predictor column.
    best_value : scalar
        The specific value in that column that maximizes P(cute | feature).
    best_ratio : float
        The maximum conditional probability itself.
    """

    best_feature = None   # name of column with the highest ratio
    best_value   = None   # specific category / value in that column
    best_ratio   = -1.0   # highest P(cute | feature) found so far

    # ------------------------------------------------------------------
    # Iterate over every column (except the outcome variable)
    # ------------------------------------------------------------------
    for col in df.columns:
        if col == target_col:
            continue  # skip the label column itself

        # ------------------------------------------------------------------
        # For each distinct value in this column, compute the ratio
        # ------------------------------------------------------------------
        # Drop NaNs before calling .unique() to avoid treating NaN as a value
        for val in df[col].dropna().unique():

            # print(f"col: {col}, val: {val}")

            # Subset where the feature equals this value
            subset = df[df[col] == val]

            print(f"length of subset: {col} {val} {len(subset)}")

            # Guard: skip if for some reason subset is empty
            if subset.empty:
                continue

            # Mean of a 0/1 column equals the proportion of 1's
            ratio = subset[target_col].mean()

            # ------------------------------------------------------------------
            # Keep track of the best (feature, value) pair observed so far
            # ------------------------------------------------------------------
            if ratio > best_ratio:
                best_ratio = ratio
                best_feature = col
                best_value = val

    # ----------------------------------------------------------------------
    # Return the triple: (feature name, feature value, conditional probability)
    # ----------------------------------------------------------------------
    return best_feature, best_value, best_ratio




def main():
    dataset_path = 'question_1_pokemon_cuteness_data.csv'
    dataset = import_data(dataset_path)

    # print((dataset["photo_contrast_level"] == ' ').sum())

    """

    Part A:

        What pokemon had the largest quantity of votes for being cute? How many people found it cute?
        For example, if Mew has 5 votes for being cute and Magmar has 2 votes for being cute, then we
        would say that Mew has the largest quantity of votes for being cute with 5 votes.

    """

    cutest_mon = find_cutest_mon_by_quantity(dataset)
    print(F"Cutest mon is{cutest_mon}")

    """

    Part B:
        
        What feature is most strongly correlated with a pokemon being labeled as 'cute?'
        In other words, what feature maximizes P(cute | f)?

    """

    cutest_feature = find_cutest_feature_by_ratio(dataset)
    print(cutest_feature)


if __name__ == "__main__":
    main()