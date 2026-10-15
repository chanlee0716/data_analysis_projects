"""

Question 5: Towards the future

For this question, we will use the dataset question_3_4_and_5_pokemon_ranking_data.csv
See question 3 for a dataset description.

After completing questions 3 and 4, you think that perhaps your predictive models
could be improved if you had information about the pokemon's habitats.
Maybe people really like pokemon that live in the mountains such a vulpix and dislike forest pokemon (bugs).
It's worth a try.

Sadly, this data is not easily available.

To overcome this limitation, we will ask ChatGPT to generate the data for us,
then we will incorporate it into our predictive modeling pipeline.

Please use https://chatgpt.com/ to help with this question. 
"""


chat_gpt_query = """

For each of the following Pokemons, tell me their habitat. Please return your answer as a csv. 
This is very important. It needs to be a CSV with line breaks like the example below. 

You may only choose from the following habitats:

'Grassland', 'Desert', 'Freshwater', 'Urban', 'Cave', 'Ocean', 'Mountain', 'Sky', 'Snow'

Example: 

Bulbasaur, Grassland
Squirtle, Freshwater
Wartortle, Freshwater
Caterpie, Grassland
Metapod, Grassland
Geodude, Mountain
Magnemite, Urban


Your turn: 

Bulbasaur
Ivysaur
Venusaur
Charmander
Charmeleon
Charizard
Squirtle
Wartortle
Blastoise
Caterpie
Metapod
Butterfree
Weedle
Kakuna
Beedrill
Pidgey
Pidgeotto
Pidgeot
Rattata
Raticate

"""



import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import defaultdict



#########################################################################################################################

def import_dataset(dataset_path, dataset_path_test):
    """ Same as the one for question 3"""
    # anything else you want to do, do it here.
    # I would suggest normalizing the data appropiately.
    df = pd.read_csv(dataset_path)
    if "Pokemon_names" in df.columns:
        df = df.set_index("Pokemon_names")

    df_test = pd.read_csv(dataset_path_test)
    if "Pokemon_names" in df_test.columns:
        df_test = df_test.set_index("Pokemon_names")

    return df, df_test

######################################################## Part A ########################################################


def process_habitat_response(pokemon_and_habitat_list):
    poke_name: list[str] = []
    habitat: list[str] = []

    for line in pokemon_and_habitat_list.splitlines():
        line = line.strip()
        if not line or "," not in line:      # skip blanks or malformed lines
            continue

        name_part, habitat_part = map(str.strip, line.split(",", 1))
        poke_name.append(name_part)
        habitat.append(habitat_part)

    # build and return a dictionary for easy look-ups
    habitat_dict = dict(zip(poke_name, habitat))
    return habitat_dict



def get_most_common_habitat(pokes_and_habitats_dict):
    # Find the mode of this dictionary's values.
    # return the mode and the quantity of habitats at this mode.
    if not pokes_and_habitats_dict:
        raise ValueError("Input dictionary is empty")

    # Count occurrences of each habitat
    habitat_counts = Counter(pokes_and_habitats_dict.values())

    # Get the most common habitat and its count
    # most_common(1) returns a list like [('Habitat', count)]
    most_common_habitat, most_common_count = habitat_counts.most_common(1)[0]

    # Get the least common habitat and its count
    # most_common() without an argument returns all items sorted descending
    # The last element is the least common (or one of the least common in case of ties)
    least_common_items = habitat_counts.most_common()
    least_common_habitat, least_common_count = least_common_items[-1]


    return most_common_habitat, most_common_count, least_common_habitat, least_common_count


########################################################################################################################


def add_habitat_data_to_pandas_dataframe(dataset, habitat_dict):
    # using the pokemon name column in the dataset,
    # add a new column to the dataset called 'habitat'
    # this column should list the habitat of the pokemon in that row.
    df = dataset.copy()
    # The index holds Pokémon names, so map directly from df.index
    # First, ensure the index is string-typed and normalized the same way as habitat_dict keys
    normalized_names = df.index.astype(str).str.strip().str.title()
    df["habitat"] = normalized_names.map(habitat_dict).fillna("Unknown")
    return df



def train_predictive_model(train_dataset: pd.DataFrame, test_dataset: pd.DataFrame) -> float:
    """
    Train a RandomForestClassifier on train_dataset (index = Pokémon names, with 'poke_tiers'
    and 'raw_rankings') to predict 'poke_tiers'.  Returns test-set accuracy after adding habitat.

    Assumes both train_dataset and test_dataset have:
      - index = Pokémon name
      - columns: 'raw_rankings', 'poke_tiers', plus any number of other features
      - (after augmentation) a column 'habitat'

    Steps:
      1. Split into X_train / y_train and X_test / y_test
      2. Build ColumnTransformer that imputes+scales numeric, imputes+OHE categorical (incl. habitat)
      3. Fit a RandomForestClassifier pipeline
      4. Return accuracy on test set
    """
    # 1) Extract labels and features
    y_train = train_dataset["poke_tiers"].values
    X_train = train_dataset.drop(columns=["poke_tiers", "raw_rankings"])

    y_test  = test_dataset["poke_tiers"].values
    X_test  = test_dataset.drop(columns=["poke_tiers", "raw_rankings"])

    # 2) Identify numeric vs categorical columns
    numeric_cols     = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    # 3) Build preprocessing pipelines
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe,     numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])

    # 4) Create full pipeline
    model = Pipeline([
        ("prep", preprocessor),
        ("rf",   RandomForestClassifier(
                     n_estimators=300,
                     random_state=42,
                     class_weight="balanced"
                 ))
    ])

    # 5) Fit on train, predict on test
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 6) Compute and return accuracy
    return (model, accuracy_score(y_test, y_pred))


def plt_cm_matrix(labels_test, y_hat, forest):
    # 1. Get the classifier’s known class ordering (original string labels)
    # 1. Extract the class order from the trained RandomForest
    classes = list(forest.classes_)

    # 2. Build the confusion matrix, forcing that exact class‐order
    cm = confusion_matrix(labels_test, y_hat, labels=classes)

    # 3. Put it into a DataFrame so seaborn will label rows/columns
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)

    # 4. Plot it
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
    plt.tight_layout()
    plt.show()

    return cm


def get_column_importances(fitted_pipeline):
    """
    Extract “max per original column” importances from a fitted Pipeline whose
    last step is a RandomForestClassifier and whose first step is a ColumnTransformer
    named 'prep' that one-hot-encodes categorical columns.

    Returns
    -------
    column_importance : pd.Series
        Index = original column names (e.g., 'looks_fluffy', 'habitat', 'photo_contrast_level', …).
        Values = maximum importance among all transformed features derived from that column.
    """
    # 1. Get the fitted ColumnTransformer and RandomForestClassifier
    preprocessor = fitted_pipeline.named_steps["prep"]         # ColumnTransformer
    rf_model     = fitted_pipeline.named_steps["rf"]          # RandomForestClassifier

    # 2. Grab the array of raw feature names AFTER transformation
    #    e.g. ['num__poke_size', 'num__speed', 'cat__habitat_Freshwater', ...]
    feat_names = preprocessor.get_feature_names_out()

    # 3. Grab the corresponding importances from the RF
    importances = rf_model.feature_importances_               # 1D array, same length as feat_names

    # 4. Map each transformed feature back to its “original column”
    #    - Numeric columns appear as 'num__<orig_col>'
    #    - Categorical OHE columns appear as 'cat__<orig_col>_<value>'
    max_imp_per_col = defaultdict(float)

    for fname, imp in zip(feat_names, importances):
        if fname.startswith("num__"):
            orig = fname.split("num__", 1)[1]   # everything after 'num__'
        elif fname.startswith("cat__"):
            # e.g. 'cat__habitat_Freshwater' → split once to get 'habitat_Freshwater'
            remainder = fname.split("cat__", 1)[1]
            # then split at the first '_' to isolate the column name
            orig = remainder.split("_", 1)[0]
        else:
            # In case you have additional transformers—you can add more branches
            orig = fname

        # Track the maximum importance seen for this original column
        if imp > max_imp_per_col[orig]:
            max_imp_per_col[orig] = imp

    # 5. Convert to a pandas Series, sorted descending
    column_importance = pd.Series(max_imp_per_col).sort_values(ascending=False)
    return column_importance


def main():
    dataset_path = 'question_3_4_5_pokemon_rankings_train.csv'
    dataset_path_test = 'question_3_4_5_pokemon_rankings_test.csv'
    dataset, test_dataset = import_dataset(dataset_path, dataset_path_test)

    """

    Part A:

        Run the code that asks ChatGPT the query in chat_gpt_query. 
        
        Note 1: You should copy the prompt from the chat_gpt_query variable and paste it into the chatgpt website.
        Save the response. However, the list of pokemon in the query as it is, is not complete, and you will need to examine the pokemon names
        in the train and test set. You will likely need to query chatgpt multiple times to get the habitat for all the pokemon.
        You will also then need to stich the response together and save it as a csv file.
        
        Note 2: You are free to, and expected to modify the code below to make the data loading and processing work.
        
        What is the most common pokemon habitat for the 100 pokemon? How many pokemon belong to this habitat? 
        What is the least common habitat? How many pokemon live here?
    """
    
    # # query_chatgpt is not a function. You will put the prompt into chatgpt and save the response manually.
    # pokes_and_habitats_raw_response = query_chatgpt(chat_gpt_query) 

    # save the response so that you can easily edit it and load it later.
    saved_response_file = "original_100_pokemon_habitats.txt"
    # pd.DataFrame(pokes_and_habitats_raw_response).to_csv(saved_response_file, index=False)

    # load back the saved response file. In future runs, you can just start from here.
    # this will save a lot of time, since querying GPT takes several minutes.


  
    import os
    if os.path.isfile(saved_response_file):
        with open(saved_response_file, 'r') as file:
            pokes_and_habitats_raw_response = file.read()

    processed_response = process_habitat_response(pokes_and_habitats_raw_response)
    most_common_hab, quantity_pokes, least_common_hab, quantity_least_pokes = get_most_common_habitat(processed_response)
    print(F"Most common habitat: {most_common_hab} with {quantity_pokes} total 'mons")
    print(F"Least common habitat: {least_common_hab} with {quantity_least_pokes} total 'mons")

    """
        Part B:

            Write code that takes this habitat information and adds it to the 
            pokemons ranking dataset as a feature. 
            Using the augmented dataset, try to predict the tier_rankings of the photos 
            using your classifier code from Question 3. 
            Report your final achieved accuracy. 
    """
    augmented_dataset = add_habitat_data_to_pandas_dataframe(dataset, habitat_dict=processed_response)
    augmented_test_dataset = add_habitat_data_to_pandas_dataframe(test_dataset, habitat_dict=processed_response)
    fitted_pipeline, final_acc = train_predictive_model(augmented_dataset, augmented_test_dataset)
    print(F"Final Accuracy is: {final_acc}")

    """
            Part C:
                Similar to question 4, generate a confusion matrix for your random forest classifier. 
                Compare the result to the confusion matrix you got in question 4.  
                Has the confusion matrix changed with the introduction of 
                additional data?
    """

    # ─── E. Build confusion matrix ───────────────────────────────────
    # 1) True labels
    name_col = "poke_tiers"
    labels_test = augmented_test_dataset[name_col].values

    # 2) Predictions
    #    Since `pipeline` includes preprocessing, we just pass raw X_test
    X_test = augmented_test_dataset.drop(columns=["poke_tiers", "raw_rankings"])
    y_hat = fitted_pipeline.predict(X_test)

    # 3) The trained RandomForest inside the pipeline
    forest = fitted_pipeline.named_steps["rf"]

    # 4) Plot and get raw confusion‐matrix counts
    cm = plt_cm_matrix(labels_test, y_hat, forest)

    # 5) (Optional) Print the raw array so you can compare with Question 4
    print("Raw confusion matrix array:")
    print(cm)


    """
                Extra credit: 2 points. 
                Part D:
                    What features is the random forest classifier selecting on? Is there any change vs the 
                    features that were selected in Question 4. In particular, are any of the new habitat 
                    variables selected on? 
    """
    # Now extract the per-column importances:
    col_imps = get_column_importances(fitted_pipeline)

    # Print the top‐5 most important columns:
    print("Top 5 most important columns:")
    print(col_imps.head(5).to_string())

    # Print the bottom‐5 (least important) columns:
    print("\nBottom 5 least important columns:")
    print(col_imps.tail(5).to_string())

    # Check specifically whether 'habitat' appears in that listing:
    hab_imp = col_imps.get("habitat", 0.0)
    if hab_imp > 0:
        print(f"\nThe 'habitat' column has importance = {hab_imp:.6f}")
    else:
        print("\nThe 'habitat' column was not used (importance = 0).")







if __name__ == "__main__":
    main()