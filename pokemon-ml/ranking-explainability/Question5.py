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


import os
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import re


def _norm(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", name).lower()

def import_data(dataset_path):
    """ Same as the one for question 3"""
    # anything else you want to do, do it here.
    # I would suggest normalizing the data appropiately.
    df = pd.read_csv(dataset_path)
    if "Pokemon_names" in df.columns:
        df = df.set_index("Pokemon_names")
    return df

# ────────────────────────────────────────────────────────────────
def process_habitat_response(csv_path="all_pokemon_habitats.csv"):
    """ 
    Read ChatGPT habitat CSV → {normalized_name : habitat}
    where normalized_name = stripped & title-cased (e.g. 'Bulbasaur').

    Returns a dictionary of {normalized_name : habitat}
    """
    df = pd.read_csv(csv_path, header=None, names=["name", "habitat"], dtype=str, keep_default_na=False)
    df["name"] = df["name"].map(_norm)       # normalise here
    df["habitat"] = df["habitat"].str.strip()
    return dict(zip(df["name"], df["habitat"]))



# def add_habitat_data_to_pandas_dataframe(dataset, habitat_dict):
#     # using the pokemon name column in the dataset,
#     # add a new column to the dataset called 'habitat'
#     # this column should list the habitat of the pokemon in that row.
#     return dataset

def add_habitat_data_to_pandas_dataframe(df, habitat_dict):
    """
    Add a 'habitat' column (categorical) to df based on its index.
    Pokémon with missing habitat get the value 'Unknown'.

    Returns a pandas dataframe with the habitat column added.
    """
    df = df.copy()
    df["habitat"] = df.index.map(lambda n: habitat_dict.get(n, "Unknown"))
    return df


# ────────────────────────────────────────────────────────────────
# MOST / LEAST COMMON HABITAT
# ────────────────────────────────────────────────────────────────
from collections import Counter

def get_most_common_habitat(habitat_dict, subset_names=None):
    """
    Find the most and least common habitat in the habitat dictionary.

    Returns the most common habitat, the count of the most common habitat,
    the least common habitat, and the count of the least common habitat.
    """
    counts = Counter(habitat_dict[n] for n in subset_names if n in habitat_dict)
    missing = [n for n in subset_names if n not in habitat_dict]
    if missing:
        print("⚠️  still missing", len(missing), "names (first 10):", missing[:10])
    most_hab, most_cnt = counts.most_common(1)[0]
    least_hab, least_cnt = counts.most_common()[-1]
    return most_hab, most_cnt, least_hab, least_cnt


# ────────────────────────────────────────────────────────────────
# PREPROCESSOR + MODEL
# ────────────────────────────────────────────────────────────────
def build_preprocessor(df):
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = [c for c in df.columns if c not in num_cols]

    num_pipe = Pipeline(
        [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    )
    cat_pipe = Pipeline(
        [("imp", SimpleImputer(strategy="most_frequent")),
         ("ohe", OneHotEncoder(handle_unknown="ignore"))]
    )

    return ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)]
    )


def train_predictive_model(train_df, test_df):
    """Return test-set accuracy after adding habitat."""
    y_train = train_df["poke_tiers"].values
    X_train = train_df.drop(columns=["poke_tiers", "raw_rankings"])

    y_test = test_df["poke_tiers"].values
    X_test = test_df.drop(columns=["poke_tiers", "raw_rankings"])

    pipe = Pipeline(
        [
            ("prep", build_preprocessor(X_train)),
            ("rf", RandomForestClassifier(
                n_estimators=300, random_state=42, class_weight="balanced"))
        ]
    )
    pipe.fit(X_train, y_train)
    y_hat = pipe.predict(X_test)
    return accuracy_score(y_test, y_hat), y_test, y_hat, pipe




def main():
    train_dataset = import_data('question_3_4_5_pokemon_rankings_train.csv')
    test_dataset = import_data('question_3_4_5_pokemon_rankings_test.csv')
    habitat_dict = process_habitat_response("all_pokemon_habitats.csv")
    print(list(habitat_dict.keys())[:10])

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

    # # save the response so that you can easily edit it and load it later.
    # saved_response_file = "chatGPT_Response.txt"
    # pd.DataFrame(pokes_and_habitats_raw_response).to_csv(saved_response_file, index=False)

    # load back the saved response file. In future runs, you can just start from here.
    # this will save a lot of time, since querying GPT takes several minutes.
    # import os
    # if os.path.isfile(saved_response_file):
    #     with open(saved_response_file, 'r') as file:
    #         pokes_and_habitats_raw_response = file.read()

    # processed_response = process_habitat_response(pokes_and_habitats_raw_response)
    # most_common_hab, quantity_pokes = get_most_common_habitat(processed_response)
    # print(F"Most common habitat{most_common_hab} with {quantity_pokes} total 'mons")

    # 1.  Load the ChatGPT habitat file into a DataFrame (two columns, no header)
    hab_df = (
        pd.read_csv("all_pokemon_habitats.csv", header=None, names=["Pokemon", "Habitat"])
        .assign(Pokemon=lambda d: d["Pokemon"].str.strip().str.title())
    )
    print(hab_df.head())

    # 2.  Extract the 100 training-set names and normalise them the same way
    name_col = "Pokemon_names" if "Pokemon_names" in train_dataset.columns else "pokemon_names"
    train_names = (
        train_dataset[name_col]
        .astype(str)
        .str.strip()
        .str.title()
    )

    # 3.  Merge names ↔ habitats, then count
    merged = train_names.to_frame(name="Pokemon").merge(hab_df, on="Pokemon", how="left")
    hab_counts = merged["Habitat"].value_counts()

    most_hab   = hab_counts.idxmax()
    most_cnt   = hab_counts.max()
    least_hab  = hab_counts.idxmin()
    least_cnt  = hab_counts.min()

    print(f"Most common habitat (training 100):  {most_hab}  ({most_cnt})")
    print(f"Least common habitat (training 100): {least_hab} ({least_cnt})")

    """
        Part B:

            Write code that takes this habitat information and adds it to the 
            pokemons ranking dataset as a feature. 
            Using the augmented dataset, try to predict the tier_rankings of the photos 
            using your classifier code from Question 3. 
            Report your final achieved accuracy. 
    """
    # augmented_dataset = add_habitat_data_to_pandas_dataframe(dataset, habitat_dict=pokes_and_habitats_raw_response)
    # augmented_test_dataset = add_habitat_data_to_pandas_dataframe(test_dataset, habitat_dict=pokes_and_habitats_raw_response)
    # final_acc = train_predictive_model(augmented_dataset, augmented_test_dataset)
    # print(F"Final Accuracy is: {final_acc}")

    """
            Part C:
                Similar to question 4, generate a confusion matrix for your random forest classifier. 
                Compare the result to the confusion matrix you got in question 4.  
                Has the confusion matrix changed with the introduction of 
                additional data?
    """

    # plt_cm_matrix(labels_test=None, y_hat=None, forest=None)


    """
                Extra credit: 2 points. 
                Part D:
                    What features is the random forest classifier selecting on? Is there any change vs the 
                    features that were selected in Question 4. In particular, are any of the new habitat 
                    variables selected on? 
        """



if __name__ == "__main__":
    main()