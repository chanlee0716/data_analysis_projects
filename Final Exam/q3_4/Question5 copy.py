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
import re

#optional helper to normalise names
def _norm(name: str) -> str:
    """Strip whitespace, title-case, remove trailing dots, keep only A–Z0–9."""
    return re.sub(r"[^A-Za-z0-9]", "", name.strip().title())


from pathlib import Path

def csv_to_habitat_txt(csv_path: str, txt_path: str | None = None) -> Path:
    """
    Convert a two-column CSV (name, habitat) into a plain-text file where each line
    looks like:   Bulbasaur, Grassland

    Parameters
    ----------
    csv_path : str
        Path to the source CSV (no header, two comma-separated columns).
    txt_path : str or None
        Where to save the .txt file.  If None, use the same basename
        with .txt extension in the same folder.

    Returns
    -------
    Path
        Path object pointing to the written .txt file.
    """
    if txt_path is None:
        txt_path = csv_path.with_suffix(".txt")
    txt_path = Path(txt_path)

    # 1️⃣ load as strings, but keep_default_na=True means NaN -> "nan" string
    df = pd.read_csv(
        csv_path,
        header=None,
        names=["Pokemon", "Habitat"],
        dtype=str,
        keep_default_na=False,   # important: empty → "" (not NaN)
    )

    # 2️⃣ drop rows missing either value
    df = df[(df["Pokemon"].str.strip() != "") & (df["Habitat"].str.strip() != "")]

    # 3️⃣ build "Pokemon, Habitat" lines
    lines = [
        f"{str(p).strip()}, {str(h).strip()}"
        for p, h in zip(df["Pokemon"], df["Habitat"])
    ]

    # 4️⃣ write
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Wrote {len(lines)} lines → {txt_path}")
    return txt_path


def import_data(dataset_path):
    """ Same as the one for question 3"""
    # anything else you want to do, do it here.
    # I would suggest normalizing the data appropiately.
    df = pd.read_csv(dataset_path)
    if "Pokemon_names" in df.columns:
        df = df.set_index("Pokemon_names")
    return df



def process_habitat_response(txt_path):
    # poke_name = []
    # habitat = []

    # for line in pokemon_and_habitat_list.splitlines():
    #     pass
    """
    Read the ChatGPT-generated habitat CSV and return {pokemon: habitat}.
    The file should have no header, one "Name,Habitat" per line.
    """
    # 1. Load as strings
    txt_path = Path(txt_path)
    habitat_dict: dict[str, str] = {}

    with txt_path.open("r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or "," not in ln:
                continue                       # skip blank / malformed line
            name, hab = map(str.strip, ln.split(",", 1))

            key = _norm(name) if normalise else name
            habitat_dict[key] = hab           # last one wins if duplicates

    print(f"[INFO] Parsed {len(habitat_dict)} Pokémon → habitat pairs from {txt_path}")
    return habitat_dict


def add_habitat_data_to_pandas_dataframe(dataset, habitat_dict):
    # using the pokemon name column in the dataset,
    # add a new column to the dataset called 'habitat'
    # this column should list the habitat of the pokemon in that row.
    return dataset


def train_predictive_model(dataset, test_dataset):
    return 'MSE or accuracy'

def get_most_common_habitat(pokes_and_habitats_dict):
    # Find the mode of this dictionary's values.
    # return the mode and the quantity of habitats at this mode.
    # return 'Outer space', 42
    if not pokes_and_habitats_dict:
        raise ValueError("habitat_dict is empty")
    counts = Counter(pokes_and_habitats_dict.values())
    mode_habitat, mode_count = counts.most_common(1)[0]
    return mode_habitat, mode_count


def main():
    train_dataset = import_data('question_3_4_5_pokemon_rankings_train.csv')
    test_dataset = import_data('question_3_4_5_pokemon_rankings_test.csv')

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

    # # load back the saved response file. In future runs, you can just start from here.
    # # this will save a lot of time, since querying GPT takes several minutes.
    # import os
    # if os.path.isfile(saved_response_file):
    #     with open(saved_response_file, 'r') as file:
    #         pokes_and_habitats_raw_response = file.read()

    # 1) Convert CSV → TXT (only needed once)
    csv_to_habitat_txt("all_pokemon_habitats.csv")  # creates all_pokemon_habitats.txt

    # 2) Load that TXT into a dict
    habitat_dict = process_habitat_response("all_pokemon_habitats.txt")

    # 3) Example look-up
    print(habitat_dict.get(_norm("Mr. Mime"), "Unknown"))

    processed_response = process_habitat_response("all_pokemon_habitats.csv")
    most_common_hab, quantity_pokes = get_most_common_habitat(processed_response)
    print(F"Most common habitat{most_common_hab} with {quantity_pokes} total 'mons")

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