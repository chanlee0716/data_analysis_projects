"""

Question 2: Recommender systems.

For this question, we will use the dataset question_2_pokemon_biology_data.csv
This dataset contains biological data about pokemon. 
Your goal is to find the most similar pokemon to an input pokemon, in terms of cosine distance. 
You can think of this as a recommender system that takes a user's favorite pokemon and returns a recommended pokemon. 

Note the pokemon's name should be treated as an index.

Dataset description:

    name: index variable. Not a feature. Treat this as a row name. 
    attack: pokemon attack value. Continuous. 
    defense: pokemon defense. Continuous. 
    height_m: pokemon height in meters. Continuous. 
    percentage_male: percentage of males for this species. Continuous. 
    speed: pokemon's speed. Continuous. 
    weight_kg: pokemon's weight. Continuous. 


"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances   # fast, vectorised


def import_data(dataset_path):
    '''
    Import the data from the dataset path and return a pandas dataframe.
    '''
    data = pd.read_csv(dataset_path)

    # Detect correct case of the name column
    if "name" in data.columns:
        data = data.set_index("name")
    elif "Name" in data.columns:
        data = data.set_index("Name")
    else:
        raise KeyError("No 'name' column found in the dataset.")

    col_means = data.mean(numeric_only=True)   # Series of means
    data = data.fillna(col_means)                # in-place replacement
    
    # ----------------------------------------------------
    # 3. Mean scaling  (divide each column by its mean)
    # ----------------------------------------------------
    data_scaled = data / col_means

    return data_scaled, col_means


def find_most_similar_mon(dataset, input_pokemon='snorlax'):
    '''
    Given a pokemon, find the most similar pokemon, in terms of
    cosine distance in the feature space.
    '''

    # Map lower-case index to canonical name for easy lookup
    lower_index_map = {name.lower(): name for name in dataset.index}

    if input_pokemon.lower() not in lower_index_map:
        raise KeyError(f"Pokémon '{input_pokemon}' not found in dataset index.")

    query_name = lower_index_map[input_pokemon.lower()]

    # Extract the 1×d query vector and the full (n×d) matrix
    query_vec = dataset.loc[[query_name]].values        # shape (1, d)
    matrix    = dataset.values                          # shape (n, d)

    # Compute cosine distances to every Pokémon
    dists = cosine_distances(query_vec, matrix)[0]      # 0-th row

    # Exclude the query Pokémon itself by setting its dist = ∞
    self_idx = dataset.index.get_loc(query_name)
    dists[self_idx] = np.inf

    # Identify the minimum distance and corresponding Pokémon
    min_idx   = dists.argmin()
    min_dist  = float(dists[min_idx])
    closest_name = dataset.index[min_idx]

    return closest_name, min_dist


def main():
    dataset_path = 'question_2_pokemon_biology_data.csv'
    dataset, _ = import_data(dataset_path)

    # print(dataset.head())

    """
    
    First, you should handle missing values via imputation. Please impute the missing values
    with the mean value in that column.  
    
    Second, normalize all data columns in the dataset using mean scaling. 
    That is, divide the column values by the mean value in the column. 
    If you fail to scale correctly, you might get the wrong answer. 
    So please be careful. 
    
    
    Using the dataset, find the pokemon that are most similar to 
    Snorlax, Haunter, and Mew, in terms of consine distance. 
        
    Note, you should not return the pokemon itself as an answer. 
    Of course the most similar pokemon to Snorlax is Snorlax. 
    But this doesn't count. 
    
    Part a -- find the most similar pokemon to snorlax 
    
    Part b -- find the most similar pokemon to haunter 
    
    Part c -- find the most similar pokemon to mew
    

    """

    snoarlax_rec = find_most_similar_mon(dataset, input_pokemon='Snorlax')
    haunter_rec = find_most_similar_mon(dataset, input_pokemon='Haunter')
    mew_rec = find_most_similar_mon(dataset, input_pokemon='Mew')

    print(f"The most similar pokemon to Snorlax is {snoarlax_rec[0]} with a distance of {snoarlax_rec[1]:.2f}")
    print(f"The most similar pokemon to Haunter is {haunter_rec[0]} with a distance of {haunter_rec[1]:.2f}")
    print(f"The most similar pokemon to Mew is {mew_rec[0]} with a distance of {mew_rec[1]:.2f}")


if __name__ == "__main__":
    main()