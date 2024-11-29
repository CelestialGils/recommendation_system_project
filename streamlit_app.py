# Import the necessary libraries/modules utilized for this project
import streamlit as st
import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.autograd import Variable
from tqdm import tqdm_notebook as tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# import the necessary classes
from matrix_factorization import MatrixFactorization
from loader import Loader

def save_complete_model(model, file_name):
  """Save the pre-trained model completely"""
  torch.save(model, file_name)


def load_complete_model(file_name):
  """Load the complete pre-trained model"""
  model = MatrixFactorization(n_users, n_items, n_factors=8)
  model = torch.load(file_name, weights_only=False, map_location=torch.device('cpu')) # change the execution to be done on the cpu
  model.eval()
  return model

# Following code was generated by Gemini and extensively modified to include
# a feature of finiding the proper movie title.
# Prompt: Write code that allows a user to give a movie they like and using the
# kmean model predict in which cluster it belongs.
def predict_cluster(movie_title):
  """Predicts the cluster for a given of movie title.

  Args:
    movie_title: Name of the movie provided by the user.

  Returns:
    The cluster number that the movie belongs to and the complete movie title
    the user was refering to.
  """
  # list to store the index for each movie in the cluster
  movie_indices = []

  # list to store the possible movies the user is refering to
  possible_movies = []
  try:
    # Loop through the title feature to identify movies with similar titles
    for movie in movies_df['title']:
      if movie_title in movie:
        possible_movies.append(movie)

    selection = 0
    # Perform the following if there are more than one movie with similar titles
    if len(possible_movies) > 1:
      print("Possible movies you are refering to:")
      movie_number = 1
      for movie in possible_movies:
        print(f"{movie_number} - {movie}")
        movie_number += 1
      selection = int(input(f"{formatting}\nPlease enter the number correspoding"
                            " to the movie title you are refering: "))

      while (selection > len(possible_movies) or selection <= 0):
        selection = int(input(f"{formatting}\nPlease enter a valid number from "
                              "thelist above: "))
      selection -= 1
    movie_title = possible_movies[selection]
    movie_id = movies_df.movieId[movies_df['title'] == movie_title].iloc[0]
    movie_idx = train_set.movieid2idx[movie_id]
    movie_indices.append(movie_idx)
  except (IndexError, KeyError):
    print(f"Warning: Movie '{movie_title}' not found in the dataset or not in"
          f"the training set.")

  if movie_indices:
    movie_embeddings = trained_movie_embeddings[movie_indices]
    cluster_kmeans = kmeans.predict(movie_embeddings).astype(int)
    # Assuming you want the most frequent cluster among the input movies
    most_frequent_clusters = (np.bincount(cluster_kmeans).argmax())
    return most_frequent_clusters, movie_title
  else:
    print("Error: No valid movie titles provided.")
    return None


def recomendation_clsuters(model, cluster):
  """This function would take the cluster number as an argument to print out
     the movies inside the cluster as recommendation back to the user"""
  # Top movies to be displayed
  top = 20

  movs = [] # list to store the movies title inside the cluster
  for movidx in np.where(model.labels_ == cluster)[0]:
    movid = train_set.idx2movieid[movidx]

    # introduced .iloc to fix an error that prevented output to be displayed properly
    rat_count = ratings_df.loc[ratings_df["movieId"] == movid].count().iloc[0]

    movid_name = movie_names[movid] # movie title

    # retreieve the movie's genres
    mov_genres = movies_df.loc[movies_df["movieId"] == movid, "genres"].iloc[0]
    movs.append((movid_name, rat_count, mov_genres))

  # Display the movies with their genres
  for mov in sorted(movs, key=lambda tup: tup[1], reverse=True)[:top]:
    # Skip this iteration, if the movie is the same as the input
    if mov[0] == movie_title:
      top += 1 # Increase the top by one to be able to compesate the skip
      continue
    print(f"\t{mov[0]}   {mov[2]}\n\t{formatting}")


# Formatting to keep the output clean
formatting = ('----------------------------------------------------------------'
              '----------------------------------------------')

st.title("🍿 Project Recommendation System")
st.write(
    "Welcome to Project Recommendation System! Powered by collaborative filtering and machine learning."
)

# Read the csv files and store their datasets
# MovieLens dataset that includes movie titles, their IDs, & genres
movies_df = pd.read_csv('movies.csv')

# MovieLens dataset that includes movie ids, users' ID and rating
# Limit the rating dataframe to include 653144 ratings for performance purposes
ratings_df = pd.read_csv('ratings.csv').head(653144)

# Store the movies ID and their titles into dictionary
movie_names = movies_df.set_index('movieId')['title'].to_dict()

# Store the number of unique users and movies - code autocompleted using Gemini
n_users = ratings_df.userId.unique().shape[0]
n_items = ratings_df.movieId.unique().shape[0]

# Load complete pre-trained model
model = load_complete_model('recommendation_model_complete.pth')

# sample size on data trainig
batch_size = 32

# Store the training dataset
train_set = Loader(ratings_df)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True) 

# Store the movies embedding 
trained_movie_embeddings = model.item_factors.weight.data.cpu().numpy()

# Initiate the KMeans model and fit the movie embedding
kmeans = KMeans(n_clusters=308, random_state=42)
kmeans.fit(trained_movie_embeddings)

# Get user's input for movie title
movie_title = input("Enter a movie title you like: ")
movie_title = movie_title.strip()  # Remove leading/trailing spaces

# Predict the cluster
predicted_cluster, movie_title = predict_cluster(movie_title)

if predicted_cluster is not None:
  print(f"The movies you provided belong to clusters: {predicted_cluster}")

print(f"The top 20 recommendations for {movie_title} are:\n")
recomendation_clsuters(kmeans, predicted_cluster)