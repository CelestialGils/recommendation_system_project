# Originally by Spencer Pao*
class Loader(Dataset):
  def __init__(self, Dataset):
    """Extract the users and movies ID to create mapping between them for matrix
    factorization. Additioanlly, extract the userId and ratings features so they
    can be transform into tensors for deep learning modeling
    """
    self.ratings = Dataset.copy()

    # Extract the user and movies IDs for the matrix factorization
    users = Dataset.userId.unique() # Users ID
    movies = Dataset.movieId.unique() # Movies ID

    self.userid2idx = {o:i for i,o in enumerate(users)} # User ID to index
    self.movieid2idx = {o:i for i,o in enumerate(movies)} # Movie ID to index

    self.idx2userid = {i:o for o,i in self.userid2idx.items()} # Index to user ID
    self.idx2movieid = {i:o for o,i in self.movieid2idx.items()} # Index to movie ID

    self.ratings.movieId = Dataset.movieId.apply(lambda x: self.movieid2idx[x])
    self.ratings.userId = Dataset.userId.apply(lambda x: self.userid2idx[x])

    # Store the [userId, movieId] from the ratings dataframe
    self.x = self.ratings.drop(['rating', 'timestamp'], axis=1).values

    # Store the ratings values
    self.y = self.ratings['rating'].values

    # Transform the data to tensors (this is for deep learning modeling)
    self.x, self.y = torch.tensor(self.x), torch.tensor(self.y)

  def __getitem__(self, index):
    return (self.x[index], self.y[index])

  def __len__(self):
    return len(self.ratings)