import torch

# Class where the Matrix Factorization is going to take place
# code retrieved from Spencer Pao's program and using Gemini's autocomplete
class MatrixFactorization(torch.nn.Module):
  def __init__(self, n_users, n_items, n_factors=20):
    """This function creates the embeddings for the users and items/movies as
    well as assigning their weights"""
    super().__init__()

    # create user embeddings
    self.user_factors = torch.nn.Embedding(n_users, n_factors)

    # create item embeddings
    self.item_factors = torch.nn.Embedding(n_items, n_factors)

    # assign the weights to the embeddings
    self.user_factors.weight.data.uniform_(0, 0.05)
    self.item_factors.weight.data.uniform_(0, 0.05)

  def forward(self, data):
    """Multiply the matrix together"""
    # matrix multiplication
    users, items = data[:,0], data[:,1]
    return (self.user_factors(users)*self.item_factors(items)).sum(1)

  def predict(self, user, item):
    return self.forward(user, item)