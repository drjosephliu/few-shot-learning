import torch
from torch.optim import Optimizer
from torch.nn import Module
from typing import Callable

from few_shot.utils import pairwise_distances

def autoencoder_episode(model: Module,
                        optimiser: Optimizer,
                        loss_fn: Callable, 
                        x: torch.Tensor,
                        y: torch.Tensor,
                        n_shot: int,
                        k_way: int,
                        q_queries: int,
                        distance: str,
                        train: bool):

    """Performs a single training episode for the baseline nearest neigbhbour
    model.
    """

    if train:
        # Zero gradients
        model.train()
        optimiser.zero_grad()
    else:
        model.eval()

    x = x.view(x.size(0), 1, -1)

    # Embed all samples
    embeddings = model(x)

    # Samples are ordered by the NShotWrapper class as follows: # k lots of n support samples from a particular class
    # k lots of q query samples from those classes
    support = embeddings[:n_shot*k_way]
    queries = embeddings[n_shot*k_way:]

    # Calculate squared distances between all queries and all prototypes
    # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
    distances = pairwise_distances(queries, support, distance)

    # Calculate log p_{phi} (y = k | x)
    log_p_y = (-distances).log_softmax(dim=1)
    loss = loss_fn(log_p_y, y)

    # Prediction probabilities are softmax over distances
    y_pred = (-distances).softmax(dim=1)

    if train:
        # Take gradient step
        loss.backward()
        optimiser.step()
    else:
        pass

    return loss, y_pred

def nearest_neighbour_episode(model: Module,
                              optimiser: Optimizer,
                              loss_fn: Callable,
                              x: torch.Tensor,
                              y: torch.Tensor,
                              n_shot: int,
                              k_way: int,
                              q_queries: int,
                              distance: str,
                              train: bool):
    """Performs a single training episode for the baseline nearest neigbhbour
    model.
    """

    if train:
        # Zero gradients
        model.train()
        optimiser.zero_grad()
    else:
        model.eval()

    x = x.view(x.size(0), 1, -1)

    # Embed all samples
    embeddings = model.encoder(x)

    # Samples are ordered by the NShotWrapper class as follows: # k lots of n support samples from a particular class
    # k lots of q query samples from those classes
    support = embeddings[:n_shot*k_way]
    queries = embeddings[n_shot*k_way:]

    # Calculate squared distances between all queries and all prototypes
    # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
    distances = pairwise_distances(queries, support, distance)

    # Calculate log p_{phi} (y = k | x)
    log_p_y = (-distances).log_softmax(dim=1)
    loss = loss_fn(log_p_y, y)

    # Prediction probabilities are softmax over distances
    y_pred = (-distances).softmax(dim=1)

    if train:
        # Take gradient step
        loss.backward()
        optimiser.step()
    else:
        pass

    return loss, y_pred


