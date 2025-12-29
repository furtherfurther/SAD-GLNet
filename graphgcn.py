import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


def generate_random_mask(num_nodes, num_features, mask_prob=0.5):
    """
    Randomly generate a mask matrix. Each feature is masked based on the given probability.
    :param num_nodes: Number of nodes
    :param num_features: Feature dimension of each node
    :param mask_prob: Masking probability (between 0 and 1); higher values result in more masked features.
    :return: Boolean mask matrix
    """
    mask = torch.rand((num_nodes, num_features)) > mask_prob  # Generate a boolean mask
    return mask

    
