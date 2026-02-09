import torch
import torch.nn as nn

class HierarchicalGraphPlanner(nn.Module):
    """
    Hierarchical Graph-Planner (HGP) model for NewsSumm
    """

    def __init__(self, encoder_dim):
        super().__init__()

        # Sentence/document representation projection
        self.projection = nn.Linear(encoder_dim, encoder_dim)

        # Graph aggregation (placeholder for GNN)
        self.graph_layer = nn.MultiheadAttention(
            embed_dim=encoder_dim,
            num_heads=4,
            batch_first=True
        )

        # Planner head
        self.planner = nn.Linear(encoder_dim, encoder_dim)

    def forward(self, representations):
        """
        representations: Tensor [num_nodes, hidden_dim]
        """
        h = self.projection(representations)
        h, _ = self.graph_layer(h, h, h)
        plan = self.planner(h)
        return plan
