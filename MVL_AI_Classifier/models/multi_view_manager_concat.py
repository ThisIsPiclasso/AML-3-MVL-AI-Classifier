from layer_arc import MLPLayer

import torch
import torch.nn as nn


class MultiViewNet(nn.Module):
    """
    This is the main multi-view network that takes in multiple views (branches)
    and performs intermediate fusion using a transformer encoder.
    """

    def __init__(self, encoders: dict, embed_dim: int = 512) -> None:
        """
        Initialize the multi-view network.
        Args:
            encoders (dict): A dictionary of encoder modules for each view(layer).
            embed_dim (int): The dimension of the output embedding from each encoder. Defaults to 512.

        Example encoders:
        encoders = {
            "azimuthal": MLPLayer(input_size=128, embed_dim=512),
            "noise": CNNLayer(arch="resnet18", input_chan=3),
        }
        """
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.embed_dim = embed_dim
        self.num_views = len(encoders)

        # Create an auxiliary head for every banch to get branch
        # specific prediction
        self.aux_heads = nn.ModuleDict(
            {key: nn.Linear(embed_dim, 2) for key in encoders.keys()}
        )

        # Final MLP prediction network
        self.classifier = MLPLayer(
            input_size=self.embed_dim * self.num_views, hidden_dim=[256], embed_dim=2
        )

    def forward(self, x_dict: dict) -> dict:
        """
        Forward pass through the multi-view network.
        Args:
            x_dict (dict): A dictionary containing input tensors for each view.
        Returns:
            dict: A dictionary containing the fusion output, branch outputs, and embeddings.
        """
        branch_features = {}
        branch_logits = {}

        # Extract features and immediately calculate branch-specific predictions
        for key, encoder in self.encoders.items():
            feat = encoder(x_dict[key])
            branch_features[key] = feat

            # Save branch logit for deep supervision/mutual learning
            branch_logits[key] = self.aux_heads[key](feat)

        # Average tokens into a 1x512 Global Feature Vector using mean pooling
        global_feature = torch.cat(list(branch_features.values()), dim=1)

        # Final MLP prediction
        fusion_logit = self.classifier(global_feature)

        # Return a dictionary for the custom loss function
        return {
            "fusion": fusion_logit,  # For L_fusion
            "branches": branch_logits,  # For L_branch
            "embeddings": branch_features,  # For feature analysis/visualization
        }
