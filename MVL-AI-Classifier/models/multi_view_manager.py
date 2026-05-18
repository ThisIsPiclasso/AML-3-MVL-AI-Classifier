from layer_arc import MLPLayer

import torch
import torch.nn as nn


class MultiViewNet(nn.Module):
    def __init__(
        self, encoders: dict, embed_dim: int = 512, num_heads: int = 8
    ) -> None:
        """
        Initialize the multi-view network.
        Args:
            encoders (dict): A dictionary of encoder modules for each view.
            embed_dim (int): The dimension of the output embedding from each encoder. Defaults to 512.
            num_heads (int): The number of attention heads for the transformer. Defaults to 8.
        """
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.embed_dim = embed_dim

        # Create an auxiliary head for every banch
        self.aux_heads = nn.ModuleDict(
            {key: nn.Linear(embed_dim, 1) for key in encoders.keys()}
        )

        # Intermediate fusion block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Final MLP prediction network
        self.classifier = MLPLayer(input_size=embed_dim, hidden_dim=[256], embed_dim=1)

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
            feat = encoder(x_dict[key])  # Expected shape - [Batch, embed_dim]
            branch_features[key] = feat

            # Save branch logit for deep supervision/mutual learning
            branch_logits[key] = self.aux_heads[key](feat)

        # Stack individual 1x512 vectors into tokens for the Transformer
        stacked_tokens = torch.stack(list(branch_features.values()), dim=1)

        # Apply Intermediate Fusion via Self-Attention
        fused_tokens = self.transformer(stacked_tokens)

        # Collapse tokens into a 1x512 Global Feature Vector via Mean Pooling
        global_feature = fused_tokens.mean(dim=1)

        # Final MLP prediction
        fusion_logit = self.classifier(global_feature)

        # Return a dictionary for the custom loss function
        return {
            "fusion": fusion_logit,  # For L_fusion
            "branches": branch_logits,  # For L_branch and L_KD
            "embeddings": branch_features,  # For feature analysis/visualization
        }
