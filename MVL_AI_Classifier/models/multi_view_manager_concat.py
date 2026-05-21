import torch
import torch.nn as nn
from MVL_AI_Classifier.models.layer_arc import MODEL_DICTIONARY


class MultiViewNet(nn.Module):
    """
    This is the main multi-view network that takes in multiple views (branches)
    and performs intermediate fusion using a transformer encoder.
    """

    def __init__(self, view_configuration: dict, embed_dim: int = 512) -> None:
        """
        Initialize the multi-view network.
        Args:
            view_configuration (dict): A dictionary containing the configuration for each model and preprocessor per view
            embed_dim (int): The dimension of the output embedding from each encoder. Defaults to 512.

        Example view_configuration:
        VIEW_CONFIGURATION = {
            "aps": {
                    "preprocessor": AzimuthalPowerSpectrumPreprocessor(),
                    "model_type": "cnn",  # or mlp
                    "input_shape": (DEFAULT_N_BINS,),
            },
        }
        """
        super().__init__()
        self.view_configuration = view_configuration
        self.embed_dim = embed_dim
        self.num_views = len(view_configuration)

        self.encoders = nn.ModuleDict()
        # based on the vieq_configuration the model will now create all views specified and will add the live encoders to the list
        for view_name, config in view_configuration.items():
            model_type = config["model_type"]
            input_shape = config["input_shape"]

            # quick check if model actually is present in the MODEL_DICTIONARY
            if model_type not in MODEL_DICTIONARY:
                raise KeyError(
                    "model type specified in view configuration does not exist"
                )

            model = MODEL_DICTIONARY[model_type]
            self.encoders[view_name] = model(
                input_shape=input_shape, embed_dim=self.embed_dim
            )

        # Create an auxiliary head for every banch to get branch
        # specific prediction
        self.aux_heads = nn.ModuleDict(
            {key: nn.Linear(embed_dim, 2) for key in view_configuration.keys()}
        )

        # Final MLP prediction network, built using design factory
        mlp_class = MODEL_DICTIONARY["mlp"]
        self.classifier = mlp_class(
            input_shape=(self.embed_dim * self.num_views,),
            hidden_dim=[256],
            embed_dim=2,
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

        # concatenating embeddings into one big 1d array
        global_feature = torch.cat(list(branch_features.values()), dim=1)

        # Final MLP prediction
        fusion_logit = self.classifier(global_feature)

        # Return a dictionary for the custom loss function
        return {
            "fusion": fusion_logit,  # For L_fusion
            "branches": branch_logits,  # For L_branch
            "embeddings": branch_features,  # For feature analysis/visualization
        }
