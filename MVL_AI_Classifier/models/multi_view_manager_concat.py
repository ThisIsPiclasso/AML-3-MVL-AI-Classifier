import torch
import torch.nn as nn


from MVL_AI_Classifier.models.layer_arc import MODEL_DICTIONARY
from MVL_AI_Classifier.constants import (
    MAX_TRAINING_TIME,
    N_TRIALS,
    MAX_TUNE_EPOCHS,
)


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

    def tune(self, n_trials: int = N_TRIALS, timeout: int = MAX_TRAINING_TIME) -> dict:
        """Runs an automated hyperparameter tuning sweep on this architecture configuration.

        Args:
            n_trials (int): Maximum number of parameter combinations to evaluate.
            timeout (int): Total seconds allowed for the tuning process.

        Returns:
            dict: The optimal hyperparameter values found during tuning.
        """
        # Lazy imports to keep sepparate tuning specific dependecies
        import optuna
        from optuna import TrialPruned
        from train import train_epoch, validate, get_dataloaders
        from MVL_AI_Classifier.models.custom_loss import MultiViewLoss

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Hyperparameter tuning initialized on device: {device}")

        def objective(trial: optuna.Trial) -> float:
            """
            The objective function for Optuna hyperparameter tuning.
            It trains the model with the given trial's hyperparameters and returns the validation accuracy.
            Args:
                trial (optuna.Trial): The current trial object containing the hyperparameters to evaluate.
            Returns:
                float: The validation accuracy achieved with the current trial's hyperparameters."""
            # Dynamically sample values for speeding the process
            lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            alpha = trial.suggest_float("alpha", 0.1, 1.0, step=0.1)
            beta = trial.suggest_float("beta", 0.01, 0.5, log=True)
            temperature = trial.suggest_float("temperature", 1.5, 4.0, step=0.5)

            # Initialize custom multi-view loss module using tuned configurations
            criterion = MultiViewLoss(alpha=alpha, beta=beta, temperature=temperature)

            # Building loaders with the optimized trial batch size
            train_loader, val_loader = get_dataloaders(
                view_configuration=self.view_configuration,
                batch_size=batch_size,
                use_cache=True,
            )

            # Model setup using the custom architecture
            model = MultiViewNet(
                view_configuration=self.view_configuration,
                embed_dim=self.embed_dim,
            ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

            val_loss = 0.0
            val_accuracy = 0.0

            for epoch in range(MAX_TUNE_EPOCHS):
                # Training loop
                _, _ = train_epoch(model, train_loader, optimizer, criterion, device)

                val_loss, val_accuracy = validate(model, val_loader, criterion, device)

                print(
                    f"Epoch [{epoch+1}/{MAX_TUNE_EPOCHS}] -> Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%"
                )

                trial.report(val_accuracy, epoch)
                if trial.should_prune():
                    raise TrialPruned()

            trial.set_user_attr("accuracy", val_accuracy)
            return val_loss

        # Initialise Optuna study with automated pruner logic
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=3, n_warmup_steps=1, interval_steps=1
            ),
        )

        # Start optimisation
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
        )

        print("\n--- Tuning Optimization Complete ---")
        print(f"Best Trial Val Accuracy: {study.best_trial.value:.2f}")
        return study.best_trial.params
