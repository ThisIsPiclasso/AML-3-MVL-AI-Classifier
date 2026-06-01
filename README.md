# Multi View Learning AI Image Classifier

## Prerequisites
The following software is required to run the model:

- **UV**: The virtual environment, python install and all required packages are handled through uv [click here](https://docs.astral.sh/uv/).



## Getting Started
### Getting the repo ready
1. Clone this repo locally.
2. Download trained_model.pt [here](https://drive.proton.me/urls/XSV847SNJ4#VeQddjygDw1E)
3. Put the model file in the models/ folder
### Running the code
1. Sync UV to get all required packages
    ```bash
    uv sync
    ```
2. Initilize the API (default is port 8000)
    ```bash
   uv run uvicorn main:app --reload --port 8000
    ```
3. Access the API docs (and current webui for inference) on 127.0.0.1:8000/docs
That's all, you can commit and push as always. The tracked files will be automatically stored with Git LFS.

## File and folder structure
```bash
├───data  # Stores .csv
│   ├───data_cache  # stores cached preprocessed tensors for every data split (not tracked in git)
│   ├───data_parquet  # stores metadata dataframes of dataset (not tracked in git)
│   ├───raw  # stores full dataset (not tracked in git)
│   ├───subset  # stores subset of dataset used in local development (not tracked in git)
├───models  # Stores .pt model
├───notebooks  # Contains experimental .ipynbs
├───MVL_AI_Classifier
│   ├───data  # stores all data handling steps
│   ├───features # stores all feature preprocessors
│   └───models  # stores model and loss function logic
│   └───constants.py  # stores all default values used across the project
│   └───model_configuration.py  # stores the model configuration dictionary used in the dynamic instantiation of the multiview model
├───reports
├───tests
│   ├───data
│   ├───features
│   └───models
├───.gitignore
├───.pre-commit-config.yaml
├───main.py
├───train_model.py
├───Pipfile
├───Pipfile.lock
├───README.md
```
