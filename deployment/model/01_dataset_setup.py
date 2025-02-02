from kaggle import kagglehub

# TODO: Set up kaggle key with secret manager

# Download latest version
path = kagglehub.dataset_download("quantbruce/real-estate-price-prediction")

print("Path to dataset files:", path)