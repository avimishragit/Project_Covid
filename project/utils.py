import joblib
import json

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def save_params(params, path):
    with open(path, 'w') as f:
        json.dump(params, f)

def load_params(path):
    with open(path, 'r') as f:
        return json.load(f)
