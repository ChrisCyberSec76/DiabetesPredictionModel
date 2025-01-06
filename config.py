# config.py
# Configuration settings for the ML project

from pathlib import Path

# Define project structure
PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / 'data'
MODEL_PATH = PROJECT_ROOT / 'models'
RESULTS_PATH = PROJECT_ROOT / 'results'

# Create directories if they don't exist
for path in [DATA_PATH, MODEL_PATH, RESULTS_PATH]:
    path.mkdir(exist_ok=True)

# Model parameters
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42
}