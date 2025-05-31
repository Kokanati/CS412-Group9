#developed by: Reginald Hingano

# Dataset
DATA_PATH = "data/phishing_dataset.csv"    
TARGET_COLUMN = "status"                   

# Feature Selection
TOP_K_FEATURES = 30

# GAN settings
Z_DIM = 64
EPOCHS = 100
BATCH_SIZE = 64
MCMC_STEPS = 10
MCMC_STEP_SIZE = 0.05

# Train/test split
TEST_SIZE = 0.3
RANDOM_STATE = 42

# Output directories
LOG_DIR = "logs"
OUTPUT_DIR = "outputs"
