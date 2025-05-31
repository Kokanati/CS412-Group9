import subprocess

python_exec = "/Users/lipe/.pyenv/versions/phishing310/bin/python"

# List of dataset files (as seen in your screenshot)
dataset_files = [
    "dataset/dataset80_20_top10.csv",
    "dataset/dataset80_20_top20.csv",
    "dataset/dataset80_20_top30.csv",
    "dataset/dataset90_10_top10.csv",
    "dataset/dataset90_10_top20.csv",
    "dataset/dataset90_10_top30.csv",
    "dataset/dataset95_5_top10.csv",
    "dataset/dataset95_5_top20.csv",
    "dataset/dataset95_5_top30.csv"
]

# Run the script for each dataset
for file in dataset_files:
    print(f"\n=== Running GAN-MCMC pipeline for {file} ===\n")
    result = subprocess.run(
        [python_exec, "MCMC_GAN.py", file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("Errors:\n", result.stderr)

