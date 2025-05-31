import pandas as pd
from sklearn.utils import shuffle

# Load dataset
file_path = 'raw_dataset.csv'  # 🔁 Replace with your actual CSV file path
df = pd.read_csv(file_path)

# Define label column and values
label_column = 'phishing'       # 1 = phishing, 0 = legitimate
phishing_value = 1
legitimate_value = 0

# Separate phishing and legitimate samples
df_legit = df[df[label_column] == legitimate_value]
df_phish = df[df[label_column] == phishing_value]

print(f"Original count - Legitimate: {len(df_legit)}, Phishing: {len(df_phish)}")

# Desired ratios
ratios = {
    "80_20": (0.8, 0.2),
    "90_10": (0.9, 0.1),
    "95_5": (0.95, 0.05),
}

for name, (legit_ratio, phish_ratio) in ratios.items():
    # Limit total size based on the number of available legitimate samples
    target_legit = len(df_legit)
    target_phish = int(target_legit * (phish_ratio / legit_ratio))

    # Make sure we don't request more phishing samples than we have
    target_phish = min(target_phish, len(df_phish))
    target_legit = int(target_phish * (legit_ratio / phish_ratio))

    # Sample the data
    df_legit_sampled = df_legit.sample(n=target_legit, random_state=42)
    df_phish_sampled = df_phish.sample(n=target_phish, random_state=42)

    # Combine and shuffle
    df_final = shuffle(pd.concat([df_legit_sampled, df_phish_sampled]), random_state=42)

    # Save to file
    output_file = f"dataset_{name}.csv"
    df_final.to_csv(output_file, index=False)
    print(f"Saved {output_file} with Legitimate={target_legit}, Phishing={target_phish}")
