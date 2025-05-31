import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Load undersampled dataset
file_path = 'dataset_80_20.csv'  # ✅ Update if needed
df = pd.read_csv(file_path)

# Define target and features
label_column = 'phishing'  # ✅ Updated from 'status'
y = df[label_column]
X = df.drop(columns=[label_column])

# Feature names for reference
feature_names = X.columns

# Define different K values
k_values = [30, 20, 10]

for k in k_values:
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_new = selector.fit_transform(X, y)

    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = feature_names[selected_mask]

    # Create new dataframe
    df_selected = pd.DataFrame(X_new, columns=selected_features)
    df_selected[label_column] = y.reset_index(drop=True)  # Add target column back

    # Save to CSV
    output_file = f"dataset80_20_top{k}.csv"
    df_selected.to_csv(output_file, index=False)
    print(f"Saved: {output_file} with top {k} features.")
