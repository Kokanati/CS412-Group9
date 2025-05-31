# developed by: Reginald Hingano

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def select_top_k_features(df, target_col, k=30):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_columns = X.columns[selector.get_support()]

    df_selected = pd.DataFrame(X_new, columns=selected_columns)
    df_selected[target_col] = y.values  # Add target column back

    return df_selected
