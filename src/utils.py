import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    # Drop irrelevant column
    df = df.drop(columns=["customerID"])

    # Target
    y = df["Churn"].map({"Yes": 1, "No": 0})

    X = df.drop(columns=["Churn"])

    # Encode categorical features
    cat_cols = X.select_dtypes(include="object").columns
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Scale numerical features
    scaler = StandardScaler()
    X[X.columns] = scaler.fit_transform(X)

    return X.values, y.values
