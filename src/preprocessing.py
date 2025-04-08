import pandas as pd
import os

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

# Load data
train_df = pd.read_csv("data/adult.data", names=columns,
                       na_values=" ?", skipinitialspace=True)
test_df = pd.read_csv("data/adult.test", names=columns,
                      na_values=" ?", skipinitialspace=True, skiprows=1)

# Basic preprocessing
combined = pd.concat([train_df, test_df])
combined.dropna(inplace=True)
combined = combined.drop(
    ["fnlwgt", "education", "capital-gain", "capital-loss"], axis=1)

# Split again
train_clean = combined.sample(frac=0.8, random_state=42)
test_clean = combined.drop(train_clean.index)

# Save processed data
train_clean.to_csv("data/train_processed.csv", index=False)
test_clean.to_csv("data/test_processed.csv", index=False)
