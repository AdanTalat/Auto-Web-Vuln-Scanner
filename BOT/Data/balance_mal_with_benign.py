import pandas as pd
from sklearn.utils import resample

# Load dataset CSV
df = pd.read_csv("combined_dataset.csv")

# Check class distribution
print("Original class distribution:")
print(df['Label'].value_counts())

# Separate majority and minority classes
malicious = df[df['Label'] == 1]
benign = df[df['Label'] == 0]

# Determine minority class size
minority_size = min(len(malicious), len(benign))

# Downsample majority class
malicious_downsampled = resample(malicious, 
                                replace=False,    # without replacement
                                n_samples=minority_size,
                                random_state=42)

benign_downsampled = resample(benign,
                              replace=False,
                              n_samples=minority_size,
                              random_state=42)

# Combine to form balanced dataset
balanced_df = pd.concat([malicious_downsampled, benign_downsampled])

# Shuffle rows
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Check new distribution
print("Balanced class distribution:")
print(balanced_df['Label'].value_counts())

# Save balanced dataset
balanced_df.to_csv("balanced_dataset.csv", index=False)

print("Balanced dataset saved as 'balanced_dataset.csv'")
