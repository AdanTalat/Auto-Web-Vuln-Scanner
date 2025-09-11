import pandas as pd

# Load your original dataset
df = pd.read_csv("processed_dataset.csv")

# Randomly take 40,000 samples (change 40000 to what you want)
df_sampled = df.sample(n=10000, random_state=42)

# Save the reduced dataset
df_sampled.to_csv("reducted_dataset.csv", index=False)

print(f"Dataset reduced from {len(df)} to {len(df_sampled)} rows.")
