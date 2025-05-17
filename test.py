import pandas as pd

df = pd.read_csv("crime_dataset.csv", sep=";")
# Count total rows
total_samples = len(df)

# Count rows where status_code is "AO"
ao_count = df[df["status_code"] == "IC"].shape[0]

# Calculate percentage
ao_percentage = (ao_count / total_samples) * 100

print(f"Percentage of samples with status_code 'AO': {ao_percentage:.2f}%")
