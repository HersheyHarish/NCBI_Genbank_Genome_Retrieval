import pandas as pd
import random

def random_viral_samples(csv_file, k):
    """
    Selects k random virus genome records from the given CSV file.

    Args:
        csv_file (str): Path to the CSV file.
        k (int): Number of random entries to retrieve.

    Returns:
        pd.DataFrame: DataFrame containing k randomly selected records.
    """
    # Load the CSV into a Pandas DataFrame
    df = pd.read_csv(csv_file)

    # Ensure k does not exceed available rows
    k = min(k, len(df))

    # Randomly sample k rows
    sampled_df = df.sample(n=k, random_state=42)  # random_state ensures reproducibility

    # Select only the required columns
    selected_columns = ["accession", "seq_definition", "seq_length", "GBSeq_organism"]
    sampled_df = sampled_df[selected_columns]

    return sampled_df

# Example usage:
csv_file_path = "your_file.csv"  # Replace with your actual file path
k = 10  # Define how many random entries you want

# Get the random samples
random_samples = random_viral_samples(csv_file_path, k)

# Display the output
print(random_samples)

# Optionally, save to a new CSV
random_samples.to_csv("random_viral_samples.csv", index=False)
