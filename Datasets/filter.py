



import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Load datasets
sars_df = pd.read_csv("coronaviridae_batch_1.csv")  # 4,800 genomes
original_df = pd.read_csv("coronaviridae_no_sars_headers.csv")  # SARS-related genomes

# Remove duplicates based on Accession and Length
original_df = original_df.drop_duplicates(subset=["Accession", "Length"])
sars_df = sars_df.drop_duplicates(subset=["Accession", "Length"])

# Ensure SARS dataset is diverse
sars_df["Source"] = sars_df["Source"].fillna("Unknown")  # Fill missing source values
sars_df["Collection Date"] = sars_df["Collection Date"].fillna("Unknown")

# Stratify SARS dataset by Source and Organism
sars_df["Strata"] = sars_df["Source"] + "_" + sars_df["Organism"]

# Remove strata that have only one sample (not usable for stratification)
strata_counts = sars_df["Strata"].value_counts()
valid_strata = strata_counts[strata_counts > 1].index
sars_df = sars_df[sars_df["Strata"].isin(valid_strata)]

# Determine 15-20% of original dataset size, but ensure it does not exceed available SARS data
sample_size = min(int(len(original_df) * 0.15), len(sars_df))

# Perform stratified sampling only if sample size is valid
if sample_size > 0 and len(sars_df["Strata"].unique()) > 1:
    split = StratifiedShuffleSplit(n_splits=1, test_size=sample_size, random_state=42)
    for train_idx, sample_idx in split.split(sars_df, sars_df["Strata"]):
        sampled_sars = sars_df.iloc[sample_idx]
else:
    sampled_sars = sars_df.sample(n=sample_size, random_state=42)  # Fallback to random sampling

# Combine original dataset with sampled SARS sequences
final_df = pd.concat([original_df, sampled_sars]).reset_index(drop=True)

# Save the final dataset
final_df.to_csv("balanced_coronaviridae_dataset.csv", index=False)

print(f"Final dataset saved with {len(final_df)} entries")
