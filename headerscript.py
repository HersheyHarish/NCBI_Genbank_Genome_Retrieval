import pandas as pd

# Define column names
column_names = ["Accession", "Description", "Organism", "Length", "Source", "Collection Date", "Isolate",]  # Modify as needed

# FIELDS = ["Accession", "Description", "Organism", "Length", "Source", "Collection Date", "Isolate",]

# Read CSV without header
df = pd.read_csv("./Datasets/coronaviridae_no_sars.csv", header=None, names=column_names)

# Save the CSV with the new header
df.to_csv("coronaviridae_no_sars_headers.csv", index=False)

# df = pd.read_csv("coronaviridae_batch_2_w_headers.csv")

# unique_values = df["Description"].unique()
# print(len(unique_values))
# print(unique_values)