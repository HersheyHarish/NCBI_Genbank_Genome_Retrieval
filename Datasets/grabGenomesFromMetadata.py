import os
import pandas as pd
from Bio import Entrez, SeqIO
from dotenv import load_dotenv
import time

# Load API key from .env
load_dotenv()
NCBI_API_KEY = os.getenv("NCBI_API_KEY")

# Configure Entrez
Entrez.email = "hsundar@ucdavis.edu"  # Replace with your email
Entrez.api_key = NCBI_API_KEY  # Use API key for faster requests

# Load dataset
csv_file = "filtered_output_v1.csv"  # Replace with your CSV file path
df = pd.read_csv(csv_file)

# Extract unique accession numbers
accession_list = df["accession"].dropna().unique().tolist()

# Function to fetch a single sequence
def fetch_sequence(accession):
    """Fetch the sequence for a single accession from NCBI."""
    try:
        with Entrez.efetch(db="nucleotide", id=accession, rettype="gb", retmode="text") as handle:
            record = SeqIO.read(handle, "gb")
            sequence = str(record.seq)
            print(f"Fetched {accession}: {sequence[:50]}...")  # Print first 50 bases for verification
            return sequence
    except Exception as e:
        print(f"Error fetching {accession}: {e}")
        return None

# Fetch sequences and store them in a dictionary
sequence_dict = {}
for i, acc in enumerate(accession_list):
    print(f"Fetching {i+1}/{len(accession_list)}: {acc}")
    sequence_dict[acc] = fetch_sequence(acc)

# Map sequences back to DataFrame
df["Genomic Sequence"] = df["accession"].map(sequence_dict)

# Save the updated dataset
output_file = "Random_BertDNA_RUN_1.csv"
df.to_csv(output_file, index=False)

print(f"\n Updated CSV saved as {output_file}")
