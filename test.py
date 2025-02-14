from Bio import Entrez, SeqIO
import time
import csv
import pandas as pd
from tqdm import tqdm  # For progress bars

# Global Variables
Entrez.email = "hsundar@ucdavis.edu"  # Your email for NCBI
API_KEY = "012754ffe3ff51e270b98fe8d378af24a109"  # Replace with your NCBI API key

# Function to search complete genomes
def search_complete_genomes(query, retmax):
    """
    Search the NCBI GenBank nucleotide database for complete genomes.
    """
    try:
        with Entrez.esearch(db="nucleotide", term=query, retmax=retmax, api_key=API_KEY) as handle:
            record = Entrez.read(handle)
        return int(record["Count"]), record["IdList"]
    except Exception as e:
        print(f"Error searching for complete genomes: {e}")
        return 0, []

# Function to fetch genome details
def fetch_genome_details(accession_id):
    """
    Fetch metadata and genome sequence for a given accession ID.
    """
    try:
        with Entrez.efetch(db="nucleotide", id=accession_id, rettype="gb", retmode="text", api_key=API_KEY) as handle:
            record = SeqIO.read(handle, "genbank")

        collection_date = "Unknown"
        isolation_source = "Unknown"
        taxonomy_id = "Unknown"

        for feature in record.features:
            if feature.type == "source":
                collection_date = feature.qualifiers.get("collection_date", ["Unknown"])[0]
                isolation_source = feature.qualifiers.get("isolation_source", ["Unknown"])[0]
                taxonomy_id = feature.qualifiers.get("db_xref", ["Unknown"])[0].replace("taxon:", "")

        return {
            "Accession": record.id,
            "Description": record.description,
            "Organism": record.annotations.get("organism", "Unknown"),
            "Taxonomy ID": taxonomy_id,
            "Length": len(record.seq),
            "Source": record.annotations.get("source", "Unknown"),
            "Taxonomy": "; ".join(record.annotations.get("taxonomy", [])),
            "Collection Date": collection_date,
            "Isolation Source": isolation_source,
            "Sequence": str(record.seq),
        }
    except Exception as e:
        print(f"Error fetching details for {accession_id}: {e}")
        return None

# Save data to CSV
def save_to_csv(data, filename):
    """Save genome data to a CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}.")

# Main Execution
query = "Coronaviridae[Organism] AND complete genome[Title]"
retmax = 100000  # Adjust based on your needs
output_file = "coronaviridae_genome_data.csv"

# Step 1: Search for records
print(f"Searching GenBank with query: {query}")
total_count, record_ids = search_complete_genomes(query, retmax=retmax)
print(f"Total records found: {total_count}")

# Step 2: Fetch and filter genome details
print("Fetching genome details...")
genome_data = []
unique_tax_ids = set()

for idx, record_id in enumerate(tqdm(record_ids, desc="Fetching records")):
    details = fetch_genome_details(record_id)
    if details and details["Taxonomy ID"] not in unique_tax_ids:
        unique_tax_ids.add(details["Taxonomy ID"])
        genome_data.append(details)

# Step 3: Save results
save_to_csv(genome_data, output_file)
