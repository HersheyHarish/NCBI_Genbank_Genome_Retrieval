import time
import csv
import os
import sys
from tqdm import tqdm
from Bio import Entrez
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
NCBI_API_KEY = os.getenv("NCBI_API_KEY")

#
Entrez.email = "hsundar@ucdavis.edu"  # Replace with your email
Entrez.api_key = NCBI_API_KEY  # Enable API key usage

OUTPUT_FILE = "coronaviridae_no_sars.csv"
CHECKPOINT_FILE = "checkpoint.txt"

BATCH_SIZE = 1000   
FIELDS = ["Accession", "Description", "Organism", "Length", "Source", "Collection Date", "Isolate",]

# Function to save the last retrieved index
def save_checkpoint(last_index):
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(last_index))

# Function to load last checkpoint
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return int(f.read().strip())
    return 0

def search_coronaviridae_genomes():
    query = 'Coronaviridae [Organism] AND Complete Genome [Title] NOT Severe acute respiratory syndrome-related coronavirus'
    handle = Entrez.esearch(db="nucleotide", term=query, retmax=100000000, usehistory="y")  #
    results = Entrez.read(handle)
    handle.close()
    return results["IdList"], results["WebEnv"], results["QueryKey"]

def fetch_metadata(id_list, webenv, query_key, start_index):
    with open(OUTPUT_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        
        if start_index == 0 and not os.path.exists(OUTPUT_FILE):
            writer.writerow(FIELDS)

        with tqdm(total=len(id_list) - start_index, desc="Fetching metadata", initial=start_index) as pbar:
            for start in range(start_index, len(id_list), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(id_list))
                fetch_handle = Entrez.efetch(
                    db="nucleotide",
                    rettype="gb",
                    retmode="xml",
                    webenv=webenv,
                    query_key=query_key,
                    retstart=start,
                    retmax=BATCH_SIZE
                )
                records = Entrez.read(fetch_handle)
                fetch_handle.close()

                for record in records:
                    accession = record.get("GBSeq_primary-accession", "Unknown Accession")
                    description = record.get("GBSeq_definition", "No Description")
                    organism = record.get("GBSeq_organism", "Unknown Organism")
                    length = int(record.get("GBSeq_length", 0))
                    source = record.get("GBSeq_source", "Unknown Source")
                    
                    collection_date, isolate = "Unknown", "Unknown"
                    if "GBSeq_feature-table" in record:
                        for feature in record["GBSeq_feature-table"]:
                            if feature["GBFeature_key"] == "source":
                                for qualifier in feature.get("GBFeature_quals", []):
                                    if qualifier["GBQualifier_name"] == "collection_date":
                                        collection_date = qualifier["GBQualifier_value"]
                                    elif qualifier["GBQualifier_name"] == "isolate":
                                        isolate = qualifier["GBQualifier_value"]


                    writer.writerow([accession, description, organism, length, source, collection_date, isolate,])

                save_checkpoint(end)  # Save progress
                pbar.update(end - start)

                print(f"Batch {start}-{end} processed. Progress saved.")

# Main execution
if __name__ == "__main__":
    print("Searching for Coronaviridae genomes...")
    ids, webenv, query_key = search_coronaviridae_genomes()

    total_records = len(ids)
    print(f"Found {total_records} genome records.")

    # Resume from last checkpoint
    last_index = load_checkpoint()
    print(f"Resuming from index {last_index}...")

    fetch_metadata(ids, webenv, query_key, last_index)

    print(f"All records processed. Data saved in {OUTPUT_FILE}.")