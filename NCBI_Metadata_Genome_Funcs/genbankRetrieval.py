from Bio import Entrez, SeqIO
import time, csv

Entrez.email = "hsundar@ucdavis.edu"  

def search_complete_genomes(query, retmax):
    """
    Search the NCBI genbank nucleotide database for complete genomes.

    Input:
        Query(Str): Query to search for
        Retmax(int): num records to search (0 defaults to all)

    Returns:
        [Count] : total count for query
        [IdList]: list of virus id's
    """
    try:
        with Entrez.esearch(db="nucleotide", term=query, retmax=retmax) as handle:
            record = Entrez.read(handle)
        return record["Count"], record["IdList"]
    except Exception as e:
        print(f"Error searching for complete genomes: {e}")
        return []


def fetch_genome_details(accession_id):
    """
    Method to fetch metadata and genome sequence for a given accession ID

    Input: 
        Accession_id(int): id to look up in database
  
    """
    try:
        with Entrez.efetch(db = "nucleotide", id = accession_id, rettype = "gb", retmode="text") as handle:
            record = SeqIO.read(handle, "genbank")

        collection_date = "Unknown"
        isolation_source = "Unknown"

        for feature in record.features:
            if feature.type == "source":
                collection_date = feature.qualifiers.get("collection_date", ["Unknown"])[0]
                isolation_source = feature.qualifiers.get("isolation_source", ["Unknown"])[0]

        return {
            "Accession": record.id,
            "Description": record.description,
            "Organism": record.annotations.get("organism", "Unknown"),
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


def save_to_csv(data, filename):
    """Save genome data to a CSV file."""
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Accession", "Description", "Organism", "Length", "Source", "Taxonomy","Collection Date", "Isolation Source","Sequence"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Data saved to {filename}.")


def main():
    query = "Coronaviridae[Organism] AND complete genome[Title]"
    retmax = 10  
    total_count, record_ids = search_complete_genomes(query, retmax=retmax)

    print(f"For query: {query}, found {total_count} records.")
    print("Fetching genome details...")

    genome_data = []
    for idx, record_id in enumerate(record_ids):
        print(f"Fetching record {idx + 1}/{len(record_ids)}: {record_id}")
        details = fetch_genome_details(record_id)
        if details:
            genome_data.append(details)
        time.sleep(1)  # Avoid exceeding NCBI rate limits

    save_to_csv(genome_data, "coronaviridae_test_run.csv")


if __name__ == "__main__":
    main()