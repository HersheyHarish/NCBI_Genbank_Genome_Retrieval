import pandas as pd
from collections import Counter
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertForMaskedLM, BertConfig, AdamW
import random
import os
import json
import math

# Load CSV file
def load_genomes_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    sequences = df["Genomic Sequence"].dropna().tolist()  # Ensure no NaN values
    return sequences

# Example Usage
csv_file = "../Datasets/coronaviridae_sequences_with_genomes_final.csv"
genome_sequences = load_genomes_from_csv(csv_file)
print(f"Loaded {len(genome_sequences)} genome sequences.")

def kmer_tokenizer(sequence, k=6):
    """Generate k-mers from a sequence."""
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

# Build vocabulary with frequency filtering
k = 6
counter = Counter()
for seq in genome_sequences:
    counter.update(kmer_tokenizer(seq, k))

# Remove rare k-mers (appear only once)
min_freq = 2
filtered_kmers = {kmer for kmer, count in counter.items() if count >= min_freq}

# Assign unique indices to k-mers
vocab = {kmer: idx for idx, kmer in enumerate(filtered_kmers)}

print(f"Filtered Vocabulary Size: {len(vocab)}")

# Define padding and mask indices
PAD_IDX = len(vocab)  # Padding index
MASK_IDX = len(vocab) + 1  # Mask index (next available index after PAD_IDX)

def encode_sequence(sequence, vocab, k=6):
    """Convert a sequence into token IDs using the vocabulary."""
    return [vocab[kmer] for kmer in kmer_tokenizer(sequence, k) if kmer in vocab]

# Save encoded sequences in batches to avoid memory overload
encoded_sequences = []
for seq in genome_sequences:
    encoded_sequences.append(encode_sequence(seq, vocab))

# Convert to a tensor dataset
max_len = max(len(seq) for seq in encoded_sequences)  # Find longest sequence

# Chunk long sequences before padding
MAX_SEQ_LENGTH = 512

def chunk_sequence(seq, max_length=MAX_SEQ_LENGTH):
    """Split a long sequence into smaller overlapping chunks."""
    return [seq[i:i + max_length] for i in range(0, len(seq), max_length)]

chunked_sequences = []
for seq in encoded_sequences:
    chunked_sequences.extend(chunk_sequence(seq))

# Convert to tensor dataset
padded_sequences = torch.full((len(chunked_sequences), MAX_SEQ_LENGTH), PAD_IDX, dtype=torch.long)
for i, seq in enumerate(chunked_sequences):
    padded_sequences[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

# Create attention mask (1 for real tokens, 0 for padding)
attention_mask = (padded_sequences != PAD_IDX).long()

# Convert to TensorDataset
dataset = TensorDataset(padded_sequences, attention_mask)

# Create DataLoader
batch_size = 16  # Adjust based on GPU memory
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"Dataset shape after chunking: {padded_sequences.shape}")
print(f"Training dataset size after chunking: {len(dataset)}")
print(f"Batch size: {batch_size}")

torch.manual_seed(42)
random.seed(42)

# Define BERT model configuration
config = BertConfig(
    vocab_size=len(vocab) + 2,  # Adding PAD and MASK tokens
    hidden_size=256,  # Adjust based on memory constraints
    num_hidden_layers=4,
    num_attention_heads=8,
    intermediate_size=512
)

# Initialize BERT model with the custom configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForMaskedLM(config)
model.to(device)

# Define Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# # Training Loop
# epochs = 3
# loss_fn = torch.nn.CrossEntropyLoss()

# for epoch in range(epochs):
#     model.train()
#     total_loss = 0

#     for batch in dataloader:
#         inputs, attention_mask = [b.to(device) for b in batch]

#         # Randomly mask some tokens (80% [MASK], 10% random token, 10% unchanged)
#         labels = inputs.clone()
#         mask_prob = 0.15  # 15% of tokens are masked
#         mask_tensor = torch.rand(labels.shape, device=device) < mask_prob
#         random_tensor = (torch.rand(labels.shape, device=device) < 0.1) & ~mask_tensor
#         unchanged_tensor = ~mask_tensor & ~random_tensor

#         labels[~mask_tensor] = -100  # Ignore non-masked tokens in loss
#         labels[random_tensor] = torch.randint(len(vocab), labels.shape, device=device)[random_tensor]  # Random token
#         labels[unchanged_tensor] = inputs[unchanged_tensor]  # Keep unchanged tokens as they are

#         optimizer.zero_grad()
#         outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss

#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

# print("Training complete!")


# Training Loop
epochs = 3
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in dataloader:
        inputs, attention_mask = [b.to(device) for b in batch]

        # Randomly mask some tokens (80% [MASK], 10% random token, 10% unchanged)
        labels = inputs.clone()
        mask_prob = 0.15  # 15% of tokens are masked
        mask_tensor = torch.rand(labels.shape, device=device) < mask_prob
        random_tensor = (torch.rand(labels.shape, device=device) < 0.1) & ~mask_tensor
        unchanged_tensor = ~mask_tensor & ~random_tensor

        labels[~mask_tensor] = -100  # Ignore non-masked tokens in loss
        labels[random_tensor] = torch.randint(len(vocab), labels.shape, device=device)[random_tensor]  # Random token
        labels[unchanged_tensor] = inputs[unchanged_tensor]  # Keep unchanged tokens as they are

        optimizer.zero_grad()
        outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

print("Training complete!")

# Save trained model and vocabulary
def save_model(model, vocab, save_dir="viralgpt_model"):
    os.makedirs(save_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(save_dir, "model.pt")
    torch.save(model.state_dict(), model_path)

    # Save vocabulary
    vocab_path = os.path.join(save_dir, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=4)

    print(f"Model saved to {model_path}")
    print(f"Vocabulary saved to {vocab_path}")

# Save the model
save_model(model, vocab)

# Metrics calculation functions
def save_metrics(metrics, filename="metrics.json"):
    with open(filename, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {filename}")


def masked_prediction_accuracy(model, dataloader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            batch = [item.to(device) for item in batch]
            inputs, targets = batch

            outputs = model(input_ids=inputs).logits
            predictions = torch.argmax(outputs, dim=-1)

            mask_indices = targets != -100
            correct += (predictions[mask_indices] == targets[mask_indices]).sum().item()
            total += mask_indices.sum().item()

    accuracy = correct / total if total > 0 else 0
    return accuracy

def compute_perplexity(model, dataloader, device):
    model.eval()
    total_loss, total_count = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            batch = [item.to(device) for item in batch]
            inputs, targets = batch

            outputs = model(input_ids=inputs, labels=targets)
            loss = outputs.loss

            total_loss += loss.item() * inputs.size(0)
            total_count += inputs.size(0)

    avg_loss = total_loss / total_count
    perplexity = math.exp(avg_loss)
    return perplexity

# Compute and save metrics
accuracy = masked_prediction_accuracy(model, dataloader, device)
perplexity = compute_perplexity(model, dataloader, device)

metrics = {
    "Masked Prediction Accuracy": accuracy,
    "Perplexity": perplexity
}

save_metrics(metrics)