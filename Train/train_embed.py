# import pandas as pd
# from collections import Counter
# import torch
# from torch.utils.data import TensorDataset, DataLoader
# from transformers import BertForMaskedLM, BertConfig, AdamW
# import random
# import os
# import json
# import math


# save_path = "saved_models/viral_bert"
# os.makedirs(save_path, exist_ok=True)  # Ensure directory exists

# def save_model(model, epoch):
#     """Save model checkpoint."""
#     torch.save(model.state_dict(), f"{save_path}/bert_epoch_{epoch}.pt")
#     print(f"Model saved at epoch {epoch}")

# def load_model(model, epoch):
#     """Load model checkpoint."""
#     model.load_state_dict(torch.load(f"{save_path}/bert_epoch_{epoch}.pt"))
#     model.to(device)
#     print(f"Loaded model from epoch {epoch}")


# # Load CSV file
# def load_genomes_from_csv(csv_file):
#     df = pd.read_csv(csv_file)
#     sequences = df["Genomic Sequence"].dropna().tolist()  # Ensure no NaN values
#     return sequences

# # Example Usage
# csv_file = "../Datasets/coronaviridae_sequences_with_genomes_final.csv"
# genome_sequences = load_genomes_from_csv(csv_file)
# print(f"Loaded {len(genome_sequences)} genome sequences.")

# def kmer_tokenizer(sequence, k=6):
#     """Generate k-mers from a sequence."""
#     return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]



# from collections import Counter

# def build_kmer_vocab(sequences, k=6, vocab_size=30000):
#     """Build a vocabulary from k-mers in sequences."""
#     kmer_counts = Counter()
#     for seq in sequences:
#         kmer_counts.update(kmer_tokenizer(seq, k))

#     most_common_kmers = [kmer for kmer, _ in kmer_counts.most_common(vocab_size - 2)]
    
#     vocab = {"[PAD]": 0, "[MASK]": 1}  # Add special tokens
#     vocab.update({kmer: i+2 for i, kmer in enumerate(most_common_kmers)})  

#     return vocab

# # Build vocab
# vocab = build_kmer_vocab(genome_sequences, k=6, vocab_size=30000)
# print(f"Vocabulary size: {len(vocab)}")





# def encode_sequence(sequence, vocab, k=6):
#     """Convert a sequence into token IDs based on the k-mer vocabulary."""
#     kmers = kmer_tokenizer(sequence, k)
#     return [vocab.get(kmer, 0) for kmer in kmers]  # Use [PAD]=0 for unknown kmers

# # Example encoding
# encoded_sequences = [encode_sequence(seq, vocab) for seq in genome_sequences]

# # Convert to PyTorch tensors
# max_length = max(len(seq) for seq in encoded_sequences)  # Adjust padding length
# padded_sequences = [seq + [0] * (max_length - len(seq)) for seq in encoded_sequences]
# input_ids = torch.tensor(padded_sequences)

# print(f"Input tensor shape: {input_ids.shape}")




# # def mask_tokens(input_ids, mask_prob=0.15):
# #     """Randomly mask tokens for MLM."""
# #     labels = input_ids.clone()
# #     mask = torch.full(labels.shape, mask_prob).bernoulli().bool()  # 15% masking
# #     input_ids[mask] = vocab["[MASK]"]  # Replace with mask token
# #     return input_ids, labels

# # # Apply masking
# # masked_input_ids, labels = mask_tokens(input_ids)



# # config = BertConfig(
# #     vocab_size=len(vocab),
# #     hidden_size=256,  # Reduce if memory issues arise
# #     num_hidden_layers=4,  # Fewer layers for smaller model
# #     num_attention_heads=6,
# #     max_position_embeddings=max_length,
# # )

# # model = BertForMaskedLM(config)
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model.to(device)



# # # Dataloader
# # batch_size = 8
# # dataset = TensorDataset(masked_input_ids, labels)
# # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # # Optimizer
# # optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

# # from tqdm import tqdm

# # epochs = 5
# # total_batches = len(dataloader)

# # for epoch in range(epochs):
# #     model.train()
# #     total_loss = 0
# #     progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

# #     for batch in progress_bar:
# #         batch = [b.to(device) for b in batch]
# #         input_ids, labels = batch
# #         optimizer.zero_grad()
        
# #         outputs = model(input_ids, labels=labels)
# #         loss = outputs.loss
# #         loss.backward()
# #         optimizer.step()

# #         total_loss += loss.item()
# #         progress_bar.set_postfix(loss=loss.item())

# #     avg_loss = total_loss / total_batches
# #     print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

# #     save_model(model, epoch + 1)

# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader
# from transformers import BertForMaskedLM, BertConfig, AdamW
# from tqdm import tqdm
# import gc

# # -------------------- Data Preparation --------------------

# def mask_tokens(input_ids, mask_prob=0.15, mask_token_id=103):  
#     """Randomly mask tokens for MLM."""
#     labels = input_ids.clone()
#     mask = torch.full(labels.shape, mask_prob).bernoulli().bool()  # 15% masking
#     input_ids = torch.where(mask, torch.full_like(input_ids, mask_token_id), input_ids)  # Avoid shape issues
#     return input_ids, labels

# # Example Data (Replace with actual dataset)
# vocab = {"[MASK]": 103}  # Mask token ID for BERT
# max_length = 512  # Adjust as needed
# input_ids = torch.randint(0, 30522, (1000, max_length))  # Fake dataset for testing

# # Apply masking
# masked_input_ids, labels = mask_tokens(input_ids)

# # -------------------- Model & Config --------------------

# config = BertConfig(
#     vocab_size=26609,  # Adjust based on actual vocab size
#     hidden_size=256,  # Reduce if memory issues arise
#     num_hidden_layers=4,  
#     num_attention_heads=8,
#     max_position_embeddings=512,
# )

# model = BertForMaskedLM(config)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # -------------------- Training Setup --------------------

# batch_size = 8
# dataset = TensorDataset(masked_input_ids, labels)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

# scaler = torch.cuda.amp.GradScaler()  # Mixed Precision Safety

# epochs = 5
# total_batches = len(dataloader)

# # -------------------- Training Loop with AMP --------------------
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

#     for batch in progress_bar:
#         batch = [b.to(device) for b in batch]
#         input_ids, labels = batch
#         optimizer.zero_grad()

#         with torch.cuda.amp.autocast():  # Enables FP16 training
#             outputs = model(input_ids, labels=labels)
#             loss = outputs.loss

#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         total_loss += loss.item()
#         progress_bar.set_postfix(loss=loss.item())

#     avg_loss = total_loss / total_batches
#     print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

#     # Free CUDA Memory
#     torch.cuda.empty_cache()
#     gc.collect()

#     # Save model
#     torch.save(model.state_dict(), f"bert_mlm_epoch_{epoch+1}.pt")

# #__________________________________________________________________

# def get_embeddings(sequence):
#     model.eval()
#     input_ids = torch.tensor(encode_sequence(sequence, vocab)).unsqueeze(0).to(device)
#     with torch.no_grad():
#         outputs = model.bert(input_ids)
#     return outputs.last_hidden_state.cpu().numpy()  # Shape: (1, seq_length, hidden_size)

# # Example embedding extraction
# embedding = get_embeddings(genome_sequences[0])
# print("Embedding shape:", embedding.shape)



# import math

# def evaluate_perplexity(model, dataloader):
#     """Compute perplexity of the model."""
#     model.eval()
#     total_loss = 0
#     total_batches = len(dataloader)

#     with torch.no_grad():
#         for batch in dataloader:
#             batch = [b.to(device) for b in batch]
#             input_ids, labels = batch
#             outputs = model(input_ids, labels=labels)
#             total_loss += outputs.loss.item()

#     avg_loss = total_loss / total_batches
#     ppl = math.exp(avg_loss)
#     return avg_loss, ppl

# # Evaluate after training
# avg_loss, ppl = evaluate_perplexity(model, dataloader)
# print(f"Evaluation Loss: {avg_loss:.4f} | Perplexity: {ppl:.4f}")





# import torch.nn.functional as F

# def compute_masked_accuracy(model, dataloader):
#     """Evaluate masked token prediction accuracy."""
#     model.eval()
#     correct, total = 0, 0

#     with torch.no_grad():
#         for batch in dataloader:
#             batch = [b.to(device) for b in batch]
#             input_ids, labels = batch
#             outputs = model(input_ids).logits  # Get predictions

#             # Only consider masked positions
#             mask = labels != -100
#             predictions = outputs.argmax(dim=-1)  # Get best predicted token

#             correct += (predictions[mask] == labels[mask]).sum().item()
#             total += mask.sum().item()

#     accuracy = correct / total if total > 0 else 0
#     return accuracy

# # Compute masked accuracy
# masked_accuracy = compute_masked_accuracy(model, dataloader)
# print(f"Masked Token Prediction Accuracy: {masked_accuracy:.4f}")



# # Load the best checkpoint (e.g., last epoch)
# best_epoch = 5  # Change based on which epoch performed best
# load_model(model, best_epoch)

# # Evaluate
# avg_loss, ppl = evaluate_perplexity(model, dataloader)
# masked_accuracy = compute_masked_accuracy(model, dataloader)

# print(f"Final Evaluation -> Loss: {avg_loss:.4f} | PPL: {ppl:.4f} | Accuracy: {masked_accuracy:.4f}")




import pandas as pd
from collections import Counter
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForMaskedLM, BertConfig, AdamW, get_linear_schedule_with_warmup
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Load CSV File
def load_genomes_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    sequences = df["Genomic Sequence"].dropna().tolist()  # Ensure no NaN values
    return sequences

csv_file = "../Datasets/coronaviridae_sequences_with_genomes_final.csv"
genome_sequences = load_genomes_from_csv(csv_file)

# K-mer Tokenizer
def kmer_tokenizer(sequence, k=6):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

# Build Vocabulary
def build_kmer_vocab(sequences, k=6, vocab_size=30000):
    kmer_counts = Counter()
    for seq in sequences:
        kmer_counts.update(kmer_tokenizer(seq, k))
    most_common_kmers = [kmer for kmer, _ in kmer_counts.most_common(vocab_size - 2)]
    
    vocab = {"[PAD]": 0, "[MASK]": 1}
    vocab.update({kmer: i+2 for i, kmer in enumerate(most_common_kmers)})
    return vocab

vocab = build_kmer_vocab(genome_sequences, k=6, vocab_size=30000)
print(f"Vocabulary size: {len(vocab)}")

# Encode Sequences
def encode_sequence(sequence, vocab, k=6):
    kmers = kmer_tokenizer(sequence, k)
    return [vocab.get(kmer, 0) for kmer in kmers]

encoded_sequences = [encode_sequence(seq, vocab) for seq in genome_sequences]

# Create Overlapping Windows
def create_overlapping_windows(sequence, window_size=512, stride=256):
    windows = [sequence[i:i+window_size] for i in range(0, len(sequence), stride)]
    if len(windows[-1]) < window_size:
        windows[-1] += [0] * (window_size - len(windows[-1]))  # Pad last window
    return windows

windowed_sequences = [create_overlapping_windows(seq) for seq in encoded_sequences]
flattened_windows = [win for seq in windowed_sequences for win in seq]
print(f"Total windows: {len(flattened_windows)}")

# Masking Function
def mask_tokens(input_ids, mask_prob=0.15):
    labels = input_ids.clone()
    mask = torch.full(labels.shape, mask_prob).bernoulli().bool()
    input_ids[mask] = vocab["[MASK]"]
    return input_ids, labels

# Custom Dataset Class
class GenomicDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_ids = torch.tensor(self.sequences[idx])
        input_ids, labels = mask_tokens(input_ids)
        return input_ids, labels

dataset = GenomicDataset(flattened_windows)

# Dynamic Padding Function
def collate_fn(batch):
    input_ids, labels = zip(*batch)
    input_ids = [torch.tensor(seq) for seq in input_ids]
    labels = [torch.tensor(lbl) for lbl in labels]
    
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=vocab["[PAD]"])
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 is ignored in loss
    
    attention_mask = (input_ids != vocab["[PAD]"]).long()
    return input_ids, attention_mask, labels

# Dataloader
batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Model Configuration
config = BertConfig(
    vocab_size=len(vocab),
    hidden_size=288,
    num_hidden_layers=4,
    num_attention_heads=6,  # 288 is divisible by 6
    max_position_embeddings=512
)
model = BertForMaskedLM(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
total_steps = len(dataloader) * 5  # 5 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# FP16 Mixed Precision Training
scaler = GradScaler()

epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

    for input_ids, attention_mask, labels in progress_bar:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
    
    # Save Model
    model.save_pretrained(f"viral_bert_epoch_{epoch+1}")