import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments

# Load dataset
df = pd.read_csv("your_dataset.csv")  # Ensure it has 'Genomic Sequence' and a corresponding 'Label'

# Select relevant columns
df = df[['Genomic Sequence', 'Label']]
df.columns = ['sequence', 'label']  # Rename for consistency

# Split dataset (80% train, 10% dev, 10% test)
train, temp = train_test_split(df, test_size=0.2, random_state=42)
dev, test = train_test_split(temp, test_size=0.5, random_state=42)

# Save CSVs
train.to_csv("train.csv", index=False)
dev.to_csv("dev.csv", index=False)
test.to_csv("test.csv", index=False)

print("Dataset split completed: train.csv, dev.csv, test.csv created.")

def load_model_and_tokenizer():
    print("Loading DNA-BERT2 model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    return model, tokenizer

def compute_metrics(pred):
    """
    Compute metrics for model evaluation
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# python train.py \
#     --model_name_or_path zhihan1996/DNABERT-2-117M \
#     --data_path  ${DATA_PATH} \
#     --kmer -1 \
#     --run_name DNABERT2_${DATA_PATH} \
#     --model_max_length ${MAX_LENGTH} \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 1 \
#     --learning_rate ${LR} \
#     --num_train_epochs 5 \
#     --fp16 \
#     --save_steps 200 \
#     --output_dir output/dnabert2 \
#     --evaluation_strategy steps \
#     --eval_steps 200 \
#     --warmup_steps 50 \
#     --logging_steps 100 \
#     --overwrite_output_dir True \
#     --log_level info \
#     --find_unused_parameters False



# export DATA_PATH="./sample_data"
# export MAX_LENGTH=100  # 0.25 * 400 (for synthetic data with 400 bases)
# export LR=3e-5

# # Step 4: Run the training
# echo "Running DNA-BERT2 fine-tuning..."

# # Check if GPU is available
# if [ $(python -c "import torch; print(torch.cuda.device_count())") -gt 0 ]; then
#     echo "GPU detected, using GPU for training..."
#     # Check if multiple GPUs are available
#     num_gpu=$(python -c "import torch; print(torch.cuda.device_count())")
#     if [ $num_gpu -gt 1 ]; then
#         echo "Multiple GPUs detected ($num_gpu), using DistributedDataParallel..."
#         torchrun --nproc_per_node=$num_gpu train.py \
#             --model_name_or_path zhihan1996/DNABERT-2-117M \
#             --data_path ${DATA_PATH} \
#             --kmer -1 \
#             --run_name DNABERT2_viral_dna \
#             --model_max_length ${MAX_LENGTH} \
#             --per_device_train_batch_size 8 \
#             --per_device_eval_batch_size 16 \
#             --gradient_accumulation_steps 1 \
#             --learning_rate ${LR} \
#             --num_train_epochs 5 \
#             --fp16 \
#             --save_steps 200 \
#             --output_dir output/dnabert2 \
#             --evaluation_strategy steps \
#             --eval_steps 200 \
#             --warmup_steps 50 \
#             --logging_steps 100 \
#             --overwrite_output_dir True \
#             --log_level info \
#             --find_unused_parameters False
#     else
#         echo "Single GPU detected, using DataParallel..."
#         python train.py \
#             --model_name_or_path zhihan1996/DNABERT-2-117M \
#             --data_path ${DATA_PATH} \
#             --kmer -1 \
#             --run_name DNABERT2_viral_dna \
#             --model_max_length ${MAX_LENGTH} \
#             --per_device_train_batch_size 8 \
#             --per_device_eval_batch_size 16 \
#             --gradient_accumulation_steps 1 \
#             --learning_rate ${LR} \
#             --num_train_epochs 5 \
#             --fp16 \
#             --