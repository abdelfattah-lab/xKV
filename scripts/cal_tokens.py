import os
from transformers import AutoTokenizer

# Calcalate average tokens in gsm8k_5shot and bbh_3shot datasets
"""
cd 3rdparty/lm-evaluation-harness
python scripts/write_out.py \
    --tasks bbh \
    --sets test \
    --num_fewshot 3 \
    --num_examples 10 \
    --output_base_path bbh_3shot

python scripts/write_out.py \
    --tasks gsm8k \
    --sets test \
    --num_fewshot 5 \
    --num_examples 10 \
    --output_base_path gsm8k_5shot
"""

# Load the LLaMA 3 tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Set the path to the folder containing your text files
# folder_path = "3rdparty/lm-evaluation-harness/gsm8k_5shot"
folder_path = "3rdparty/lm-evaluation-harness/bbh_3shot"

# Get all files in the folder that end with .txt
file_list = [f for f in os.listdir(folder_path)]

# Initialize total token counter
total_tokens = 0

print("Token count per file:\n" + "-" * 30)

# Loop through each .txt file in the folder
for filename in file_list:
    file_path = os.path.join(folder_path, filename)
    
    # Read the file contents
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        
        # Encode the text using LLaMA 3 tokenizer and count tokens
        tokens = tokenizer.encode(text)
        token_count = len(tokens)
        
        # Add to total and print per-file token count
        total_tokens += token_count
        print(f"{filename}: {token_count} tokens")

# Print total token count across all files
print("-" * 30)
print(f"Total files in folder: {len(file_list)}")
print(f"Total tokens in folder: {total_tokens}")
print(f"Average tokens per file: {total_tokens / len(file_list):.2f}")
