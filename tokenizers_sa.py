from datasets import Dataset
# from transformers import LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load your text file and clean up newlines if necessary
file_path = "./input.txt"
with open(file_path, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f.readlines()]

# Create a dataset from the text data
data = {"text": lines}
dataset = Dataset.from_dict(data)

# Tokenize the dataset using the LlamaTokenizer
model_name = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenized_dataset = dataset.map(
    lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=512),
    batched=True
)

# Save the tokenized dataset to disk
tokenized_dataset.save_to_disk("./tokenized_dataset")
