from transformers import LlamaForCausalLM, Trainer, TrainingArguments
from datasets import load_from_disk

# Load the model
model = LlamaForCausalLM.from_pretrained('google/gemma-2b-it')

# Load the tokenized dataset
tokenized_dataset = load_from_disk("./tokenized_dataset")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Start training
trainer.train()
