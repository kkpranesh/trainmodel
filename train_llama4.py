from datasets import load_dataset
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer, DataCollatorForLanguageModeling
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
  dataset = load_dataset("text", data_files={"train": "Input.txt"})
  tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b") # or google/gemma-7b
  print('here')
  tokenizer.pad_token = tokenizer.eos_token #Gemma needs a padding token.
  print('here1')
  print(dir(tokenizer))
  def tokenize_function(examples):
      return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512) # adjust max_length
  print('here2')

  tokenized_datasets = dataset.map(tokenize_function, batched=True)
  print('here3')

  data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
  print('here4')

  model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
  print('here5')

  training_args = TrainingArguments(
      output_dir="./gemma_finetuned",
      overwrite_output_dir=True,
      num_train_epochs=3,  # Adjust as needed
      per_device_train_batch_size=4,  # Adjust based on GPU memory
      save_steps=1000,
      save_total_limit=2,
      logging_dir="./logs",
      logging_steps=10,
      learning_rate=2e-5, # Adjust learning rate
      weight_decay=0.01,
      push_to_hub=False, #set to true if desired.
      #fp16=True, #Enable if you have a GPU supporting fp16.
      #bf16=True, #Enable if you have a GPU supporting bf16.
  )
  print('here6')

  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=tokenized_datasets["train"],
      data_collator=data_collator,
  )
  print('here7')

  trainer.train()
  print('here8')

  trainer.save_model("./gemma_finetuned")
except Exception as e:
    logging.error(f"An error occurred: {e}")