from PyPDF2 import PdfReader
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BloomForCausalLM
from transformers import BloomTokenizerFast

# Step 1: PDF Text Extraction
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        number_of_pages = len(pdf_reader.pages)
        text = ''
        for page_num in range(number_of_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

# Replace 'pdf_path' with the actual path to your PDF file
pdf_text = extract_text_from_pdf("./sample.pdf")

# Step 2: Tokenization
pretrained_model_name = "bigscience/bloom-560m"

tokenizer = BloomTokenizerFast.from_pretrained(pretrained_model_name)
tokenized_data = tokenizer(pdf_text, return_tensors='pt', padding=True, truncation=True)
print(tokenized_data, 'tokenized_data')

# Create a custom dataset
class CustomDataset(Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data['input_ids']
        self.attention_mask = tokenized_data['attention_mask']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx]}

dataset = CustomDataset(tokenized_data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Step 3: Fine-tuning
model = BloomForCausalLM.from_pretrained(pretrained_model_name)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 3

for epoch in range(num_epochs):
    total_loss = 0

    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}')

# Save the fine-tuned model
model.save_pretrained('bloom_fine_tuned_model')
tokenizer.save_pretrained('bloom_fine_tuned_model')
