import json
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Step 1: Load the JSON Data
with open("horizon_club_info.json", "r") as f:
    horizon_data = json.load(f)

# Prepare data for training in instruction-response format
training_data = [
    {
        "instruction": key,
        "response": value if isinstance(value, str) else json.dumps(value)
    }
    for key, value in horizon_data.items()
]

# Step 2: Prepare the Training Dataset
# Convert Data to Dataset Format
dataset = Dataset.from_list(training_data)

# Split Dataset into Training and Evaluation Sets
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset['train']
eval_dataset = dataset['test']

# Step 3: Initialize the Model and Tokenizer
model_name = "llama2"  # Replace with the name of the model you're using
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# Step 4: Tokenize the Dataset
def tokenize_function(examples):
    return tokenizer(examples["instruction"], truncation=True, padding="max_length", max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Step 5: Set Up Training Arguments
training_args = TrainingArguments(
    output_dir="./output",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2
)

# Step 6: Set Up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset
)

# Step 7: Train the Model
trainer.train()

# Step 8: Save the Fine-Tuned Model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Step 9: Load and Use the Fine-Tuned Model
fine_tuned_model = LlamaForCausalLM.from_pretrained("./fine_tuned_model")
fine_tuned_tokenizer = LlamaTokenizer.from_pretrained("./fine_tuned_model")

# Step 10: Generate a Response
input_text = "Tell me about Reality Check."
inputs = fine_tuned_tokenizer(input_text, return_tensors="pt")
outputs = fine_tuned_model.generate(**inputs)
response = fine_tuned_tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
