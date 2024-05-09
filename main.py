import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples['sms'], padding="max_length", truncation=True)

dataset = load_dataset("sms_spam")
dataset = dataset.map(tokenize_function, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

train_test_split = dataset['train'].train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

instruction = "Classify whether the following message is spam or not: "
train_with_instructions = train_dataset.map(lambda x: {'sms': instruction + x['sms']})
test_with_instructions = test_dataset.map(lambda x: {'sms': instruction + x['sms']})

train_with_instructions = train_with_instructions.map(tokenize_function, batched=True)
test_with_instructions = test_with_instructions.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
)

for dataset_name, train_data, test_data in [("Base Dataset", train_dataset, test_dataset), ("Instruction Dataset", train_with_instructions, test_with_instructions)]:
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data
    )

    trainer.train()

    model.save_pretrained(f'./saved_models/model_{dataset_name.replace(" ", "_").lower()}')

    results = trainer.evaluate()

    print(f"Results for {dataset_name}: {results}")
