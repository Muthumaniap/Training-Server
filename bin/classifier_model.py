import string
import random
import torch
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, pipeline

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['label'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def prepare_training_data(intents_data):
    training_data = []
    for label in intents_data:
        for text in intents_data[label]:
            training_data.append({"label": label, "text": text})
    random.shuffle(training_data)
    return training_data

def preprocess_texts_and_labels(training_data, labelname2id):
    train_texts = []
    train_labels = []
    for entry in training_data:
        train_texts.append(entry['text'].translate(str.maketrans('', '', string.punctuation)).lower())
        train_labels.append(labelname2id[entry['label']])
    return train_texts, train_labels

def prepare_test_data(intents_data, test_texts, test_labels):
    label_names = list(intents_data.keys())
    return label_names, test_texts, test_labels

def fine_tune_models(intents_data, model_id, train_texts, dev_texts, train_labels, dev_labels, test_texts, test_labels, id2labelname):
    print(f"*** {model_id} ***")
    tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=len(intents_data.keys()))
    train_encodings = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")
    dev_encodings = tokenizer(dev_texts, padding=True, truncation=True, return_tensors="pt")
    test_encodings = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt")
    train_dataset = ClassificationDataset(train_encodings, train_labels)
    dev_dataset = ClassificationDataset(dev_encodings, dev_labels)
    test_dataset = ClassificationDataset(test_encodings, test_labels)
    training_args = TrainingArguments(
        output_dir='./data/results/' + model_id, num_train_epochs=10, per_device_train_batch_size=16, per_device_eval_batch_size=64, warmup_steps=int(len(train_dataset) / 16), weight_decay=0.01, logging_dir='./logs/' + model_id,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=10,
        load_best_model_at_end=True,
        no_cuda=False
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )
    trainer.train()
    trainer.save_model('./data/models/' + model_id)
    tokenizer.save_pretrained('./data/models/' + model_id)
    test_results = trainer.evaluate(test_dataset)
    with open('./data/models/' + model_id + "/config.json", 'r') as config_file:
        config = json.load(config_file)
        config.update({"id2label": id2labelname})
    with open('./data/models/' + model_id + "/config.json", 'w') as newconfig:
        json.dump(config, newconfig, indent=4)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

def classify_text_with_models(input_text, model_id, label_names):
    input_text = input_text.translate(str.maketrans('', '', string.punctuation)).lower()
    print(model_id, " : ", end="\n")
    pipe = pipeline("text-classification",
                    model="./data/models/" + model_id,
                    device='cpu',
                    padding=True,
                    truncation=True,
                    top_k=None)
    data = pipe(input_text)
    for i in data[0]:
        # print(i, end='\t')
        if i['label'] in label_names:
            # print("Result : ", i)
            return i
            # break

if __name__ == "__main__":
    # Step 1: Read the JSON file
    with open("data/training_data.json") as file:
        intents_data = json.load(file)

    # Step 2: Create label mappings
    id2labelname = {id: name for id, name in enumerate(intents_data.keys())}
    labelname2id = {name: id for id, name in enumerate(intents_data.keys())}

    # Preparing training data
    training_data = prepare_training_data(intents_data)
    train_texts, train_labels = preprocess_texts_and_labels(training_data, labelname2id)

    model_id = "bert-base-uncased"

    # Preparing test data
    test_texts = ["i have told you so many times not to call me on this matter just get lost"]
    test_labels = [4]
    label_names, test_texts, test_labels = prepare_test_data(intents_data, test_texts, test_labels)

    # Splitting data into train and dev sets
    train_texts, dev_texts, train_labels, dev_labels = train_test_split(train_texts, train_labels, test_size=0.1, shuffle=True, random_state=1)

    # Fine-tuning models
    # fine_tune_models(intents_data, model_id, train_texts, dev_texts, train_labels, dev_labels, test_texts, test_labels, id2labelname)

    # Classifying text with models
    results = classify_text_with_models("How can I help you?", model_id, label_names)
    # pass the results to anywhere
    print("Label predicted : ", results)