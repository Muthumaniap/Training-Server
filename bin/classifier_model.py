import logging.config

import json
import random
import string
import torch

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    pipeline,
)

logging = logging.getLogger()


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["label"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def prepare_training_data(intents_data):
    """
    Prepares the training data for the classifier model.

    Args:
        intents_data (dict): A dictionary containing the intents data.

    Returns:
        list: A list of dictionaries with "label" and "text" keys for each training data entry.
    """
    logging.info("Preparing training data")
    training_data = []
    for label in intents_data:
        for text in intents_data[label]:
            training_data.append({"label": label, "text": text})
    random.shuffle(training_data)
    return training_data


def preprocess_texts_and_labels(training_data, labelname2id):
    """
    Preprocesses the texts and labels in the given training data.

    Args:
        training_data (list): A list of dictionaries representing the training data. Each dictionary should have "text" and "label" keys.
        labelname2id (dict): A dictionary mapping label names to their corresponding IDs.

    Returns:
        tuple: A tuple containing two lists - train_texts and train_labels. train_texts contains the preprocessed texts from the training data, with punctuation removed and converted to lowercase. train_labels contains the corresponding labels converted to their respective IDs using the labelname2id dictionary.
    """
    logging.info("Preprocessing texts and labels")
    train_texts = []
    train_labels = []
    for entry in training_data:
        train_texts.append(
            entry["text"].translate(str.maketrans("", "", string.punctuation)).lower()
        )
        train_labels.append(labelname2id[entry["label"]])
    return train_texts, train_labels


def train_model(
    intents_data,
    model_id,
    train_texts,
    dev_texts,
    train_labels,
    dev_labels,
    test_texts,
    test_labels,
    id2labelname,
):
    """
    Trains a model using the provided training data and saves the trained model and tokenizer.
    
    Args:
        intents_data (dict): A dictionary containing the training data for each intent.
        model_id (str): The ID of the model to be trained.
        train_texts (list): A list of texts for training.
        dev_texts (list): A list of texts for development.
        train_labels (list): A list of labels for training.
        dev_labels (list): A list of labels for development.
        test_texts (list): A list of texts for testing.
        test_labels (list): A list of labels for testing.
        id2labelname (dict): A dictionary mapping label IDs to label names.
        
    Returns:
        None
    """
    logging.info("Training model: " + model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(intents_data.keys())
    )

    train_encodings = tokenizer(
        train_texts, padding=True, truncation=True, return_tensors="pt"
    )
    dev_encodings = tokenizer(
        dev_texts, padding=True, truncation=True, return_tensors="pt"
    )
    test_encodings = tokenizer(
        test_texts, padding=True, truncation=True, return_tensors="pt"
    )

    train_dataset = ClassificationDataset(train_encodings, train_labels)
    dev_dataset = ClassificationDataset(dev_encodings, dev_labels)
    test_dataset = ClassificationDataset(test_encodings, test_labels)

    training_args = TrainingArguments(
        output_dir="data/results/" + model_id,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=int(len(train_dataset) / 16),
        weight_decay=0.01,
        logging_dir="./logs/" + model_id,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=10,
        load_best_model_at_end=True,
        no_cuda=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )

    trainer.train()
    trainer.save_model("./data/models/" + model_id)
    tokenizer.save_pretrained("./data/models/" + model_id)
    test_results = trainer.evaluate(test_dataset)

    logging.info("Test results: " + str(test_results))

    with open("./data/models/" + model_id + "/config.json", "r") as config_file:
        config = json.load(config_file)
        config.update({"id2label": id2labelname})
    with open("./data/models/" + model_id + "/config.json", "w") as newconfig:
        json.dump(config, newconfig, indent=4)


def compute_metrics(pred):
    """
    Computes the accuracy metrics based on the predicted and true labels.

    Args:
        pred: The prediction object containing label_ids and predictions.

    Returns:
        dictionary: A dictionary containing the accuracy value under the key "accuracy".
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def test_model(input_text, model_id, label_names):
    """
    Tests a model using the input text, model ID, and label names.

    Args:
        input_text (str): The text to be classified.
        model_id (str): The ID of the model to be used for classification.
        label_names (list): A list of label names for classification.

    Returns:
        The classification result for the input text.
    """
    input_text = input_text.translate(str.maketrans("", "", string.punctuation)).lower()
    logging.info("Model name: " + model_id)
    pipe = pipeline(
        "text-classification",
        model="./data/models/" + model_id,
        device="cpu",
        padding=True,
        truncation=True,
        top_k=None,
    )
    data = pipe(input_text)
    return data[0]


def test_classifier_model(input_text):
    """
    Tests a classifier model using the provided input text.
    
    Args:
        input_text (str): The text to be classified.
        
    Returns:
        The classification result for the input text.
    """
    try:
        with open("./data/training_data.json") as file:
            intents_data = json.load(file)

        label_names = list(intents_data.keys())

        model_id = "bert-base-uncased"

        result = test_model(input_text, model_id, label_names)
        return result

    except Exception as err:
        logging.expection("Testing Failed")
        return str(err)


def train_classifier_model():
    """
    Trains a classifier model using the provided training data and saves the trained model and tokenizer.
    
    Returns:
        dict: A dictionary containing the status of the model training. The key is the model ID and the value is either "Trained" or "Failed".
    """
    try:
        logging.info("Entering Model Training")
        with open("data/training_data.json") as file:
            intents_data = json.load(file)

        id2labelname = {id: name for id, name in enumerate(intents_data.keys())}
        labelname2id = {name: id for id, name in enumerate(intents_data.keys())}

        training_data = prepare_training_data(intents_data)
        train_texts, train_labels = preprocess_texts_and_labels(
            training_data, labelname2id
        )

        model_id = "bert-base-uncased"

        test_texts = [
            "i have told you so many times not to call me on this matter just get lost"
        ]
        test_labels = [4]

        train_texts, dev_texts, train_labels, dev_labels = train_test_split(
            train_texts, train_labels, test_size=0.1, shuffle=True, random_state=1
        )

        train_model(
            intents_data,
            model_id,
            train_texts,
            dev_texts,
            train_labels,
            dev_labels,
            test_texts,
            test_labels,
            id2labelname,
        )
        logging.info("Model Trained Successfully")

        return {model_id:"Trained"}

    except Exception as err:
        logging.exception(err)
        return {model_id: "Failed"}