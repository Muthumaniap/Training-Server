import json
import torch
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sklearn.model_selection import train_test_split
import logging.config

logging = logging.getLogger()

# Function to load data from a JSON file
def load_data_from_json(file_path):
    try:
        logging.info("Loading data from JSON file: %s", file_path)
        with open(r"C:\Mukesh\White Mastery\Training-Server\bin\training_data.json") as file:
            data = json.load(file)
        return data
    except Exception as e:
        logging.error("Error loading data from JSON: %s", e)
        return None

# Function to preprocess the loaded data
def preprocess_data(data):
    try:
        logging.info("Preprocessing data")
        texts = []
        labels = []
        # Iterate through the dictionary items (label, sentences)
        for label, sentences in data.items():
            for sentence in sentences:
                texts.append(sentence)
                labels.append(label)
        return texts, labels
    except Exception as e:
        logging.error("Error preprocessing data: %s", e)
        return None, None
    
# Function to load and preprocess data from JSON file
def load_and_preprocess_data(file_path):
    try:
        logging.info("Loading and preprocessing data from JSON file: %s", file_path)
        # Load data from JSON file
        data = load_data_from_json(file_path)
        if data is None:
            return None, None, None
        # Preprocess loaded data
        texts, labels = preprocess_data(data)
        if texts is None or labels is None:
            return None, None, None

        # Initialize BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # Tokenize input texts
        tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        # Create a label mapping dictionary
        label_map = {label: i for i, label in enumerate(set(labels))}
        # Convert labels to label IDs
        label_ids = [label_map[label] for label in labels]
        # Convert label IDs to tensor
        label_tensor = torch.tensor(label_ids)

        # Create a PyTorch dataset
        dataset = TensorDataset(tokenized_texts["input_ids"], tokenized_texts["attention_mask"], label_tensor)
        # Split dataset into training and testing sets
        train_data, test_data = train_test_split(dataset, test_size=0.1, random_state=42)
        
        return train_data, test_data, label_map
    except Exception as e:
        logging.error("Error loading and preprocessing data: %s", e)
        return None, None, None

# Function to train the BERT model
def train_model(train_data):
    try:
        logging.info("Training model")
        # Initialize the BERT model for sequence classification
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=30)

        # Set device (CPU in this case)
        device = torch.device("cpu")
        model.to(device)

        # Initialize AdamW optimizer
        optimizer = optim.AdamW(model.parameters(), lr=2e-3)

        # Create a data loader for training data
        train_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=32)

        # Set model to training mode
        model.train()
        # Train the model
        for epoch in range(4):
            for batch in train_loader:
                batch = tuple(t.to(device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
                optimizer.zero_grad()
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

        return model
    
    except Exception as e:
        logging.error("Error training model: %s", e)
        return None




# Main function
def main():
    try:
        logging.info("Starting program execution")
        # Load and preprocess data
        train_data, test_data, label_map = load_and_preprocess_data("./training_data.json")
        if train_data is None or test_data is None or label_map is None:
            return
        # Train the model
        model = train_model(train_data)
        logging.info("Model training completed successfully")
        if model is not None:
            return {"Classifier":"Trained"}
        else:
            return {"Classifier":"Failed"}
    except Exception as e:
        logging.error("An error occurred: %s", e)

def test_classifier(input_text: str):
    """
    Function to test a classifier model on a single input text.

    Parameters:
    input_text (str): The input text to be classified.

    Returns:
    dict: A dictionary containing the prediction results.
    """
    try:
        # Initialize BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Tokenize input text
        tokenized_text = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")

        # Load pre-trained BERT model
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=30)

        # Make prediction
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            outputs = model(**tokenized_text)

        return {"Prediction":outputs}

    except Exception as err:
        logging.expection(err)
        return str(err)

# Entry point of the program
if __name__ == "__main__":
    main()
