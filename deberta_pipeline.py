import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DebertaV2Tokenizer, DebertaV2Model
from matplotlib import pyplot as plt
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import numpy as np

import os


# Define the distribution prediction model
class DistributionPredictor(nn.Module):
    def __init__(self, embedding_dim, output_dim, hidden_dim=256, dropout_prob=0.3):
        super(DistributionPredictor, self).__init__()
        # First hidden layer
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.relu1 = nn.ReLU()

        # Second hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.relu2 = nn.ReLU()

        # Output layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.log_softmax(x)

        return x


# Function to get embeddings for a single batch
def get_embeddings_for_batch(texts, tokenizer, model, device='cpu'):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # Get CLS token embeddings
    return embeddings


def calculate_metrics(outputs, ground_truth, threshold=0.4):
    # Apply softmax to get probabilities
    outputs = torch.exp(outputs)

    # Move tensors to CPU and convert to numpy arrays
    ground_truth = ground_truth.cpu().numpy()
    outputs = outputs.cpu().detach().numpy()

    # Calculate predicted and true class indices
    predicted_classes = np.argmax(outputs, axis=1)
    true_classes = np.argmax(ground_truth, axis=1)

    # Calculate accuracy and macro F1 score
    accuracy = accuracy_score(true_classes, predicted_classes)
    f1 = f1_score(true_classes, predicted_classes, average='macro')

    # Threshold the outputs and ground truth to create binary labels for each class
    predicted_thresholded = (outputs > threshold).astype(int)
    true_thresholded = (ground_truth > threshold).astype(int)

    # Compute Class Threshold F1 score (CT F1) per class
    ct_f1_per_class = []
    for class_idx in range(ground_truth.shape[1]):
        ct_f1_class = f1_score(true_thresholded[:, class_idx], predicted_thresholded[:, class_idx])
        ct_f1_per_class.append(ct_f1_class)

    # Compute average CT F1 across all classes
    ct_f1 = np.mean(ct_f1_per_class)

    # Convert numpy arrays back to tensors for CE loss calculation
    outputs_tensor = torch.tensor(outputs).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    ground_truth_tensor = torch.tensor(ground_truth).to(outputs_tensor.device)

    # Compute Cross-Entropy Loss (CE)
    ce_loss = F.cross_entropy(outputs_tensor, ground_truth_tensor.argmax(dim=1)).item()

    # Compute Jensen-Shannon Divergence (JSD) using scipy's jensenshannon function
    jsd_per_sample = [
        jensenshannon(outputs[i], ground_truth[i]) ** 2 for i in range(outputs.shape[0])
    ]
    jsd = np.mean(jsd_per_sample)

    return accuracy, f1, ct_f1, ce_loss, jsd


def get_and_save_embeddings_for_batch(texts, tokenizer, model, batch_idx, dataset_type, save_path, device='cpu'):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # Get CLS token embeddings

    # Save embeddings to disk
    os.makedirs(save_path, exist_ok=True)
    torch.save(embeddings, os.path.join(save_path, f"{dataset_type}_embeddings_batch_{batch_idx}.pt"))
    return embeddings


def load_embeddings_from_disk(batch_idx, dataset_type, save_path, device='cpu'):
    # Load embeddings from disk
    return torch.load(os.path.join(save_path, f"{dataset_type}_embeddings_batch_{batch_idx}.pt")).to(device)


def delete_saved_embeddings(save_path):
    if os.path.exists(save_path):
        for file_name in os.listdir(save_path):
            file_path = os.path.join(save_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(save_path)


def list_of_strings_to_tensor(lst, device='cpu'):
    lst = [list(map(float, s.strip('[]').split(', '))) for s in lst]
    return torch.tensor(lst).to(device)


def main():
    dataset_name = 'measuring-hate-speech'
    baseline_name = 'crowdtruth'

    results_path = f'./results/{dataset_name}/{baseline_name}'
    df = pd.read_csv('./datasets/measuring-hate-speech/measuring-hate-speech-crowdtruth-vectors.csv')

    # New flag: Set to True if you want to split one dataframe into train and test.
    use_split_data = True

    if use_split_data:
        # Split the single dataframe into train and test
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    else:
        # Load separate dataframes for train and test
        train_df = pd.read_csv('./datasets/measuring-hate-speech/train.csv')
        test_df = pd.read_csv('./datasets/measuring-hate-speech/test.csv')

    num_epochs = 75
    batch_size = 128

    # Filter out rows with all zeros in the label column
    train_df = train_df[train_df['label'] != '[0.0, 0.0, 0.0]']
    test_df = test_df[test_df['label'] != '[0.0, 0.0, 0.0]']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')
    model = DebertaV2Model.from_pretrained('microsoft/deberta-v3-base').to(device)

    train_annotation_distributions = list_of_strings_to_tensor(train_df['label'].tolist(), device=device)
    test_annotation_distributions = list_of_strings_to_tensor(test_df['label'].tolist(), device=device)

    output_dim = train_annotation_distributions[0].size()[0]
    input_dim = model.config.hidden_size

    predictor = DistributionPredictor(embedding_dim=input_dim, output_dim=output_dim).to(device)

    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)

    metrics = {
        "Accuracy": {"train": [], "test": []},
        "F1 Score": {"train": [], "test": []},
        "CT F1 Score": {"train": [], "test": []},
        "Cross-Entropy Loss": {"train": [], "test": []},
        "JSD": {"train": [], "test": []},
    }

    save_path = './saved_embeddings/'

    for epoch in range(num_epochs):
        predictor.train()
        epoch_loss = 0

        for i in tqdm(range(0, len(train_df), batch_size), desc=f"Epoch {epoch + 1}/{num_epochs}"):
            batch_texts = train_df['text'].iloc[i:i + batch_size].tolist()
            batch_labels = train_annotation_distributions[i:i + batch_size]

            if epoch == 0:
                # Embed and save in the first epoch
                train_embeddings = get_and_save_embeddings_for_batch(batch_texts, tokenizer, model, i // batch_size,
                                                                     'train', save_path, device=device)
            else:
                # Load from disk in subsequent epochs
                train_embeddings = load_embeddings_from_disk(i // batch_size, 'train', save_path, device=device)

            optimizer.zero_grad()
            outputs = predictor(train_embeddings)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_df):.4f}')

        # Save metrics for train set
        train_accuracy, train_f1, train_ct_f1, train_ce_loss, train_jsd = calculate_metrics(outputs, batch_labels)
        metrics["Accuracy"]["train"].append(train_accuracy)
        metrics["F1 Score"]["train"].append(train_f1)
        metrics["CT F1 Score"]["train"].append(train_ct_f1)
        metrics["Cross-Entropy Loss"]["train"].append(train_ce_loss)
        metrics["JSD"]["train"].append(train_jsd)

        # Evaluate on the test set
        predictor.eval()
        test_loss = 0
        all_test_outputs = []
        all_test_labels = []

        with torch.no_grad():
            for i in tqdm(range(0, len(test_df), batch_size), desc="Evaluating on test set"):
                batch_texts = test_df['text'].iloc[i:i + batch_size].tolist()
                batch_labels = test_annotation_distributions[i:i + batch_size]

                if epoch == 0:
                    test_embeddings = get_and_save_embeddings_for_batch(batch_texts, tokenizer, model, i // batch_size,
                                                                        'test', save_path, device=device)
                else:
                    test_embeddings = load_embeddings_from_disk(i // batch_size, 'test', save_path, device=device)

                outputs = predictor(test_embeddings)
                all_test_outputs.append(outputs)
                all_test_labels.append(batch_labels)
                loss = criterion(outputs, batch_labels)
                test_loss += loss.item()

        # Aggregate all test outputs and labels for metrics calculation
        all_test_outputs = torch.cat(all_test_outputs)
        all_test_labels = torch.cat(all_test_labels)
        test_accuracy, test_f1, test_ct_f1, test_ce_loss, test_jsd = calculate_metrics(all_test_outputs,
                                                                                       all_test_labels)

        metrics["Accuracy"]["test"].append(test_accuracy)
        metrics["F1 Score"]["test"].append(test_f1)
        metrics["CT F1 Score"]["test"].append(test_ct_f1)
        metrics["Cross-Entropy Loss"]["test"].append(test_ce_loss)
        metrics["JSD"]["test"].append(test_jsd)

        print(f'Test Loss: {test_loss / len(test_df):.4f}')

    delete_saved_embeddings(save_path)

    # Plotting metrics
    epochs = list(range(1, num_epochs + 1))
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (metric, values) in zip(axes, metrics.items()):
        ax.plot(epochs, values["train"], label=f'Train {metric}', marker='o', color='blue')
        ax.plot(epochs, values["test"], label=f'Test {metric}', marker='x', color='red')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} over Epochs')
        ax.legend()

    plt.tight_layout()
    plt.show()

    os.makedirs(results_path, exist_ok=True)
    torch.save(predictor.state_dict(), os.path.join(results_path, 'deberta_model.pth'))
    fig.savefig(os.path.join(results_path, 'metrics.png'))


if __name__ == '__main__':
    main()
