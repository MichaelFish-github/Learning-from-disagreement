import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import DebertaV2Tokenizer, DebertaV2Model
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from matplotlib import pyplot as plt
import scipy.stats


# Define the distribution prediction model
class DistributionPredictor(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super(DistributionPredictor, self).__init__()
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.log_softmax(x)
        return x


# Function to get embeddings in batches with tqdm progress bar
def get_embeddings_in_batches(texts, batch_size=32, device='cpu'):
    tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')
    model = DebertaV2Model.from_pretrained('microsoft/deberta-v3-base').to(device)
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # Get CLS token embeddings
        all_embeddings.append(embeddings)

    # Concatenate all embeddings into one tensor
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings


def calculate_metrics(outputs, ground_truth, threshold=0.4):
    # Apply exponentiation to convert log probabilities to normal probabilities
    outputs = torch.exp(outputs)

    # Convert ground truth to numpy arrays
    ground_truth = ground_truth.cpu().numpy()
    outputs = outputs.cpu().detach().numpy()

    # Predicted classes (argmax) for normal accuracy and F1 score
    predicted_classes = np.argmax(outputs, axis=1)
    true_classes = np.argmax(ground_truth, axis=1)

    accuracy = accuracy_score(true_classes, predicted_classes)

    # F1 Score (macro, so we average F1 across classes)
    f1 = f1_score(true_classes, predicted_classes, average='macro')

    # Class Threshold F1 (CT F1) with threshold of 0.4
    # For each class, if the predicted probability is greater than the threshold, we consider it as '1'
    predicted_thresholded = (outputs > threshold).astype(int)
    true_thresholded = (ground_truth > threshold).astype(int)

    ct_f1_per_class = []
    for class_idx in range(ground_truth.shape[1]):
        ct_f1_class = f1_score(true_thresholded[:, class_idx], predicted_thresholded[:, class_idx])
        ct_f1_per_class.append(ct_f1_class)

    # Average CT F1 across all classes
    ct_f1 = np.mean(ct_f1_per_class)

    # Cross-Entropy Loss (CE)
    # Now defining `outputs_tensor` correctly from `outputs`
    outputs_tensor = torch.tensor(outputs).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    ground_truth_tensor = torch.tensor(ground_truth).to(outputs_tensor.device)

    # Make sure to apply the correct shape when calculating cross-entropy
    ce_loss = F.cross_entropy(outputs_tensor, ground_truth_tensor.argmax(dim=1)).item()

    # Jensen-Shannon Divergence (JSD)
    def js_divergence(p, q):
        # Normalize both distributions to ensure they sum to 1
        p = p / np.sum(p, axis=1, keepdims=True)
        q = q / np.sum(q, axis=1, keepdims=True)

        m = 0.5 * (p + q)
        jsd = 0.5 * (scipy.stats.entropy(p.T, m.T) + scipy.stats.entropy(q.T, m.T))
        return np.mean(jsd)

    jsd = js_divergence(outputs, ground_truth)

    return accuracy, f1, ct_f1, ce_loss, jsd


def list_of_strings_to_tensor(lst, device='cpu'):
    lst = [list(map(float, s.strip('[]').split(', '))) for s in lst]
    return torch.tensor(lst).to(device)


def main():
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    df = pd.read_csv('./datasets/measuring-hate-speech/measuring-hate-speech-crowdtruth-vectors.csv')
    num_epochs = 75

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Get embeddings for train and test sets
    print("Embedding test data...")
    test_embeddings = get_embeddings_in_batches(test_df['text'].tolist(), batch_size=32, device=device)
    print("Finished embedding test data")

    print("Embedding train data...")
    train_embeddings = get_embeddings_in_batches(train_df['text'].tolist(), batch_size=32, device=device)
    print("Finished embedding train data")

    # Prepare the annotation distribution data
    train_annotation_distributions = list_of_strings_to_tensor(train_df['label'].tolist(), device=device)
    test_annotation_distributions = list_of_strings_to_tensor(test_df['label'].tolist(), device=device)
    output_dim = train_annotation_distributions[0].size()[0] if torch.is_tensor(
        train_annotation_distributions[0]) else 1
    input_dim = train_embeddings.size()[1]

    # Initialize the predictor model
    predictor = DistributionPredictor(embedding_dim=input_dim, output_dim=output_dim).to(device)

    # Training parameters
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)

    metrics = {
        "Accuracy": [],
        "F1 Score": [],
        "CT F1 Score": [],
        "Cross-Entropy Loss": [],
        "JSD": [],
    }

    # Training loop
    for epoch in range(num_epochs):
        predictor.train()
        optimizer.zero_grad()
        outputs = predictor(train_embeddings)
        loss = criterion(outputs, train_annotation_distributions)
        loss.backward()
        optimizer.step()

        # Calculate metrics
        accuracy, f1, ct_f1, ce_loss, jsd = calculate_metrics(outputs, train_annotation_distributions)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Accuracy: {accuracy}, F1: {f1}, CT F1: {ct_f1}, CE: {ce_loss}, JSD: {jsd}')

        # Store metrics
        metrics["Accuracy"].append(accuracy)
        metrics["F1 Score"].append(f1)
        metrics["CT F1 Score"].append(ct_f1)
        metrics["Cross-Entropy Loss"].append(ce_loss)
        metrics["JSD"].append(jsd)

    # Validate the model
    predictor.eval()
    with torch.no_grad():
        test_outputs = predictor(test_embeddings)
        test_loss = criterion(test_outputs, test_annotation_distributions)
        test_accuracy, test_f1, test_ct_f1, test_ce_loss, test_jsd = calculate_metrics(test_outputs,
                                                                                       test_annotation_distributions)
        print(
            f'Test Loss: {test_loss.item()}, Test Accuracy: {test_accuracy}, Test F1: {test_f1}, Test CT F1: {test_ct_f1}, Test CE: {test_ce_loss}, Test JSD: {test_jsd}')

    # Plotting the metrics
    epochs = list(range(1, num_epochs + 1))

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (metric, values) in zip(axes, metrics.items()):
        ax.plot(epochs, values, label=metric, marker='o')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} over Epochs')
        ax.legend()

    plt.tight_layout()
    plt.show()

    # Save the model
    torch.save(predictor.state_dict(), 'deberta_distribution_predictor.pth')
    # Save the plot
    fig.savefig('deberta_metrics_plot.png')


if __name__ == '__main__':
    main()
