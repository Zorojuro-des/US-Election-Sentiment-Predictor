import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error , r2_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+|[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens) if tokens else "empty"

class SentimentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class RegressionNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

def load_data(file, use_nlp=False, vectorizer=None, scaler=None, fit=True):
    df = pd.read_csv(file)
    df['sentiment'] = df['sentiment'].map({'negative': -1, 'neutral': 0, 'positive': 1})
    df.dropna(inplace=True)
    y = df['sentiment'].values

    if use_nlp:
        df['clean_tweet'] = df['tweet_text'].apply(clean_text)
        if fit:
            vectorizer = TfidfVectorizer(max_features=5000)
            X_text = vectorizer.fit_transform(df['clean_tweet']).toarray()
        else:
            X_text = vectorizer.transform(df['clean_tweet']).toarray()
    else:
        X_text = np.array([]).reshape(len(df), 0)

    X_num = df[['likes', 'retweets']].values
    if fit:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num)
    else:
        X_num = scaler.transform(X_num)

    X = np.hstack((X_num, X_text)).astype(np.float32)
    return X, y, vectorizer, scaler

def train_model(X_train, y_train, X_val, y_val, X_test, y_test, input_dim, epochs=30, lr=0.001):
    model = RegressionNN(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(SentimentDataset(X_train, y_train), batch_size=32, shuffle=True)

    train_losses = []
    val_losses = []
    val_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation metrics
        model.eval()
        with torch.no_grad():
            val_preds = model(torch.tensor(X_val, dtype=torch.float32)).numpy().flatten()
            val_loss = mean_squared_error(y_val, val_preds)
            val_losses.append(val_loss)

            val_accuracy = np.mean(np.abs(val_preds - y_val) <= 0.5) * 100
            val_accuracies.append(val_accuracy)

            # Test accuracy per epoch
            test_preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()
            test_accuracy = np.mean(np.abs(test_preds - y_test) <= 0.5) * 100
            test_accuracies.append(test_accuracy)

    return model, train_losses, val_accuracies, test_accuracies

def evaluate_model(model, X, y, threshold=0.5):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32)).numpy().flatten()

    preds = np.nan_to_num(preds)
    y = np.nan_to_num(y)

    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)

    accuracy = np.mean(np.abs(preds - y) <= threshold) * 100

    print(f"Mean Squared Error (RMSE): {mse:.4f}")
    print(f"R-squared (R²) Score: {r2:.4f}")
    print(f"Accuracy (±{threshold} tolerance): {accuracy:.2f}%")

    return mse,r2  


def run_nn(train_file, val_file, test_file, use_nlp=False):
    X_train, y_train, vectorizer, scaler = load_data(train_file, use_nlp, fit=True)
    X_val, y_val, _, _ = load_data(val_file, use_nlp, vectorizer, scaler, fit=False)
    X_test, y_test, _, _ = load_data(test_file, use_nlp, vectorizer, scaler, fit=False)

    model, train_losses, val_accuracies, test_accuracies = train_model(
        X_train, y_train, X_val, y_val, X_test, y_test, input_dim=X_train.shape[1]
    )

    print("Validation Set:")
    mse_val , r2_val = evaluate_model(model, X_val, y_val)
    print("Test Set:")
    mse_test, r2_test = evaluate_model(model, X_test, y_test)

    plot_training_curves(train_losses, val_accuracies, test_accuracies)

    return mse_test,r2_test,mse_val,r2_val

def plot_training_curves(train_losses, val_accuracies,test_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='green', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy per Epoch')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, test_accuracies, label='Test Accuracy', color='red', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy per Epoch')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

print("== Unprocessed with NLP ==")
print()
mse_np_test , r2_np_test , mse_np_valid , r2_np_valid = run_nn("train.csv", "val.csv", "test.csv", use_nlp=True)
print()


print("== Preprocessed with NLP ==")
print()
mse_p_test , r2_p_test , mse_p_valid , r2_p_valid = run_nn("train_preprocessed.csv", "val_preprocessed.csv", "test_preprocessed.csv", use_nlp=True)
print()


print("== Unprocessed without NLP ==")
print()
mse_np_test_linear , r2_np_test_linear , mse_np_valid_linear , r2_np_valid_linear = run_nn("train.csv", "val.csv", "test.csv", use_nlp=False)
print()


print("== Preprocessed without NLP ==")
print()
mse_p_test_linear , r2_p_test_linear , mse_p_valid_linear , r2_p_valid_linear = run_nn("train_preprocessed.csv", "val_preprocessed.csv", "test_preprocessed.csv", use_nlp=False)
print()


data = {
    "Model" : ["Unprocessed MSE" , "Processed MSE" , "Unprocessed R Square" , "Processed R Square"],
    "NLP Valid" : [mse_np_valid , mse_p_valid , r2_np_valid , r2_p_valid],
    "NLP Test" : [mse_np_test , mse_p_test , r2_np_test , r2_p_test],
    "Linear Valid" : [mse_np_valid_linear , mse_p_valid_linear , r2_np_valid_linear , r2_p_valid_linear],
    "Linear Test" : [mse_np_test_linear , mse_p_test_linear , r2_np_test_linear , r2_p_test_linear]
}

datadf = pd.DataFrame(data)
datadf.to_csv("NeuralNetworkRegressionErrors.csv",index=False)