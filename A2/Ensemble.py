import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader


class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.model(x)


class DeepEnsemble:
    def __init__(
        self,
        train_dataset,
        input_size=2,
        hidden_size=2,
        output_size=2,
        lr=0.01,
        weight_decay=0.001,
        epochs=10,
        num_models=10,
    ):
        self.train_dataset = train_dataset
        self.epochs = epochs
        self.num_models = num_models
        self.models = []
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.weight_decay = weight_decay

    def train_model(self, seed):
        # random seed
        torch.manual_seed(seed)
        model = NN(self.input_size, self.hidden_size, self.output_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        indices = torch.randint(0, len(self.train_dataset), (len(self.train_dataset),))
        bootstrap_dataset = torch.utils.data.Subset(self.train_dataset, indices)
        train_loader = DataLoader(bootstrap_dataset, batch_size=32, shuffle=True)
        for _ in range(self.epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return model

    def train_ensemble(self):
        self.models = [self.train_model(seed=i) for i in range(self.num_models)]

    def predict(self, X):
        with torch.no_grad():
            models_outputs = torch.stack(
                [model(X) for model in self.models]
            )  # Raw logits
            mean_output = torch.mean(models_outputs, dim=0)  # Aggregate logits
            variance_output = torch.var(models_outputs, dim=0)
        return torch.softmax(mean_output, dim=1), variance_output

    def evaluate(self, X, y):
        mean_preds, uncertainty = self.predict(X)
        predicted_labels = mean_preds.argmax(dim=1)
        accuracy = (predicted_labels == y).float().mean().item()
        return accuracy, uncertainty
