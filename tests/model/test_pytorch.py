import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader, Dataset


class SimpleNet(nn.Module):
    """A simple feedforward neural network with two hidden layers."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 2):
        """
        Initialize the network architecture.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layers
            output_size: Number of output classes
        """
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class FakeDataset(Dataset):
    """Generate fake data for training."""
    
    def __init__(self, num_samples: int = 1000, input_size: int = 10):
        """
        Create fake dataset.
        
        Args:
            num_samples: Number of samples to generate
            input_size: Dimension of input features
        """
        self.X = torch.randn(num_samples, input_size)
        # Create two classes with a simple rule: sum of features > 0
        self.y = (torch.sum(self.X, dim=1) > 0).long()
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

def train_model():
    """Train the neural network using fake data."""
    # Check if MPS is available
    if not torch.backends.mps.is_available():
        logger.warning("MPS not available. Using CPU instead.")
        device = torch.device("cpu")
    else:
        device = torch.device("mps")
    
    # Hyperparameters
    INPUT_SIZE = 10
    HIDDEN_SIZE = 20
    OUTPUT_SIZE = 2
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 0.001
    
    # Create model and move to device
    model = SimpleNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
    
    # Create fake dataset
    dataset = FakeDataset(num_samples=1000, input_size=INPUT_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in dataloader:
            # Move data to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            total_loss += loss.item()
        
        # Print epoch statistics
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} Accuracy: {accuracy:.2f}%")
    
    logger.info("Training completed!")
    return model

if __name__ == "__main__":
    model = train_model()