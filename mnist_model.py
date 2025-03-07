import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# ======= Ustawienia urządzenia =======
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Model pracuje na urządzeniu: {device}")

# ======= Transformacje danych =======
transform_data = transforms.Compose([
    transforms.RandomRotation(20),  # Rotacja o max 20 stopni
    transforms.RandomAffine(0, translate=(0.2, 0.2)),  # Przesunięcia do 20% obrazu
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),  # Perspektywiczne zniekształcenie
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ======= Pobranie i załadowanie danych =======
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform_data, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform_data, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# ======= Definicja modelu =======
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten() # spłaszczenie obrazu
        self.fc1 = nn.Linear(784, 256) #28x28 = 784
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 10) # warstwa wyjściowa (10 klas)
    # nadpisanie metody forward
    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        #x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

# ======= Funkcja treningowa =======
def train_model(model, num_epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoka [{epoch+1}/{num_epochs}], Strata: {total_loss/len(train_loader):.4f}")

# ======= Funkcja ewaluacyjna =======
def evaluate_model(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Dokładność na zbiorze testowym: {accuracy:.2f}%")
model = NeuralNetwork()
train_model(model, 50)
evaluate_model(model)
# zapisanie przetrenowanego modelu do pliku
torch.save(model.state_dict(), "better_model.pth")