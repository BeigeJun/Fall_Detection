import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

class FallDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        self.labels = []
        
        label_map = {'fall': 1.0, 'normal': 0.0}
        
        for label_name, label_value in label_map.items():
            folder_path = os.path.join(root_dir, label_name)
                
            for file in os.listdir(folder_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(folder_path, file)
                    df = pd.read_csv(file_path)
                    
                    for _, group in df.groupby('seq_id'):
                        if len(group) == 30:
                            features = group.iloc[:, 2:].values 
                            self.data.append(features)
                            self.labels.append(label_value)
        
        self.data = torch.FloatTensor(np.array(self.data))
        self.labels = torch.FloatTensor(np.array(self.labels)).unsqueeze(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class FallLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_layers=2):
        super(FallLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x) 
        out = self.fc(hn[-1])
        return out


def main():
    ROOT_CSV_DIR = 'D:/3.Fall_Detection_DATA/2.Pos_CSV_DATA/processed_csv'
    BATCH_SIZE = 1
    EPOCHS = 100
    LR = 0.0005
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = FallDataset(ROOT_CSV_DIR)

    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=BATCH_SIZE)

    model = FallLSTM().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                
                predicted = (outputs > 0.5).float()
                correct += (predicted == targets).sum().item()

        if (epoch + 1) % 10 == 0:
            accuracy = correct / len(val_idx) * 100
            print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Acc: {accuracy:.2f}%")

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'C:/Pycharm_Program/Fall_Detection/models/fall_lstm_pytorch.pth')

if __name__ == "__main__":
    main()