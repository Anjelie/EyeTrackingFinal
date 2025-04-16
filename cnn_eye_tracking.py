import os
import pandas as pd
import numpy as np
import torch
import random 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)  # For PyTorch
np.random.seed(seed)  # For NumPy
random.seed(seed)  # For Python's random library
torch.cuda.manual_seed_all(seed)  # If you're using GPU
torch.backends.cudnn.deterministic = True  # Ensures deterministic algorithms on GPU
torch.backends.cudnn.benchmark = False  # Prevents non-deterministic behavior

# Configuration
BASE_PATH = 'data/data'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
BATCH_SIZE = 4

# Load subject labels from CSV
labels_df = pd.read_csv('dyslexia_class_label.csv')

# Convert to a dictionary with subject_id as key and class_id (0/1) as value
subject_labels = dict(zip(labels_df['subject_id'].astype(str), labels_df['class_id']))

# Optional: Print loaded labels to verify
print("Loaded subject labels:")
for sid, label in subject_labels.items():
    print(f"Subject {sid}: {'Dyslexic' if label == 1 else 'Non-dyslexic'}")

# Mapping of task names to their actual file prefixes
task_prefixes = {
    "Syllables": "T1",
    "Meaningful_Text": "T4",
    "Pseudo_Text": "T5"
}

# Update feature loading function
def load_subject_features(subject_id, task_name):
    try:
        task_prefix = task_prefixes[task_name]
        prefix = f"Subject_{subject_id}_{task_prefix}_{task_name}"
        fix = pd.read_csv(os.path.join(BASE_PATH, f"{prefix}_fixations.csv"))
        sac = pd.read_csv(os.path.join(BASE_PATH, f"{prefix}_saccades.csv"))
        met = pd.read_csv(os.path.join(BASE_PATH, f"{prefix}_metrics.csv"))

        fix_features = fix.select_dtypes(include=[np.number]).mean().fillna(0)
        sac_features = sac.select_dtypes(include=[np.number]).mean().fillna(0)
        met_features = met.select_dtypes(include=[np.number]).mean().fillna(0)

        # # Show all column names
        print("Fixation numeric columns:", fix.select_dtypes(include=[np.number]).columns.tolist())
        print("Saccade numeric columns:", sac.select_dtypes(include=[np.number]).columns.tolist())
        print("Metrics numeric columns:", met.select_dtypes(include=[np.number]).columns.tolist())

        # # Show the mean values being extracted as features
        # print("\nFixation feature means:\n", fix.select_dtypes(include=[np.number]).mean())
        # print("\nSaccade feature means:\n", sac.select_dtypes(include=[np.number]).mean())
        # print("\nMetrics feature means:\n", met.select_dtypes(include=[np.number]).mean())

        features = pd.concat([fix_features, sac_features, met_features])
        return features.values, subject_labels.get(subject_id, 0)
    except Exception as e:
        print(f"Error loading {task_name} for subject {subject_id}: {e}")
        return None, None

# Dataset class
class EyeTrackingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].reshape(1, 10, -1), self.y[idx]

# CNN Model
class EyeCNN(nn.Module):
    def __init__(self, input_shape):
        super(EyeCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        self.flatten_dim = self._get_flatten_dim(input_shape)
        self.fc1 = nn.Linear(self.flatten_dim, 64)
        self.fc2 = nn.Linear(64, 2)

    def _get_flatten_dim(self, shape):
        dummy = torch.zeros(1, 1, *shape)
        out = self.pool(self.conv2(self.pool(self.conv1(dummy))))
        return out.view(1, -1).shape[1]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Aggregate data
task_types = ["Syllables", "Meaningful_Text", "Pseudo_Text"]
features_all = []
labels_all = []

for subject_id in subject_labels:
    for task in task_types:
        features, label = load_subject_features(subject_id, task)
        if features is not None:
            features_all.append(features)
            labels_all.append(label)

X = np.array(features_all)
y = np.array(labels_all)

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Pad and reshape
num_features = X_scaled.shape[1]
height = 10
width = int(np.ceil(num_features / height))
padded = np.zeros((X_scaled.shape[0], height * width))
padded[:, :num_features] = X_scaled
X_scaled = padded.reshape(-1, height, width)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
train_dataset = EyeTrackingDataset(X_train, y_train)
test_dataset = EyeTrackingDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model
model = EyeCNN((height, width)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

# Evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
        
Test_Accuracy = (correct / total) * 100

print(f"\nTest Accuracy: {Test_Accuracy:.2f}%")

torch.save(model.state_dict(), "cnn_eye_tracking.pth") 


