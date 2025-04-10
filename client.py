import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import json
import pickle

# Konfigurasi Klien
HOST = '0.0.0.0'  # Loopback address
PORT = 65433  # Port server
PICKLE_PROTOCOL = 4

# Model CNN1D dari dokumen pertama
class CNN1D(nn.Module):
    def __init__(self, num_classes=8):
        super(CNN1D, self).__init__()
        
        # Layer 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 2
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 3
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # After 3 max pooling with stride 2, size 128 -> 64 -> 32 -> 16
        self.fc1 = nn.Linear(64 * 16, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Layer 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Layer 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Layer 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Kelas dataset untuk data ECG dari JSON
class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, labels):
        self.data_paths = data_paths
        self.labels = labels
        
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        # Load NPY data from path
        data_path = self.data_paths[idx]
        try:
            x = np.load(data_path)
            y = self.labels[idx]
            
            # Convert data to tensors and add channel dimension
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)
            
            return x, y
        except Exception as e:
            print(f"Error loading {data_path}: {e}")
            # Return a zero tensor of the expected shape as a fallback
            return torch.zeros(128, dtype=torch.float32), torch.tensor(0, dtype=torch.long)

# Fungsi untuk memuat data dari JSON
def load_data_from_json(json_file_path, class_mapper_path, base_dir=""):
    # Load class mapper
    with open(class_mapper_path, 'r') as f:
        class_mapper = json.load(f)
    
    # Convert class mapper to idx_to_label and label_to_idx
    idx_to_label = {v: k for k, v in class_mapper.items()}
    label_to_idx = class_mapper
    
    # Check if json_file_path is a list of JSON objects or a file path
    if isinstance(json_file_path, list):
        data_entries = json_file_path
    else:
        # Read JSON file containing data entries
        with open(json_file_path, 'r') as f:
            data_entries = json.load(f)
    
    all_data_paths = []
    all_labels = []
    all_metadata = []  # Store additional metadata
    
    print(f"Loading data from JSON...")
    
    for entry in data_entries:
        try:
            path = os.path.join(base_dir, entry["path"])
            label_str = entry["label"]
            
            # Convert string label to numeric index using class_mapper
            if label_str in label_to_idx:
                label_idx = label_to_idx[label_str]
                
                # Check if the file exists
                if os.path.exists(path):
                    all_data_paths.append(path)
                    all_labels.append(label_idx)
                    
                    # Store metadata for reference
                    all_metadata.append({
                        "name": entry.get("name", ""),
                        "lead": entry.get("lead", ""),
                        "label": label_str,
                        "filename": entry.get("filename", "")
                    })
                else:
                    print(f"Warning: File {path} does not exist, skipping.")
            else:
                print(f"Warning: Unknown label {label_str}, skipping.")
        except Exception as e:
            print(f"Error processing entry {entry}: {e}")
    
    # Get unique labels
    unique_labels = sorted(list(set(all_labels)))
    label_names = [idx_to_label[idx] for idx in unique_labels]
    
    print(f"Total loaded entries: {len(all_data_paths)}")
    print(f"Detected classes: {label_names}")
    
    return all_data_paths, np.array(all_labels), all_metadata, label_names, label_to_idx, idx_to_label

def send_data(sock, data):
    """Kirim data dengan format panjang + data"""
    # Kirim panjang data (4 byte)
    data_length = len(data)
    sock.sendall(data_length.to_bytes(4, byteorder='big'))
    
    # Kirim data
    sock.sendall(data)

def receive_data(sock):
    """Terima data dengan format panjang + data"""
    # Terima panjang data
    length_bytes = sock.recv(4)
    data_length = int.from_bytes(length_bytes, byteorder='big')
    
    # Terima data
    data = b''
    remaining = data_length
    while remaining > 0:
        chunk = sock.recv(min(4096, remaining))
        if not chunk:
            raise ConnectionError("Connection closed before receiving all data")
        data += chunk
        remaining -= len(chunk)
    
    return data

def train_local_model(model, train_loader, epochs=1, learning_rate=0.001):
    """Latih model lokal"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device).unsqueeze(1)  # Add channel dimension
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')

    return model

def save_model_to_pickle(model, filename="global_model.pickle"):
    """Save model in the expected format"""
    model.cpu()
    model_data = {
        'state_dict': model.state_dict(),
        'architecture': 'CNN1D',
        'num_classes': model.fc2.out_features
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f, protocol=PICKLE_PROTOCOL)
    
    print(f"Model saved to {filename}")
    return filename

def load_model_from_pickle(filename="model.pickle"):
    """Muat model dari format pickle"""
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    
    # Create model with the same architecture
    if model_data['architecture'] == 'CNN1D':
        model = CNN1D(num_classes=model_data['num_classes'])
        model.load_state_dict(model_data['state_dict'])
        print(f"Model loaded from {filename}")
        return model
    else:
        raise ValueError(f"Unknown architecture: {model_data['architecture']}")

def main():
    # Define paths
    json_data_path = "./data/train_1.json"  # Path to your JSON file
    class_mapper_path = "./data/class-mapper.json"  # Path to class mapper
    base_dir = ""  # Base directory if paths in JSON are relative
    
    print("Loading data from JSON...")
    try:
        # Load data
        data_paths, labels, metadata, label_names, label_to_idx, idx_to_label = load_data_from_json(
            json_data_path, class_mapper_path, base_dir
        )
        
        print(f"Data loaded: {len(data_paths)} samples")
        print(f"Detected classes: {label_names}")
        
        # Create dataset and dataloader
        dataset = ECGDataset(data_paths, labels)
        
        # Use a smaller batch size if dataset is small
        batch_size = min(32, len(dataset))
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model with the appropriate number of classes
        num_classes = len(label_to_idx)
        local_model = CNN1D(num_classes=num_classes)
        print(f"Model will classify {num_classes} classes: {label_names}")
        
        # Train the model locally
        trained_model = train_local_model(local_model, train_loader, epochs=1)
        
        # Save model to pickle format
        model_pickle_file = save_model_to_pickle(trained_model, "./models/local/local_model.pickle")
        
        # Koneksi ke Server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((HOST, PORT))
                print(f"Terhubung ke server di {HOST}:{PORT}")

                # Read the pickle file and send it
                with open(model_pickle_file, 'rb') as f:
                    model_data = f.read()
                
                # Kirim model
                print(f"Mengirim model (ukuran: {len(model_data)} bytes)")
                send_data(s, model_data)

                # Terima model global
                print("Menunggu model global...")
                received_data = receive_data(s)
                print(f"Menerima model global (ukuran: {len(received_data)} bytes)")
                
                # Save received model to file
                with open("./models/global/global_model.pickle", 'wb') as f:
                    f.write(received_data)
                
                # Load the model for verification
                try:
                    global_model = load_model_from_pickle("./models/global/global_model.pickle")
                    print("Model global berhasil dimuat.")
                except Exception as e:
                    print(f"Error saat memuat model global: {e}")
                
                print("Model global diterima dan disimpan.")

            except Exception as e:
                print(f"Error pada komunikasi dengan server: {e}")
    
    except Exception as e:
        print(f"Error saat memuat data: {e}")

    print("Klien selesai.")

if __name__ == '__main__':
    main()