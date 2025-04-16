import socket
import torch
import os
import numpy as np
from cnn_model import CNN1D
from client.data_loader import load_data_from_json, ECGDataset
from client.trainer import train_local_model
from client.client_utils import save_model_to_pickle, load_model_from_pickle, send_data, receive_data, apply_mask_to_model

class FederatedLearningClient:
    def __init__(self, client_id=3, host='localhost', port=65433, val_split=0.2):
        self.client_id = client_id
        self.host = host
        self.port = port
        self.val_split = val_split  # Validation split ratio
        self.local_model_path = f"./models/local/local_model_{client_id}.pickle"
        self.global_model_path = f"./models/global/global_model_{client_id}.pickle"
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.local_model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.global_model_path), exist_ok=True)
        
        # Data paths
        self.json_data_path = f"./data/train{client_id}.json"
        self.class_mapper_path = "./data/class-mapper.json"
        self.base_dir = ""
        
        # For storing dataset information
        self.label_to_idx = None
        self.idx_to_label = None
        self.label_names = None
        
        # Store validation dataset for later use
        self.val_dataset = None
        self.val_loader = None
    
    def load_and_train_model(self):
        """Load data, split into train/val, and train the local model"""
        try:
            # Load data
            data_paths, labels, metadata, label_names, label_to_idx, idx_to_label = load_data_from_json(
                self.json_data_path, self.class_mapper_path, self.base_dir
            )
            
            # Store for later use in evaluation
            self.label_to_idx = label_to_idx
            self.idx_to_label = idx_to_label
            self.label_names = label_names
            
            total_samples = len(data_paths)
            print(f"Data loaded: {total_samples} samples")
            print(f"Detected classes: {label_names}")
            
            # Create full dataset
            full_dataset = ECGDataset(data_paths, labels)
            
            # Split into train and validation
            val_size = int(total_samples * self.val_split)
            train_size = total_samples - val_size
            
            # Set seed for reproducibility
            torch.manual_seed(42)
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
            
            # Store validation dataset for later evaluation
            self.val_dataset = val_dataset
            
            print(f"Split data into {train_size} training samples and {val_size} validation samples")
            
            # Create dataloaders
            batch_size = min(32, len(train_dataset))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize model with the appropriate number of classes
            num_classes = len(label_to_idx)
            local_model = CNN1D(num_classes=num_classes)
            print(f"Model will classify {num_classes} classes: {label_names}")
            
            # Train the model locally
            trained_model = train_local_model(local_model, train_loader, epochs=1)
            print("\n📌 Model parameters BEFORE masking:")
            for name, param in trained_model.state_dict().items():
                if param.dtype == torch.float32:
                    print(f"[{name}] mean: {param.mean().item():.6f}")

            # Apply mask to model
            masked_model = apply_mask_to_model(trained_model, self.client_id)

            print("\n📌 Model parameters AFTER masking:")
            for name, param in masked_model.state_dict().items():
                if param.dtype == torch.float32:
                    print(f"[{name}] mean: {param.mean().item():.6f}")

            
            # Save model to pickle format
            model_pickle_file = save_model_to_pickle(masked_model, self.local_model_path)
            
            return model_pickle_file, trained_model
            
        except Exception as e:
            print(f"Error loading data or training model: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def train_model_with_validation(self, model, train_loader, val_loader, epochs=3, lr=0.001):
        """Train model with validation after each epoch"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on {device}")
        
        model = model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        best_val_accuracy = 0.0
        best_model_state = None
        
        # Debug the first batch to see the data shape
        for inputs, labels in train_loader:
            print(f"Debug - Original input shape: {inputs.shape}")
            break
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                # Reshape the input to have 1 channel - this is the key fix!
                if inputs.shape[1] != 1:
                    if len(inputs.shape) == 3:  # [batch_size, channels, seq_len]
                        # Option 1: Sum across channels (if they represent different leads)
                        inputs = inputs.sum(dim=1, keepdim=True)
                        # Option 2: Take first channel only
                        # inputs = inputs[:, 0:1, :]
                        # Option 3: Average across channels
                        # inputs = inputs.mean(dim=1, keepdim=True)
                    elif len(inputs.shape) == 2:  # [batch_size, seq_len]
                        inputs = inputs.unsqueeze(1)  # Add channel dimension
                
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_accuracy = 100 * train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    # Apply the same reshaping for validation data
                    if inputs.shape[1] != 1:
                        if len(inputs.shape) == 3:  # [batch_size, channels, seq_len]
                            # Option 1: Sum across channels (if they represent different leads)
                            inputs = inputs.sum(dim=1, keepdim=True)
                        elif len(inputs.shape) == 2:  # [batch_size, seq_len]
                            inputs = inputs.unsqueeze(1)  # Add channel dimension
                    
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_accuracy = 100 * val_correct / val_total
            
            print(f'Epoch {epoch+1}/{epochs}, '
                  f'Train Loss: {running_loss/len(train_loader):.4f}, '
                  f'Train Acc: {train_accuracy:.2f}%, '
                  f'Val Loss: {val_loss/len(val_loader):.4f}, '
                  f'Val Acc: {val_accuracy:.2f}%')
            
            # Save best model based on validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = model.state_dict().copy()
                print(f"New best model saved with validation accuracy: {best_val_accuracy:.2f}%")
        
        # Load best model state
        if best_model_state:
            model.load_state_dict(best_model_state)
            print(f"Loaded best model with validation accuracy: {best_val_accuracy:.2f}%")
        
        return model
    
    def communicate_with_server(self, model_pickle_file):
        """Connect to server and exchange models"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.host, self.port))
                print(f"Connected to server at {self.host}:{self.port}")

                # Read the pickle file and send it
                with open(model_pickle_file, 'rb') as f:
                    model_data = f.read()
                
                # Send model
                print(f"Sending model (size: {len(model_data)} bytes)")
                send_data(s, model_data)

                # Receive global model
                print("Waiting for global model...")
                received_data = receive_data(s)
                print(f"Received global model (size: {len(received_data)} bytes)")
                
                # Save received model to file
                with open(self.global_model_path, 'wb') as f:
                    f.write(received_data)
                
                # Load the model for verification
                try:
                    global_model = load_model_from_pickle(self.global_model_path)
                    print("Global model successfully loaded.")
                    return global_model
                except Exception as e:
                    print(f"Error loading global model: {e}")
                    return None
                
        except Exception as e:
            print(f"Error communicating with server: {e}")
            return None
    
    def evaluate_model(self, model, is_global=False):
        """Evaluate model accuracy on validation dataset (instead of test dataset)"""
        model_name = "Global" if is_global else "Local"
        try:
            # Use the validation dataset that was already created
            if self.val_loader is None:
                print(f"Error: Validation dataset not available for evaluation")
                return None
            
            print(f"Evaluating {model_name} model on validation data")
            
            # Put model in evaluation mode
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
            
            correct = 0
            total = 0
            
            # Ensure we have proper label mapping
            if self.label_names is None or self.label_to_idx is None:
                print("Error: Label mapping not available")
                return None
                
            # Confusion matrix setup
            num_classes = len(self.label_to_idx)
            confusion_matrix = torch.zeros(num_classes, num_classes)
            
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    # Apply the same reshaping for test data
                    if inputs.shape[1] != 1:
                        if len(inputs.shape) == 3:  # [batch_size, channels, seq_len]
                            # Option 1: Sum across channels (if they represent different leads)
                            inputs = inputs.sum(dim=1, keepdim=True)
                        elif len(inputs.shape) == 2:  # [batch_size, seq_len]
                            inputs = inputs.unsqueeze(1)  # Add channel dimension
                    
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Update confusion matrix
                    for t, p in zip(labels.view(-1), predicted.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
            
            accuracy = 100 * correct / total
            print(f"{model_name} Model Accuracy: {accuracy:.2f}%")
            
            # Print per-class accuracy
            print("\nPer-class accuracy:")
            for i in range(num_classes):
                class_correct = confusion_matrix[i, i].item()
                class_total = confusion_matrix[i, :].sum().item()
                class_name = self.label_names[i] if i < len(self.label_names) else f"Unknown-{i}"
                if class_total > 0:
                    class_accuracy = 100 * class_correct / class_total
                    print(f"    {class_name}: {class_accuracy:.2f}%")
                else:
                    print(f"    {class_name}: No samples")
            
            # Save results to file
            with open(f"./results_{model_name.lower()}_model_client_{self.client_id}.txt", "w") as f:
                f.write(f"{model_name} Model Evaluation Results\n")
                f.write(f"Overall Accuracy: {accuracy:.2f}%\n\n")
                f.write("Per-class accuracy:\n")
                for i in range(num_classes):
                    class_correct = confusion_matrix[i, i].item()
                    class_total = confusion_matrix[i, :].sum().item()
                    class_name = self.label_names[i] if i < len(self.label_names) else f"Unknown-{i}"
                    if class_total > 0:
                        class_accuracy = 100 * class_correct / class_total
                        f.write(f"    {class_name}: {class_accuracy:.2f}%\n")
                    else:
                        f.write(f"    {class_name}: No samples\n")
                
                f.write("\nConfusion Matrix:\n")
                f.write("Rows: True labels, Columns: Predicted labels\n")
                for i in range(num_classes):
                    class_name = self.label_names[i] if i < len(self.label_names) else f"Unknown-{i}"
                    f.write(f"{class_name}: {confusion_matrix[i].tolist()}\n")
            
            return accuracy
            
        except Exception as e:
            print(f"Error evaluating {model_name} model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run(self):
        """Run the client process"""
        try:
            # Load data and train model
            model_pickle_file, local_model = self.load_and_train_model()
            
            if model_pickle_file and local_model:
                # Evaluate local model
                print("\nEvaluating local model performance...")
                local_accuracy = self.evaluate_model(local_model, is_global=False)
                
                # Communicate with server
                global_model = self.communicate_with_server(model_pickle_file)
                
                if global_model:
                    print("\nEvaluating global model performance...")
                    global_accuracy = self.evaluate_model(global_model, is_global=True)
                    
                    # Compare performances
                    if local_accuracy is not None and global_accuracy is not None:
                        print(f"\nAccuracy comparison:")
                        print(f"Local model: {local_accuracy:.2f}%")
                        print(f"Global model: {global_accuracy:.2f}%")
                        
                        improvement = global_accuracy - local_accuracy
                        if improvement > 0:
                            print(f"The global model performs better by {improvement:.2f}%")
                        elif improvement < 0:
                            print(f"The local model performs better by {abs(improvement):.2f}%")
                        else:
                            print("Both models perform equally")
                            
                    print("Client successfully completed federated learning round.")
                else:
                    print("Client failed to receive global model for evaluation.")
            else:
                print("Client failed to train model, cannot communicate with server.")
                
        except Exception as e:
            print(f"Error during client operation: {e}")
            import traceback
            traceback.print_exc()
        
        print("Client finished.")

def main():
    client = FederatedLearningClient(host='127.0.0.1')
    client.run()

if __name__ == '__main__':
    main()