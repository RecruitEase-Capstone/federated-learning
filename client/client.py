import socket
import torch
import os
from cnn_model import CNN1D
from client.data_loader import load_data_from_json, ECGDataset
from client.trainer import train_local_model
from client.client_utils import save_model_to_pickle, load_model_from_pickle, send_data, receive_data, apply_mask_to_model

class FederatedLearningClient:
    def __init__(self, client_id=0, host='10.34.100.121', port=65433):
        self.client_id = client_id
        self.host = host
        self.port = port
        self.local_model_path = f"./models/local/local_model_{client_id}.pickle"
        self.global_model_path = f"./models/global/global_model_{client_id}.pickle"
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.local_model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.global_model_path), exist_ok=True)
        
        # Data paths
        self.json_data_path = f"./data/train_{client_id}.json"
        self.class_mapper_path = "./data/class-mapper.json"
        self.base_dir = ""
    
    def load_and_train_model(self):
        """Load data and train the local model"""
        try:
            # Load data
            data_paths, labels, metadata, label_names, label_to_idx, idx_to_label = load_data_from_json(
                self.json_data_path, self.class_mapper_path, self.base_dir
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
            return None, None
    
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
                except Exception as e:
                    print(f"Error loading global model: {e}")
                
                print("Global model received and saved.")
                return True
                
        except Exception as e:
            print(f"Error communicating with server: {e}")
            return False
    
    def run(self):
        """Run the client process"""
        try:
            # Load data and train model
            model_pickle_file, trained_model = self.load_and_train_model()
            
            if model_pickle_file:
                # Communicate with server
                success = self.communicate_with_server(model_pickle_file)
                if success:
                    print("Client successfully completed federated learning round.")
                else:
                    print("Client failed to complete federated learning round.")
            else:
                print("Client failed to train model, cannot communicate with server.")
                
        except Exception as e:
            print(f"Error during client operation: {e}")
        
        print("Client finished.")

def main():
    client = FederatedLearningClient(host='127.0.0.1')
    client.run()

if __name__ == '__main__':
    main()