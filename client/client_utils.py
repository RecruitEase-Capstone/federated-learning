import pickle
import os
import torch
from cnn_model import CNN1D

# Pickle protocol version
PICKLE_PROTOCOL = 4

def save_model_to_pickle(model, filename="global_model.pickle"):
    """Save model in the expected format"""
    model.cpu()
    model_data = {
        'state_dict': model.state_dict(),
        'architecture': 'CNN1D',
        'num_classes': model.fc2.out_features
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f, protocol=PICKLE_PROTOCOL)
    
    print(f"Model saved to {filename}")
    return filename

def load_model_from_pickle(filename="model.pickle"):
    """Load model from pickle format"""
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

def send_data(sock, data):
    """Send data with length + data format"""
    # Send data length (4 bytes)
    data_length = len(data)
    sock.sendall(data_length.to_bytes(4, byteorder='big'))
    
    # Send data
    sock.sendall(data)

def receive_data(sock):
    """Receive data with length + data format"""
    # Receive data length
    length_bytes = sock.recv(4)
    data_length = int.from_bytes(length_bytes, byteorder='big')
    
    # Receive data
    data = b''
    remaining = data_length
    while remaining > 0:
        chunk = sock.recv(min(4096, remaining))
        if not chunk:
            raise ConnectionError("Connection closed before receiving all data")
        data += chunk
        remaining -= len(chunk)
    
    return data

def generate_mask(model, seed):
    torch.manual_seed(seed)
    mask = {}
    for name, param in model.state_dict().items():
        print(f"Masking: {name}, dtype: {param.dtype}")
        if param.dtype in [torch.float32, torch.float64, torch.float16]:  # hanya untuk float
            mask[name] = torch.randn_like(param)
        else:
            # Untuk tensor non-float, isi dengan nol agar tidak diubah
            mask[name] = torch.zeros_like(param)
    return mask

def apply_mask(model, mask, mode='add'):
    """
    Tambahkan atau kurangi mask ke model.
    """
    state_dict = model.state_dict()
    for name in state_dict:
        if mode == 'add':
            state_dict[name] += mask[name]
        elif mode == 'sub':
            state_dict[name] -= mask[name]
    model.load_state_dict(state_dict)
    return model

def apply_mask_to_model(trained_model, client_id:int):
    print(f"Client ID: {client_id} menggunakan mode: {'add (+r)' if client_id % 2 != 0 else 'sub (-r)'}")

    seed = 1234 + client_id

    mask = generate_mask(trained_model, seed)

    if client_id % 2 == 0:
        masked_model = apply_mask(trained_model, mask, mode="sub")
    else:
        masked_model = apply_mask(trained_model, mask, mode="add")

    return masked_model    