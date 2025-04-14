import os
import pickle
import socket

# Pickle protocol version
PICKLE_PROTOCOL = 4

def save_model(model, filename='global_model.pkl'):
    """Save model to pickle file"""
    try:
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(model, f, protocol=PICKLE_PROTOCOL)
        print(f"Model successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving model: {e}")

def serialize_model(model):
    """Serialize model for transmission"""
    model_data_dict = {
        'state_dict': model.state_dict(),
        'architecture': 'CNN1D',
        'num_classes': model.fc2.out_features
    }
    return pickle.dumps(model_data_dict, protocol=PICKLE_PROTOCOL)

def deserialize_model(data):
    """Deserialize model from received data"""
    return pickle.loads(data)

def send_large_data(sock, data):
    """Send large data with size verification"""
    try:
        data_size = len(data)
        sock.sendall(data_size.to_bytes(4, byteorder='big'))
        
        chunk_size = 4096
        for i in range(0, len(data), chunk_size):
            sock.sendall(data[i:i+chunk_size])
        return True
    except Exception as e:
        print(f"Error while sending data: {e}")
        return False

def recv_large_data(sock):
    """Receive large data with size verification"""
    try:
        data_size_bytes = sock.recv(4)
        if not data_size_bytes:
            raise RuntimeError("Connection dropped while reading data size")
            
        data_size = int.from_bytes(data_size_bytes, byteorder='big')
        print(f"Will receive {data_size} bytes of data")
        
        data = bytearray()
        bytes_received = 0
        timeout_counter = 0
        
        sock.settimeout(10)  # 10 second timeout for each recv operation
        
        while bytes_received < data_size:
            try:
                chunk = sock.recv(min(4096, data_size - bytes_received))
                if not chunk:
                    timeout_counter += 1
                    if timeout_counter > 5:  # After 5 timeouts, consider connection dropped
                        raise RuntimeError("Connection dropped while receiving data")
                    continue
                
                data.extend(chunk)
                bytes_received = len(data)
                print(f"Received chunk {len(chunk)} bytes, total {bytes_received}/{data_size} bytes")
                timeout_counter = 0  # Reset counter if data successfully received
            except socket.timeout:
                timeout_counter += 1
                print(f"Socket timeout #{timeout_counter}, waiting for data...")
                if timeout_counter > 5:  # After 5 timeouts, consider connection dropped
                    raise RuntimeError("Socket timeout while receiving data")
        
        sock.settimeout(None)  # Return to blocking mode
        return bytes(data)
    except Exception as e:
        print(f"Error while receiving data: {e}")
        raise