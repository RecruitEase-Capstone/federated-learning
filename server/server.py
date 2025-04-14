import socket
import threading
import os
from cnn_model import CNN1D
from server.server_util import save_model, send_large_data, recv_large_data, deserialize_model, serialize_model
from server.aggregation import federated_averaging

class FederatedLearningServer:
    def __init__(self, host='0.0.0.0', port=65433, num_clients=1):
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.global_model = CNN1D()
        self.client_models = [None] * num_clients
        self.client_weights = [1.0 / num_clients] * num_clients
        self.client_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.save_path = './models/global/'
        
        # Ensure directories exist
        os.makedirs(self.save_path, exist_ok=True)
    
    def handle_client(self, conn, addr, client_index):
        """Handle client connection and model exchange"""
        print(f"Connected by {addr}")
        try:
            # Receive model from client
            print(f"Waiting for model from client {addr}...")
            received_data = recv_large_data(conn)
            print(f"Received {len(received_data)} bytes from client {addr}")
            
            try:
                client_model_data = deserialize_model(received_data)
                print(f"Model from client {addr} successfully deserialized")
                
                # Update client models array with thread safety
                with self.client_lock:
                    self.client_models[client_index] = client_model_data
                    all_received = all(model is not None for model in self.client_models)
                
                # Model to send back to client
                model_to_send = self.global_model
                
                # If all models have been received, perform federated averaging
                if all_received:
                    print("All client models received, performing federated averaging...")
                    with self.model_lock:
                        updated_global_model = federated_averaging(
                            self.global_model, 
                            self.client_models, 
                            self.client_weights
                        )
                        model_to_send = updated_global_model
                        # Update global model for reference in other threads
                        self.global_model.load_state_dict(updated_global_model.state_dict())
                        print("Global model has been updated.")
                    
                    # Save global model to file
                    save_model(model_to_send, f'{self.save_path}global_model_round_{client_index + 1}.pkl')
                
                # Serialize model for sending to client
                print(f"Preparing model to send to client {addr}")
                model_data = serialize_model(model_to_send)
                print(f"Model serialized: {len(model_data)} bytes")
                
                # Send model
                print(f"Sending model to client {addr}")
                if send_large_data(conn, model_data):
                    print(f"Model successfully sent to client {addr}")
                else:
                    print(f"Failed to send model to client {addr}")
                    
            except Exception as pickle_error:
                print(f"Error deserializing/serializing model: {pickle_error}")
                # Send empty model as fallback
                try:
                    empty_model = CNN1D()
                    model_data = serialize_model(empty_model)
                    send_large_data(conn, model_data)
                    print(f"Empty model sent to client {addr} as fallback")
                except Exception as fallback_error:
                    print(f"Failed to send fallback model to client {addr}: {fallback_error}")
        
        except Exception as e:
            print(f"Error handling client {addr}: {e}")
        finally:
            try:
                conn.close()
                print(f"Connection with {addr} closed")
            except:
                pass
    
    def start(self):
        """Start the server and listen for client connections"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Add option to reuse address and port
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen()
            print(f"Server listening on {self.host}:{self.port}")
            
            client_threads = []
            client_count = 0
            
            try:
                while client_count < self.num_clients:
                    # Add timeout to accept()
                    s.settimeout(60)  # 60 seconds timeout
                    try:
                        conn, addr = s.accept()
                        print(f"New client connected: {addr}")
                        s.settimeout(None)  # Reset timeout after connection is accepted
                        
                        client_thread = threading.Thread(
                            target=self.handle_client,
                            args=(conn, addr, client_count)
                        )
                        client_thread.start()
                        client_threads.append(client_thread)
                        client_count += 1
                        print(f"Client {client_count}/{self.num_clients} connected")
                    except socket.timeout:
                        print("Timeout waiting for client connection. Continuing with existing clients.")
                        break
                
                # Wait for all threads to finish with timeout
                print("Waiting for all client threads to finish...")
                for i, thread in enumerate(client_threads):
                    thread.join(timeout=120)  # 2 minutes timeout for each thread
                    if thread.is_alive():
                        print(f"WARNING: Client thread {i} did not finish within the specified time")
                
                print("All clients finished, server closing.")
                
                # Save final global model
                save_model(self.global_model, f'{self.save_path}final_global_model.pkl')
            
            except KeyboardInterrupt:
                print("Server stopped by user")
            except Exception as e:
                print(f"Unexpected error in server: {e}")

def main():
    server = FederatedLearningServer()
    server.start()

if __name__ == '__main__':
    main()