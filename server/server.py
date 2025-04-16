import socket
import threading
import os
from cnn_model import CNN1D
from server.server_util import save_model, send_large_data, recv_large_data, deserialize_model, serialize_model
from server.aggregation import federated_averaging

class FederatedLearningServer:
    def __init__(self, host='0.0.0.0', port=65433, num_clients=2):
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.global_model = CNN1D()
        self.client_models = [None] * num_clients
        self.client_weights = [1.0 / num_clients] * num_clients
        self.client_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.save_path = './models/global/'
        self.all_models_received = False
        self.client_connections = []
        self.averaging_complete = threading.Event()
        
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
                    
                    # Store the connection for later response
                    self.client_connections.append((conn, addr, client_index))
                
                # If all models have been received, perform federated averaging once
                if all_received and not self.all_models_received:
                    with self.model_lock:
                        self.all_models_received = True
                        print("All client models received, performing federated averaging...")
                        updated_global_model = federated_averaging(
                            self.global_model, 
                            self.client_models, 
                            self.client_weights
                        )
                        # Update global model
                        self.global_model.load_state_dict(updated_global_model.state_dict())
                        print("Global model has been updated.")
                    
                    # Save global model to file
                    save_model(self.global_model, f'{self.save_path}global_model.pkl')
                    
                    # Signal that averaging is complete, allowing responses to be sent
                    self.averaging_complete.set()
                    
                    # Now send the updated global model to all clients
                    self.send_models_to_all_clients()
                else:
                    # Wait for averaging to complete before sending model
                    print(f"Client {addr} waiting for all models to be received...")
                    # We don't send any model yet, just keep the connection open
                    
            except Exception as pickle_error:
                print(f"Error deserializing model: {pickle_error}")
                # Close connection on error
                conn.close()
                print(f"Connection with {addr} closed due to error")
        
        except Exception as e:
            print(f"Error handling client {addr}: {e}")
            try:
                conn.close()
            except:
                pass
    
    def send_models_to_all_clients(self):
        """Send the updated global model to all connected clients"""
        print("Sending global model to all clients")
        model_data = serialize_model(self.global_model)
        print(f"Global model serialized: {len(model_data)} bytes")
        
        for conn, addr, client_index in self.client_connections:
            try:
                print(f"Sending model to client {addr}")
                if send_large_data(conn, model_data):
                    print(f"Model successfully sent to client {addr}")
                else:
                    print(f"Failed to send model to client {addr}")
            except Exception as e:
                print(f"Error sending model to client {addr}: {e}")
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
                    s.settimeout(600)  # 10 minutes timeout
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
                
                # Wait for model averaging to complete with timeout
                self.averaging_complete.wait(timeout=300)  # 5 minutes timeout
                
                # If averaging didn't complete, send whatever we have
                if not self.averaging_complete.is_set():
                    print("Timeout waiting for all client models. Using available models for averaging.")
                    with self.model_lock:
                        # Filter out None values
                        valid_models = [model for model in self.client_models if model is not None]
                        valid_weights = [self.client_weights[i] for i, model in enumerate(self.client_models) if model is not None]
                        
                        if valid_models:
                            # Normalize weights
                            sum_weights = sum(valid_weights)
                            valid_weights = [w/sum_weights for w in valid_weights]
                            
                            # Perform averaging with available models
                            updated_global_model = federated_averaging(
                                self.global_model,
                                valid_models,
                                valid_weights
                            )
                            self.global_model.load_state_dict(updated_global_model.state_dict())
                            
                            # Save partial global model
                            save_model(self.global_model, f'{self.save_path}partial_global_model.pkl')
                            
                            # Send models to clients
                            self.send_models_to_all_clients()
                
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