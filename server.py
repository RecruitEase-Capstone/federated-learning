import socket
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import pickle
import os
import torch.nn.functional as F  

#Konfigurasi Server
HOST = '0.0.0.0'  # Mendengarkan di semua interface
PORT = 65433  # Port untuk mendengarkan
PICKLE_PROTOCOL = 4  # Protokol pickle yang kompatibel

# Locks untuk thread safety
client_lock = threading.Lock()
model_lock = threading.Lock()

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

def save_model(model, filename='global_model.pkl'):
    """Simpan model ke file pickle"""
    try:
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(model, f, protocol=PICKLE_PROTOCOL)
        print(f"Model berhasil disimpan di {filename}")
    except Exception as e:
        print(f"Kesalahan menyimpan model: {e}")

def send_large_data(sock, data):
    """Kirim data besar dengan pemeriksaan ukuran"""
    try:
        data_size = len(data)
        sock.sendall(data_size.to_bytes(4, byteorder='big'))
        
        chunk_size = 4096
        for i in range(0, len(data), chunk_size):
            sock.sendall(data[i:i+chunk_size])
        return True
    except Exception as e:
        print(f"Error saat mengirim data: {e}")
        return False

def recv_large_data(sock):
    """Terima data besar dengan pemeriksaan ukuran"""
    try:
        data_size_bytes = sock.recv(4)
        if not data_size_bytes:
            raise RuntimeError("Koneksi terputus saat membaca ukuran data")
            
        data_size = int.from_bytes(data_size_bytes, byteorder='big')
        print(f"Akan menerima {data_size} bytes data")
        
        data = bytearray()
        bytes_received = 0
        timeout_counter = 0
        
        sock.settimeout(10)  # Timeout 10 detik untuk setiap operasi recv
        
        while bytes_received < data_size:
            try:
                chunk = sock.recv(min(4096, data_size - bytes_received))
                if not chunk:
                    timeout_counter += 1
                    if timeout_counter > 5:  # Setelah 5 kali timeout, anggap koneksi terputus
                        raise RuntimeError("Koneksi terputus saat menerima data")
                    continue
                
                data.extend(chunk)
                bytes_received = len(data)
                print(f"Menerima chunk {len(chunk)} bytes, total {bytes_received}/{data_size} bytes")
                timeout_counter = 0  # Reset counter jika berhasil menerima data
            except socket.timeout:
                timeout_counter += 1
                print(f"Socket timeout #{timeout_counter}, menunggu data...")
                if timeout_counter > 5:  # Setelah 5 kali timeout, anggap koneksi terputus
                    raise RuntimeError("Socket timeout saat menerima data")
        
        sock.settimeout(None)  # Kembalikan ke blocking mode
        return bytes(data)
    except Exception as e:
        print(f"Error saat menerima data: {e}")
        raise

def federated_averaging(global_model, client_models, client_weights):
    """Lakukan federated averaging dengan penanganan error yang lebih baik"""
    try:
        with torch.no_grad():
            global_params = OrderedDict(global_model.named_parameters())
            
            # Inisialisasi parameter global dengan nol
            for name, param in global_params.items():
                param.data.zero_()
            
            # Pastikan jumlah model dan bobot sama
            if len(client_models) != len(client_weights):
                raise ValueError(f"Jumlah model ({len(client_models)}) dan bobot ({len(client_weights)}) tidak sama")
            
            # Pastikan hanya model yang tidak None yang digunakan
            valid_models = [model for model in client_models if model is not None]
            num_valid_models = len(valid_models)
            
            if num_valid_models == 0:
                raise ValueError("Tidak ada model klien yang valid untuk agregasi")
            
            # Rekalkulasi bobot untuk hanya model valid
            valid_weights = [1.0 / num_valid_models] * num_valid_models
            
            # Agregasi parameter
            for idx, (client_model_data, weight) in enumerate(zip(valid_models, valid_weights)):
                try:
                    # Check if client_model is a dictionary or a model object
                    if isinstance(client_model_data, dict) and 'state_dict' in client_model_data:
                        # It's a dictionary with state_dict
                        client_state_dict = client_model_data['state_dict']
                        for name, param in global_params.items():
                            if name in client_state_dict:
                                param.data += client_state_dict[name].data * weight
                            else:
                                print(f"WARNING: Parameter {name} tidak ditemukan di model klien {idx}")
                    else:
                        # Assume it's a model object
                        client_params = OrderedDict(client_model_data.named_parameters())
                        for name, param in global_params.items():
                            if name in client_params:
                                param.data += client_params[name].data * weight
                            else:
                                print(f"WARNING: Parameter {name} tidak ditemukan di model klien {idx}")
                except Exception as model_error:
                    print(f"Error memproses model klien {idx}: {model_error}")
                    # Lanjutkan dengan model berikutnya
            
            return global_model
    except Exception as e:
        print(f"Error dalam federated averaging: {e}")
        return global_model  # Kembalikan model asli jika terjadi error
    

def handle_client(conn, addr, global_model, client_models, client_weights, client_index):
    """Tangani koneksi klien dengan penanganan error yang lebih baik"""
    print(f"Terhubung oleh {addr}")
    try:
        # Terima model dari klien
        print(f"Menunggu model dari klien {addr}...")
        received_data = recv_large_data(conn)
        print(f"Menerima {len(received_data)} bytes dari klien {addr}")
        
        try:
            client_model_data = pickle.loads(received_data)
            print(f"Model dari klien {addr} berhasil dideserialisasi")
            
            # Update array model klien dengan thread safety
            with client_lock:
                client_models[client_index] = client_model_data
                all_received = all(model is not None for model in client_models)
            
            # Model yang akan dikirim kembali ke klien
            model_to_send = global_model
            
            # Jika semua model sudah diterima, lakukan federated averaging
            if all_received:
                print("Semua model klien diterima, melakukan federated averaging...")
                with model_lock:
                    updated_global_model = federated_averaging(global_model, client_models, client_weights)
                    model_to_send = updated_global_model
                    # Update model global untuk referensi di thread lain
                    global_model.load_state_dict(updated_global_model.state_dict())
                    print("Model global telah diperbarui.")
                
                # Simpan model global ke file
                save_model(model_to_send, f'./models/global/global_model_round_{client_index + 1}.pkl')
            
            # Serialisasi model untuk dikirim ke klien
            print(f"Menyiapkan model untuk dikirim ke klien {addr}")
            # In the server's handle_client function, when serializing the model:
            model_data_dict = {
                'state_dict': model_to_send.state_dict(),
                'architecture': 'CNN1D',
                'num_classes': model_to_send.fc2.out_features
            }
            model_data = pickle.dumps(model_data_dict, protocol=PICKLE_PROTOCOL)
            print(f"Model diserialisasi: {len(model_data)} bytes")
            
            # Kirim model
            print(f"Mengirim model ke klien {addr}")
            if send_large_data(conn, model_data):
                print(f"Model berhasil dikirim ke klien {addr}")
            else:
                print(f"Gagal mengirim model ke klien {addr}")
                
        except Exception as pickle_error:
            print(f"Error deserialize/serialize model: {pickle_error}")
            # Kirim model kosong sebagai fallback
            try:
                empty_model = CNN1D()
                model_data = pickle.dumps(empty_model, protocol=PICKLE_PROTOCOL)
                send_large_data(conn, model_data)
                print(f"Model kosong dikirim ke klien {addr} sebagai fallback")
            except:
                print(f"Gagal mengirim model fallback ke klien {addr}")
    
    except Exception as e:
        print(f"Error handling client {addr}: {e}")
    finally:
        try:
            conn.close()
            print(f"Koneksi dengan {addr} ditutup")
        except:
            pass

def main():
    # Inisialisasi Model Global
    global_model = CNN1D()
    
    # Inisialisasi Model Klien
    num_clients = 1
    client_models = [None] * num_clients
    client_weights = [1.0 / num_clients] * num_clients  # Bobot sama untuk setiap klien
    
    # Buat socket TCP
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Tambahkan opsi untuk menggunakan kembali alamat dan port
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server mendengarkan di {HOST}:{PORT}")
        
        client_threads = []
        client_count = 0
        
        try:
            while client_count < num_clients:
                # Tambahkan timeout pada accept()
                s.settimeout(60)  # 60 detik timeout
                try:
                    conn, addr = s.accept()
                    print(f"Klien baru terhubung: {addr}")
                    s.settimeout(None)  # Reset timeout setelah koneksi diterima
                    
                    client_thread = threading.Thread(
                        target=handle_client,
                        args=(conn, addr, global_model, client_models, client_weights, client_count)
                    )
                    client_thread.start()
                    client_threads.append(client_thread)
                    client_count += 1
                    print(f"Klien {client_count}/{num_clients} terhubung")
                except socket.timeout:
                    print("Timeout menunggu koneksi klien. Melanjutkan dengan klien yang ada.")
                    break
            
            # Tunggu semua thread selesai dengan timeout
            print("Menunggu semua thread klien selesai...")
            for i, thread in enumerate(client_threads):
                thread.join(timeout=120)  # 2 menit timeout untuk setiap thread
                if thread.is_alive():
                    print(f"WARNING: Thread klien {i} tidak selesai dalam waktu yang ditentukan")
            
            print("Semua klien selesai, server ditutup.")
            
            # Simpan model global akhir
            save_model(global_model, './models/global/final_global_model.pkl')
        
        except KeyboardInterrupt:
            print("Server dihentikan oleh user")
        except Exception as e:
            print(f"Error tidak terduga di server: {e}")

if __name__ == '__main__':
    main()