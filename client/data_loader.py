import torch
import numpy as np
import os
import json

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

def load_data_from_json(json_file_path, class_mapper_path, base_dir=""):
    """Load and prepare dataset from JSON file"""
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