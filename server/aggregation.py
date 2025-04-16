import torch
from collections import OrderedDict

def federated_averaging(global_model, client_models, client_weights):
    """Perform federated averaging with better error handling"""
    print("Starting federated averaging...")
    try:
        with torch.no_grad():
            global_params = OrderedDict(global_model.named_parameters())
            
            # Initialize global parameters with zero
            for name, param in global_params.items():
                print(f"[{name}] mean after aggregation: {param.data.mean().item():.6f}")
                param.data.zero_()
            
            # Make sure the number of models and weights are the same
            if len(client_models) != len(client_weights):
                raise ValueError(f"Number of models ({len(client_models)}) and weights ({len(client_weights)}) mismatch")
            
            # Make sure only non-None models are used
            valid_models = [model for model in client_models if model is not None]
            num_valid_models = len(valid_models)
            
            if num_valid_models == 0:
                raise ValueError("No valid client models for aggregation")
            
            # Recalculate weights for only valid models
            valid_weights = [1.0 / num_valid_models] * num_valid_models
            
            # Aggregate parameters
            for idx, (client_model_data, weight) in enumerate(zip(valid_models, valid_weights)):
                try:
                    # Check if client_model is a dictionary or a model object
                    if isinstance(client_model_data, dict) and 'state_dict' in client_model_data:
                        # It's a dictionary with state_dict
                        client_state_dict = client_model_data['state_dict']
                        for name, param in global_params.items():
                            if name in client_state_dict:
                                mean_val = client_state_dict[name].mean().item()
                                print(f" before - [{name}] mean: {mean_val:.6f}")
                                param.data += client_state_dict[name].data * weight
                            else:
                                print(f"WARNING: Parameter {name} not found in client model {idx}")
                    else:
                        # Assume it's a model object
                        client_params = OrderedDict(client_model_data.named_parameters())
                        for name, param in global_params.items():
                            if name in client_params:
                                param.data += client_params[name].data * weight
                            else:
                                print(f"WARNING: Parameter {name} not found in client model {idx}")
                except Exception as model_error:
                    print(f"Error processing client model {idx}: {model_error}")
                    # Continue with the next model

            print(f"\n✅ Model global AFTER aggregation (expected cancel mask):")
            for name, param in global_params.items():
                if param.dtype == torch.float32:
                    print(f"  - [{name}] mean: {param.data.mean().item():.6f}")

            return global_model
    except Exception as e:
        print(f"Error in federated averaging: {e}")
        return global_model  # Return original model if error occurs