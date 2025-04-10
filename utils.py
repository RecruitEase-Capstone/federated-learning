import torch

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
