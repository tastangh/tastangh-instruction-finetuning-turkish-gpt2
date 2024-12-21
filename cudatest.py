import torch

# FP16 desteğini kontrol et
fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7

# BF16 desteğini kontrol et
bf16_supported = torch.cuda.is_bf16_supported()

print(f"FP16 destekleniyor mu? {'Evet' if fp16_supported else 'Hayır'}")
print(f"BF16 destekleniyor mu? {'Evet' if bf16_supported else 'Hayır'}")
