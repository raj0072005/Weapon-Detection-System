import torch
print('CUDA_AVAILABLE=', torch.cuda.is_available())
print('GPU_COUNT=', torch.cuda.device_count())
print('DEVICE_NAME=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')
