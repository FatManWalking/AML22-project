import torch
torch.__version__ # Get PyTorch and CUDA version
torch.cuda.is_available() # Check that CUDA works
torch.cuda.device_count() # Check how many CUDA capable devices you have

# Print device human readable names
print(torch.cuda.get_device_name(0))
#torch.cuda.get_device_name(1)
# Add more lines with +1 like get_device_name(3), get_device_name(4) if you have more devices.