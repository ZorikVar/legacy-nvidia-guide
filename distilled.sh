#!/bin/bash

mkdir TestCuda
cd TestCuda
nvim test.cu
# #include <stdio.h>
# #include <cuda_runtime.h>
# 
# __global__ void addNumbers(int *result) {
#     *result = 2 + 2;
# }
# 
# int main() {
#     int *gpuResult, cpuResult;
#     cudaMalloc(&gpuResult, sizeof(int));
# 
#     addNumbers<<<1, 1>>>(gpuResult);
# 
#     // Wait for GPU to finish
#     cudaDeviceSynchronize();
# 
#     // Copy result to CPU
#     cudaMemcpy(&cpuResult, gpuResult, sizeof(int), cudaMemcpyDeviceToHost);
# 
#     // Verify
#     if (cudaGetLastError() != cudaSuccess) {
#         printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
#         return 1;
#     }
# 
#     printf("GPU calculated: 2 + 2 = %d\n", cpuResult);  // Will print "4"
#     cudaFree(gpuResult);
#     return 0;
# }

sudo apt install nvidia-driver-470 nvidia-utils-470

sudo apt install cuda-samples-11-4
# /usr/local/cuda-11.4/samples/

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
ls cuda-ubuntu2004.pin 
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 -i
# Package: nsight-compute
# Pin: origin *ubuntu.com*
# Pin-Priority: -1
#
# Package: nsight-systems
# Pin: origin *ubuntu.com*
# Pin-Priority: -1
#
# Package: *
# Pin: release l=NVIDIA CUDA
# Pin-Priority: 600

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update

# Following command would fail:
# sudo apt install cuda-toolkit-11-4
# The following packages have unmet dependencies:
#  libcufile-11-4 : Depends: liburcu6 but it is not installable

# Install uRCU version 6
wget http://archive.ubuntu.com/ubuntu/pool/main/libu/liburcu/liburcu6_0.11.1-2_amd64.deb
sudo dpkg -i liburcu6_0.11.1-2_amd64.deb

# This seems repetitive (repository has already been added several commands before)
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
cat /etc/apt/sources.list.d/archive_uri-https_developer_download_nvidia_com_compute_cuda_repos_ubuntu2004_x86_64_-jammy.list 
# CONTENT: deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /
# CONTENT: # deb-src https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /

# Should successfully install CUDA 11.4
sudo apt install cuda-toolkit-11-4
nvidia-smi
# Fri May  2 14:38:05 2025       
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 470.256.02   Driver Version: 470.256.02   CUDA Version: 11.4     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |                               |                      |               MIG M. |
# |===============================+======================+======================|
# |   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
# | N/A   43C    P8    N/A /  N/A |      3MiB /  2004MiB |      0%      Default |
# |                               |                      |                  N/A |
# +-------------------------------+----------------------+----------------------+
#                                                                                
# +-----------------------------------------------------------------------------+
# | Processes:                                                                  |
# |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
# |        ID   ID                                                   Usage      |
# |=============================================================================|
# |    0   N/A  N/A      1642      G   /usr/lib/xorg/Xorg                  2MiB |
# +-----------------------------------------------------------------------------+

ls /usr/local/cuda
# cuda/      cuda-11/   cuda-11.4/ 
ls /usr/local/cuda-11.4/bin
# bin2c              crt       cuda-gdbserver                cu++filt   ncu                          nsight-sys     nsys-ui       nvdisasm      nv-nsight-cu-cli  nvvp
# computeprof        cudafe++  cuda-install-samples-11.4.sh  cuobjdump  ncu-ui                       nsys           nvcc          nvlink        nvprof            ptxas
# compute-sanitizer  cuda-gdb  cuda-memcheck                 fatbinary  nsight_ee_plugins_manage.sh  nsys-exporter  nvcc.profile  nv-nsight-cu  nvprune
export PATH=/usr/local/cuda-11.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2021 NVIDIA Corporation
# Built on Mon_Oct_11_21:27:02_PDT_2021
# Cuda compilation tools, release 11.4, V11.4.152
# Build cuda_11.4.r11.4/compiler.30521435_0

# Following will still fail because gcc is too new.
nvcc -arch=sm_50 test.cu -o test
# nvcc warning : The 'compute_35', 'compute_37', 'compute_50', 'sm_35', 'sm_37' and 'sm_50' architectures are deprecated, and may be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
# /usr/include/stdio.h(189): error: attribute "__malloc__" does not take arguments
#
# /usr/include/stdio.h(201): error: attribute "__malloc__" does not take arguments
#
# ...

sudo apt install gcc-9 g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9
sudo update-alternatives --config gcc
gcc --version
# gcc (Ubuntu 9.5.0-1ubuntu1~22.04) 9.5.0
# Copyright (C) 2019 Free Software Foundation, Inc.
# This is free software; see the source for copying conditions.  There is NO
# warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

nvcc -arch=sm_50 test.cu -o test \
  -Xcompiler -Wno-attribute \
  -Xcompiler -Wno-deprecated-declarations
# nvcc warning : The 'compute_35', 'compute_37', 'compute_50', 'sm_35', 'sm_37' and 'sm_50' architectures are deprecated, and may be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
./test 
# GPU calculated: 2 + 2 = 4

sudo apt install python3 python3-pip
python3 --version
# python 3.10.12
pip3 --version
# pip 22.0.2 from /usr/lib/python3/dist-packages/pip (python 3.10)

# Do NOT run the following command. It would install the wrong version of torch.
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu114
# pip3 uninstall -y torch torchvision torchaudio

pip3 install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 \
  --extra-index-url https://download.pytorch.org/whl/cu113 --no-cache-dir
export PATH=$PATH:$HOME/.local/bin

python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
# PyTorch: 1.12.0+cu113, CUDA: 11.3
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
# CUDA Available: True

nvim quicktorch.py
# import torch
# x = torch.tensor([2]).cuda()
# print(x * 2)  # Should output: tensor([4], device='cuda:0')

# quicktorch.py won't work unless we install the suitable version of numpy
pip3 install numpy==1.21.6
python3 -c "import numpy; print(numpy.__version__)"
# 1.21.6

python3 quicktorch.py
# tensor([4], device='cuda:0')

nvim hardtorch.py
# import torch
# 
# # Create large tensors on GPU
# x = torch.randn(10000, 10000).cuda()
# y = torch.randn(10000, 10000).cuda()
# 
# # Matrix multiplication (GPU-accelerated)
# z = x @ y
# 
# print(f"GPU computation successful! Result shape: {z.shape}")
python3 hardtorch.py
# GPU computation successful! Result shape: torch.Size([10000, 10000])

# Optional, try to mitigate potential future problems related to system upgrade
# Disable upgrades for critical packages
sudo apt-mark hold nvidia-driver-470 nvidia-utils-470 cuda-toolkit-11-4 gcc-9 g++-9

# Future work:

# Prepare backups for following files:
# - /etc/apt/preferences.d/cuda-repository-pin-600
# - /etc/apt/sources.list.d/*cuda*

# Use python environments because other projects may require newer
# versions of numpy etc

# Risk: If NVIDIA drivers crash, a nomodeset boot entry will be helpful.
# Fix: Add a dedicated boot entry:
# Create file /boot/efi/loader/entries/pop_os-nomodeset.conf
# CONTENT: Title: Pop!_OS (Safe Mode - No GPU)
# CONTENT: Options: root=UUID=... ro nomodeset nouveau.modeset=0
