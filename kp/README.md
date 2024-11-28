



## Regular
### Setup
source ~/traffic_env/bin/activate
[done] sudo apt install nvidia-cuda-toolkit  // May fail multiple times. Keep trying. 
[done] pip install pycuda  

[failed] pip install tensorflow==2.13.0  // Failed as matching version was not found 
[done] pip3 install torch torchvision torchaudio  // Takes long time (15min)

[done]] pip install tensorflow  // Done in 4mins

#### upgrade cuda-nn  // Done in 5mins
wget https://developer.download.nvidia.com/compute/cudnn/9.5.1/local_installers/cudnn-local-repo-ubuntu2404-9.5.1_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2404-9.5.1_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2404-9.5.1/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
sudo ldconfig
sudo apt install vim-gtk3


### Misc Setup
sudo apt install plocate


### Verify
nvcc --version
nvidia-smi


### Watch
watch -n 1 nvidia-smi


### Notes
TensorFlow 2.13.0 requires Python 3.8 - 3.10.
My Python version is 13.12.7

#### CUDA:
CUDA is a parallel computing platform and API from NVIDIA that allows developers to use the GPU for general-purpose computing.
#### cuDNN (CUDA Deep Neural Network):
cuDNN is a library built on top of CUDA that provides highly optimized primitives for deep learning.

#### Keras
Keras is an open-source, high-level deep learning framework. Keras is integrated as the default high-level API for TensorFlow.