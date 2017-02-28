# Install TensorFlow + CUDA on Ubuntu 16.04 on AWS
This install was done on AWS using Ubuntu 16.04 LTS Starting with m4.2xlarge to
save money during compile time.  Just using a p2.1xlarge instance is likely
easier an cost effective.  

Python 2.7, the default with Unbuntu 16.04

## Objective
Install NVIDIA drivers, compile tensorflow and then make an AMI that can be
launched on a p2.xLarge instance with NVIDIA GPUs (K80s) that support CUDA 8.0.
g2 instances will not work as they only support up to CUDA 7.5.

Note: This script is not intended to be the fastest approach. Some of the
steps could be turned into one-liners vs. manual work.

## VM Instance settings
If compiling TF from source start with 15GB.  To be extra safe and if there is
a desire to install other frameworks 20-30GB is a good size.  30GB is more than
enough in my expeience and 20GB should work.  If trying to keep space usage to a
minimum, it should be possible to have an 8-10GB VM and remember to delete
NVIDIA downloads after they are used.


```bash
############################################
# Install NVIDIA Driver
########################
# The best approach is to install the Ubuntu version of the driver.  Do not
# install the driver included in the CUDA package unless necessary.  
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update && sudo apt-get upgrade
# Check https://launchpad.net/~graphics-drivers/+archive/ubuntu/ppa to see the
# latest drivers available
sudo apt-get install nvidia-375

############################################
# Install basic packages needed for TensorFlow and generally needed
########################
sudo apt-get install -y build-essential git python-pip libfreetype6-dev libxft-dev libncurses-dev libopenblas-dev gfortran python-matplotlib libblas-dev liblapack-dev libatlas-base-dev python-dev python-pydot linux-headers-generic linux-image-extra-virtual unzip python-numpy swig python-pandas python-sklearn unzip wget pkg-config zip g++ zlib1g-dev libcurl3-dev
sudo apt-get install libcupti-dev bc

# Install Python package manager
sudo pip install -U pip
sudo pip install --upgrade pip
sudo pip install wheel numpy

############################################
# Install CUDA using packages rather than the script install.
########################
# If a new version of CUDA is out, get the link from NVIDIA's site.

wget https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_linux-run
./cuda_8.0.44_linux-run --extract=/home/ubuntu/
sudo ./cuda-linux64-rel-8.0.44-21122537.run

chmod +x cuda_8.0.44_linux-run
# DO NOT install the video driver and the sample are not really needed
./cuda_8.0.44_linux-run

############################################
# Install CuDNN
########################
# Download CuDNN from NVIDIA (get the Linux package not deb packages)
# scp to ec2 instance
# Cannot provide direct download due to needing to signup to get CuDNN.
tar zxf cudnn-8.0-linux-x64-v5.1.tgz

# Copy files into CUDA directories
sudo cp -P include/cudnn.h /usr/local/cuda-8.0/include/
sudo cp -P lib64/libcudnn* /usr/local/cuda-8.0/lib64/
sudo chmod a+r /usr/local/cuda-8.0/lib64/libcudnn*

# Setup Profile with CUDA environment variables
Add to ~profile
#CUDA Setup
export CUDA_HOME=/usr/local/cuda
export CUDA_ROOT=/usr/local/cuda
PATH=$PATH:$CUDA_ROOT/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64
source .profile

############################################
# Install Bazel
########################
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install bazel
sudo apt-get upgrade bazel

############################################
# Install TensorFlow
########################
git clone https://github.com/tensorflow/tensorflow

./configure
# Defaults with exception of
# Y Cuda / 8.0 / Compute 3.7 for K80s.

# Works for m2 and p2 instances and compiles in AVX2 optimizations.
bazel build -c opt --copt=-march="haswell" --config=cuda //tensorflow/tools/pip_package:build_pip_package

# Build the pip package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

# Install
pip install --upgrade --force-reinstall /tmp/tensorflow_pkg tensorflow*

###########################
#  Optional
#####################

# This removes a service that is not needed and looks, although it does not,
# to take up 100% CPU at times via top.
sudo apt-get remove gstreamer1.0

```















