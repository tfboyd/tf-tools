# Install NCCL 2.x TensorFlow 1.8+
Instructions for building with NCCL 2.x with TensorFlow post 1.7.  
This should not require any real effort in the future and thus this document
will be obsolete.

# Steps

## Prerequisites
  * Everything needed to build TensorFlow from source with CUDA.
  * Download NCCL 2.x from NVIDIA because it is protected behind a login :-(. I
    chose to download the Linux agnostic version.
  * Get TensorFlow source.  I started at SHA-HASH:
    4f92c16be8b48c9a84053a42fdb736cae247b14f. Anything as of 09-PAR-2018 is
    fine.

## Setup

"Install" NCCL 2.x.  This is how I do it and is not necessarily the cleanest.
There is no need to do this unless you are doing to run TensorFlow on this
machine with nccl.  To only build just run the untar command and skip to the
next section.
  * tar xf nccl_*
  * sudo cp -P lib/libnccl* /usr/local/cuda/lib64
  * sudo chmod a+r /usr/local/cuda/lib64/libnccl*
  * run `ldconfig` assuming /usr/local/cuda/lib64 is in your ld.so.config

## Doing the build
When doing the build and running ./configure keep the following in mind:
   * NCCL 2.1 is considered 2.x so answer the question about NCCL version with 
     2.
   * ./configure wants the NCCL path to be the root of the files you downloaded
     it will add /include and /lib.

./configure output from just before TF 1.10.

```bash
You have bazel 0.11.1 installed.
Please specify the location of python. [Default is /usr/bin/python]: 


Found possible Python library paths:
  /usr/local/lib/python2.7/dist-packages
  /usr/lib/python2.7/dist-packages
Please input the desired Python library path to use.  Default is [/usr/local/lib/python2.7/dist-packages]

Do you wish to build TensorFlow with jemalloc as malloc support? [Y/n]: 
jemalloc as malloc support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: 
Google Cloud Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: 
Hadoop File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Amazon AWS Platform support? [Y/n]: 
Amazon AWS Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Apache Kafka Platform support? [Y/n]: 
Apache Kafka Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with XLA JIT support? [y/N]: 
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with GDR support? [y/N]: 
No GDR support will be enabled for TensorFlow.

Do you wish to build TensorFlow with VERBS support? [y/N]: 
No VERBS support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: 
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: y
CUDA support will be enabled for TensorFlow.

Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 9.0]: 

Please specify the location where CUDA 9.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 

Invalid path to CUDA 9.0 toolkit. /usr/local/cuda/lib64/libcudart.so.9.0 cannot be found
Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 9.0]: 9.2

Please specify the location where CUDA 9.2 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 

Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]: 

Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:

Do you wish to build TensorFlow with TensorRT support? [y/N]: 
No TensorRT support will be enabled for TensorFlow.

Please specify the NCCL version you want to use. [Leave empty to default to NCCL 1.3]: 2

Please specify the location where NCCL 2 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:/home/ubuntu/nccl_2.1.15-1+cuda9.0_x86_64

Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 3.5,5.2]7.0

Do you want to use clang as CUDA compiler? [y/N]: 
nvcc will be used as CUDA compiler.

Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: 

Do you wish to build TensorFlow with MPI support? [y/N]: 
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 

Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: 
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See tools/bazel.rc for more details.
	--config=mkl         	# Build with MKL support.
	--config=monolithic  	# Config for mostly static monolithic build.


bazel build -c opt --copt=-march="haswell" --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

```

## Running
NCCL 2.x is not packaged with TensorFlow due to NVIDIA licensing it has to be
installed just like CUDA in any machine you are using. 
  
