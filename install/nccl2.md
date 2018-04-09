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
  * sudo cp -P lib/nccl_* /usr/local/cuda/lib64
  * sudo chmod a+r /usr/local/cuda-8.0/lib64/nccl*
  * run `ldconfig` assuming /usr/local/cuda/lib64 is in your ld.so.config

## Doing the build
When doing the build and running ./configure keep the following in mind:
   * NCCL 2.1 is considered 2.x so answer the question about NCCL version with 
     2.
   * ./configure wants the NCCL path to be the root of the files you downloaded
     it will add /include and /lib.


## Running
NCCL 2.x is not packaged with TensorFlow due to NVIDIA licensing it has to be
installed just like CUDA in any machine you are using. 
  
