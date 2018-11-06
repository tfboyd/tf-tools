# Compile TensorFlow with CUDA 10 and cuDNN 7.3 (8.0 is likely coming soon)
First, as of TF 1.11 and likely before, nothing is needed for CUDA 10 to work.
There are no major incompatible changes in CUDA 10 as there were in the move from CUDA 8.x
to CUDA 9.0. I also want to stress there is no point in upgrading unless you know
of a feature you need and that it has been implemented. In my limited testing cuDNN
7.2 and 7.3 make the difference not CUDA, although I am aware of a substantical
(2x more more) speedup in CUDA 9.2 and CUDA 10 for fp16 batch matmul but you would
have to be using it.  I beleive transformer and some other mlperf models benefit.

**These are notes not step-by-step instructions**:

# Environment
I use the tgz packages in this case because I find it more comfortable.  For my 
runtime I almost always use apt-get to install everything.  Due to how some of the
libraires rae install with apt-get it can make building from source a pain.  We
are working on improving the situation and part of it is beyond TensorFlow's control:

  * Ubuntu 16.04 LTS on GCE from base Google Cloud Ubuntu image.
  * Python 2.7 because that is how I roll and someday I will change.
  * Install 410+ driver using apt-get
  * CUDA 10.0.130  from .tgz install  (do not install the driver)
  * cudnn-10.0 v7.3.0.29 rom tgz install
  * nccl 2.3.4 for CUDA 10 (I am not sure it matters I have mixed them up before)
  * TensorRT-5.0.0 for CUDA 10 cudnn 7.3  (TensorRT 5 was not stated as supported by 
    Tensorflow when I did this but I suspect it works fine, but use what you want).
  * I am not covering all the pip packages I had to install, I usually follow 
    the install guide on tensorflow.org as a starting point.

**Note:** I use a clean VM for builds so I do not care if it gets mess because
I often start over if I want to do a different library set.  That said if you
use the .tgz install approach it is, in my opinion, easier to keep a clean setup with multiple
versions around.

# Doing the build
I install all of my NVIDIA libraires into /usr/local by hand or if the script does
it and then I add the paths to ldconfig because bazel makes me nuts and I using ldconfig
just gets the job done.

```bash
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout tags/v1.12.0-rc0
./configure
# Answers of interest
# Default almost everything, but
# CUDA -> Y and Version 10.0
# Yes NCCL  2.3
# Yes tensorrt 

# I build with haswell which gives AVX2 support and I am 
# too lazy to ensure I type out all of the various flags I want.
# use I think ivybridge if you want AVX.  If your GCC is older
# it may not support the haswell alias.
bazel build -c opt --copt=-march="broadwell" //tensorflow/tools/pip_package:build_pip_package
# Make the .whl
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
  
