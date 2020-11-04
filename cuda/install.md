# tips

## non docker mode

* buy a nvidia gpu, e.g. RTX 2080
* using ubuntu is prefer, or docker
* install cuda drivers (latest with your ubuntu version)
* pip install tensorflow-gpu
* pip install torch torchvision torchaudio ..

test py tf
```
import tensorflow as tf

#test = tf.test.is_gpu_available( cuda_only=False, min_cuda_compute_capability=None)
#test = tf.test.is_gpu_available( cuda_only=False )
#test = tf.test.is_gpu_available( cuda_only=True )
#print(test)
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print( tf.config.list_physical_devices('GPU'))

```

test py tc
```
import torch
print( torch.cuda.is_available())

```

## docker mode

* docker pull tensorflow/tensorflow:latest-gpu
* build on top
```
CUDA_VERSION=$(nvidia-smi | grep " Version" | awk '{print $9}' | cut -c1-)
echo CUDA_VERSION=$CUDA_VERSION

cat << EEE | docker build - -t partnernetsoftware/gpu:cuda-$CUDA_VERSION
FROM tensorflow/tensorflow:latest-gpu

RUN apt install -y ffmpeg

RUN pip install -U pip

#RUN pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
RUN pip install torch torchvision torchaudio -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com && rm -rf /root/.cache

#RUN apt install -y python3-venv
EEE
```

test tf-cuda-docker

```
CUDA_VERSION=$(nvidia-smi | grep " Version" | awk '{print $9}' | cut -c1-)
echo CUDA_VERSION=$CUDA_VERSION

sudo docker run --gpus all -it --rm partnernetsoftware/gpu:cuda-$CUDA_VERSION \
        python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

test tc-cuda-docker
```
CUDA_VERSION=$(nvidia-smi | grep " Version" | awk '{print $9}' | cut -c1-)
echo CUDA_VERSION=$CUDA_VERSION

sudo docker run --gpus all -it --rm partnernetsoftware/gpu:cuda-$CUDA_VERSION \
        python -c "import torch;print( torch.cuda.is_available())"
```

# lnx

* https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html
# http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_mac.dmg

# mac

brew cask install cuda

