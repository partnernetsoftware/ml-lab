CUDA_VERSION=$(nvidia-smi | grep " Version" | awk '{print $9}' | cut -c1-)
echo CUDA_VERSION=$CUDA_VERSION

# TODO
CUDA_PY_VER=cu110

DOCKER_IMG_NAME="partnernetsoftware"

cat << EEE | docker build - -t $DOCKER_IMG_NAME/gpu:cuda-$CUDA_VERSION
FROM tensorflow/tensorflow:latest-gpu

RUN apt install -y ffmpeg

RUN pip install -U pip

#RUN pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

RUN pip install torch torchvision torchaudio -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com && rm -rf /root/.cache

EEE
