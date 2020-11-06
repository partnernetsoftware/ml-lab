CUDA_VERSION=$(nvidia-smi | grep " Version" | awk '{print $9}' | cut -c1-)
sudo docker run --gpus all -it --rm partnernetsoftware/gpu:cuda-$CUDA_VERSION bash
