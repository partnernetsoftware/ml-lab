# sudo docker run --gpus all --rm nvidia/cuda nvidia-smi
# sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

CUDA_VERSION=$(nvidia-smi | grep " Version" | awk '{print $9}' | cut -c1-)
echo CUDA_VERSION=$CUDA_VERSION

# sudo docker run --gpus all -it --rm tensorflow/tensorflow:latest-gpu \
sudo docker run --gpus all -it --rm szubdi/gpu:cuda-$CUDA_VERSION \
	python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
