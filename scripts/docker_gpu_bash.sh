CUDA_VERSION=$(nvidia-smi | grep " Version" | awk '{print $9}' | cut -c1-)
DOCKER_IMG_NAME=partnernetsoftware/gpu
docker run --gpus all -v $PWD:/work/ -w /work/ -u $(id -u):$(id -g) -it \
	$DOCKER_IMG_NAME:cuda-$CUDA_VERSION bash
