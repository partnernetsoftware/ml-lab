
@ztuc-mk13
@ztuc-mk

```
https://github.com/tanveer-hussain/Video-Summarization
	=> https://drive.google.com/drive/folders/1p8M3rjWF8h5km7uyQkYaxwo0ELx4OFsb?usp=sharing
		* cp *.caffemodel VS-Python/Models
		* mkdir -p $TargetPath(e.g. Keyframes/Cam1/temp) and put the testing file
		* main.py (tune pathh)
```

# setup

## install nvidia w/ GPU

https://github.com/NVIDIA/nvidia-docker

```
steps (notes)

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
echo $distribution
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## check whether docker using GPU now (Q: check where?)

sudo docker run --gpus all nvidia/cuda:10.0-base nvidia-smi

## pull caffe image (Q: who is wzj751127122? and where find him for future update? A: https://blog.csdn.net/a751127122 ?)

* https://hub.docker.com/r/wzj751127122/caffe

```
sudo docker pull wzj751127122/caffe:19.12-py3

# w/ caffe0.17.3，NVIDIA_caffe19.12，Python3.6，cuda10.2， cuda_driver 440.33

# Dockerfile
https://hub.docker.com/layers/wzj751127122/caffe/19.11-py3/images/sha256-8765cd0f5e72aff383abbaffd79f3fa845bd8b0299977c94a944872a3b2a11cd?context=explore

```

## ffmpeg

```
sudo apt install ffmpeg
```

## python

```
# ubuntu 
## install pip3
sudo apt install -y python3-pip
## some depends
sudo apt install -y cmake

#
python3 -m venv venv
source venv/bin/activate
python -V
pip -V

# update pip it self
pip -V && pip install --upgrade pip && pip -V

# pip install scikit-build
pip install scikit-build -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# opencv for python
# CDN # pip --default-timeout=100 install opencv-python -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
# pip install opencv-python
pip install opencv-python -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# sklearn
# CDN # pip --default-timeout=100 install sklearn -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
# pip install sklearn
pip install sklearn -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

```

## setup tensorflow

```
source venv/bin/activate
pip install tensorflow -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

```
# References

* https://github.com/tanveer-hussain/DeepRes-Video-Summarization
