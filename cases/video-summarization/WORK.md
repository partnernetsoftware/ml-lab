#

* https://github.com/tanveer-hussain/Video-Summarization
	* https://drive.google.com/drive/folders/1p8M3rjWF8h5km7uyQkYaxwo0ELx4OFsb?usp=sharing
		* cp *.caffemodel VS-Python/Models
		* mkdir -p $TargetPath(e.g. Keyframes/Cam1/temp) and put the testing file
		* main.py (change the pathh)
```
# using host GPU
https://github.com/NVIDIA/nvidia-docker
   
* distribution=$(./etc/os‐release;echo$ID$VERSION_ID)
* curl ‐s ‐L https://nvidia.github.io/nvidia‐docker/gpgkey|sudoapt‐keyadd‐
* curl ‐s ‐L https://nvidia.github.io/nvidia‐docker/$distribution/nvidia‐docker.list | sudo tee /etc/apt/sources.list.d/nvidia‐docker.list
* sudo apt‐get update && sudo apt‐get install ‐y nvidia‐container‐toolkit
* sudo systemctl restart docker

# which whether docker using GPU now
sudo docker run --gpus all nvidia/cuda:10.0-base nvidia-smi

# pull caffe image
sudo docker pull wzj751127122/caffe:19.12-py3

```
* https://github.com/tanveer-hussain/DeepRes-Video-Summarization
