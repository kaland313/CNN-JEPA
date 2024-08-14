host_ssh_port=2222
image_name=ijepa
container_name=ijepa
data_path=/mnt/hdd2/datasets/

run: build
	nvidia-docker run \
	-it --rm \
	--shm-size 16G \
	--network host \
	-e NVIDIA_VISIBLE_DEVICES=0,1 \
	--name $(container_name) \
	-v $(shell pwd):/workspace/I-JEPA-CNN \
	-v $(data_path):/data \
	$(image_name) \
	/bin/bash

build:
	docker build --tag $(image_name)  .

push:
	docker push $(image_name) 

pull:
	docker pull $(image_name) 

stop:
	docker stop $(container_name)

ssh: build
	nvidia-docker run \
	-dt --rm \
	--shm-size 16G \
	-p $(host_ssh_port):22 \
	-e NVIDIA_VISIBLE_DEVICES=0,1 \
	--name $(container_name) \
	-v $(shell pwd):/workspace/I-JEPA-CNN \
	-v $(data_path):/data \
	$(image_name) \
	/usr/sbin/sshd -D
