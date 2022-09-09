FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# RUN apt-get update -y
# RUN ainstall -y --no-install-recommends \
# 	curl \
# 	wget 
RUN apt-key del 3bf863cc
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update -y

RUN	apt-get install -y --no-install-recommends \
	curl \
	wget \
	git \
        python3 \
        python3-dev \
	python3-setuptools \
        python3-pip \
	unzip \
	vim \
	sudo



RUN apt-get update -y && \
    apt-get install -y apt-utils && \
    apt-get install -y software-properties-common htop openssh-server procps \
    sudo

RUN apt-get install 'ffmpeg' \
    'libsm6' \
    'libxext6' -y

RUN pip3 install --upgrade pip

RUN pip3 install \
	cmake \
        jupyter \
        matplotlib \
        pandas \
	sklearn \
	scikit-image \
	imgaug \
	scikit-build \
    rsa \
	QtPy \
	PyYAML \
	qtconsole \
	pydicom \
	nilearn \
	nibabel \
	pytorch-lightning\
	#mesh-tensorflow \
	torchio \
	pynetdicom \
	Keras \
	jedi \
	h5py \
	six \
	mecab-python3 \
	tensorflow \
	moviepy \
	Pillow \
	boto \
	boto3 \
	botocore \
	tensorboardX \
	SimpleITK \
	scipy \
	lmdb \
	opencv-python \
	cryptography \
	h5py \
	pycrypto \
	tqdm \
	urllib3 \
	zipp \
	tables \
	wheel \
	albumentations
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install \
	optuna \
	numpy \
	coral_pytorch \
	sklearn \
	glob2 \
	efficientnet_pytorch 

#ENV export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/targets/x86_64-linux/lib

RUN apt update && apt install -y sudo
# IPython
EXPOSE 8888
EXPOSE 22
RUN useradd -u 1008 -G sudo,root shono
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
# SSH接続のための設定です。
# "password"のところを任意のパスワードへ変更してください。
# いろいろ設定しているのは、Ubuntu18.04でrootユーザでSSH接続するための作業です。
RUN mkdir /var/run/sshd
RUN echo 'root:password' | chpasswd
RUN echo 'shono:pass' | chpasswd
RUN echo '\nPermitRootLogin yes' >> /etc/ssh/sshd_config
RUN echo '\nX11UseLocalhost no' >> /etc/ssh/sshd_config
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# GPUの設定です。GPUを使用しない場合は不要です。
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES utility,compute

# SSH接続のための設定です。
ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
RUN usermod --shell /bin/bash shono
USER shono
VOLUME /shono
#COPY set_up_tf.sh /
COPY --chown=shono notebooks /notebooks
WORKDIR "/notebooks"
CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/notebooks --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token=\"shono\""]
