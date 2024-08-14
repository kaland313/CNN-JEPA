FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

# Fix error: The repository 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease' is not signed.
# Based on: https://github.com/NVIDIA/nvidia-docker/issues/1631#issuecomment-1112682423
# NVIDIA solution: https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb


RUN apt-get update -y && apt-get install -y --no-install-recommends \
    tmux \
    nano \
    htop \
    wget \
    curl \
    git \
    libsm6 \        
    libxrender1 \  
    libfontconfig1 \ 
    ffmpeg \
    libxext6 \
    openssh-server \
    cmake \
    libncurses5-dev \
    libncursesw5-dev \
    build-essential

RUN echo 'PermitRootLogin yes\nSubsystem sftp internal-sftp\nX11Forwarding yes\nX11UseLocalhost no\nAllowTcpForwarding yes' > /etc/ssh/sshd_config
EXPOSE 22
RUN groupadd sshgroup
RUN mkdir /var/run/sshd
RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh

# ADD YOUR PUBLIC KEY HERE 
# COPY cm-docker.pub /root/.ssh
# RUN cat /root/.ssh/cm-docker.pub >> /root/.ssh/authorized_keys
RUN echo 'PATH=$PATH:/opt/conda/bin' >> ~/.bashrc # somehow conda is missing from PATH if login via ssh

# REPLACE 655Q6b3&k*9! WITH YOUR PASSWORD
RUN echo 'root:655Q6b3&k*9!' | chpasswd

# Force bash color prompt
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/g' ~/.bashrc

RUN git clone https://github.com/Syllo/nvtop.git -b 3.0.1 ~/nvtop
RUN mkdir -p ~/nvtop/build
RUN cd ~/nvtop/build && cmake .. -DNVIDIA_SUPPORT=ON -DAMDGPU_SUPPORT=OFF -DINTEL_SUPPORT=OFF
RUN cd ~/nvtop/build && make
RUN cd ~/nvtop/build && make install

RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh \n" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

COPY requirements.txt /workspace
RUN ["conda", "run", "-n", "base", "pip", "install", "-r", "/workspace/requirements.txt"]

CMD ["/bin/bash"]
