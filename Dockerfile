FROM tensorflow/tensorflow:1.13.1-gpu-py3

RUN pip install keras==2.*
RUN ln -nfs /usr/lib/x86_64-linux-gnu/libcudnn.so.7 /usr/lib/x86_64-linux-gnu/libcudnn.so
ADD init_cifar10.py /
RUN python init_cifar10.py