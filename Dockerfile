FROM tensorflow/tensorflow:1.13.1-gpu-py3

RUN pip install keras==2.*
RUN ln -nfs /usr/lib/x86_64-linux-gnu/libcudnn.so.7 /usr/lib/x86_64-linux-gnu/libcudnn.so
RUN mkdir -p /root/.keras/datasets/ && curl https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz > /root/.keras/datasets/cifar-10-batches-py.tar.gz