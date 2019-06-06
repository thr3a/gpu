FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

RUN pip install scikit-learn==0.21.*
# RUN ln -nfs /usr/lib/x86_64-linux-gnu/libcudnn.so.7 /usr/lib/x86_64-linux-gnu/libcudnn.so
# RUN mkdir -p /root/.keras/datasets/ && curl https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz > /root/.keras/datasets/cifar-10-batches-py.tar.gz