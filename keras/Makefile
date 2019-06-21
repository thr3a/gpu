CPUS := "0"
build:
	docker build -t tahara/keras-sample .
benchmark1:
	docker run -v `pwd`:/works --cpuset-cpus=${CPUS} -it --rm --runtime=nvidia tahara/keras-sample python works/scripts/cifar10_cnn.py
benchmark2:
	docker run -v `pwd`:/works --cpuset-cpus=${CPUS} -it --rm --runtime=nvidia tahara/keras-sample python works/scripts/babi_rnn.py
