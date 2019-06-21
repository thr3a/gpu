CPUS := "0"
build:
	docker build -t tahara/keras-sample .
benchmark1:
	time docker run -v `pwd`:/works --cpuset-cpus=${CPUS} -it --rm --runtime=nvidia tahara/keras-sample python works/scripts/cifar10_cnn.py
benchmark2:
	time docker run -v `pwd`:/works --cpuset-cpus=${CPUS} -it --rm --runtime=nvidia tahara/keras-sample python works/scripts/babi_rnn.py
