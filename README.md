# 必要な環境

- Docker
- [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker)

# 使い方

準備

```
make build
```

実行

```
make benchmark1
make benchmark2
```

マルチコアCPUでやる場合

```
# 4コアで実行する場合
make benchmark1 CPUS=0-3
```