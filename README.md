# Introduction

*DistML* is a [Ray](https://github.com/ray-project/ray) extension library to support large-scale distributed ML training 
on heterogeneous multi-node multi-GPU clusters. This library is under active development and we are adding more advanced 
training strategies and auto-parallelization features. 

DistML currently supports:
* Distributed training strategies
    * Data parallelism
        * AllReduce strategy
        * Sharded parameter server strategy
        * BytePS strategy
    Pipeline parallleism
        * Micro-batch pipeline parallelism
    
* DL Frameworks:
    * PyTorch
    * JAX

# Installation

### Install Dependencies
Depending on your CUDA version, install cupy following https://docs.cupy.dev/en/stable/install.html.

### Install from source for dev
```python
pip install -e .
```