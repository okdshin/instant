# Instant

[![Build Status](https://travis-ci.org/okdshin/instant.svg?branch=master)](https://travis-ci.org/okdshin/instant)

Instant is DNN inference library written in C++.

Instant is released under MIT Licence.

## Goal

- DNN Inference with CPU
- C++
- ONNX support
- Easy to use.

# Requirement

- MKL-DNN Library
- ProtocolBuffers

# Build

Execute below commands in root directory.

```
sh retrieve_data.sh
mkdir build && cd build
cmake ..
make
```

# Installation

Execute below command in build directory created at Build section.

```
make install
```

# Run VGG16 example

Execute below command in root directory.

```
./example/vgg16_example
```

# Current supported nodes

- Conv (2D)
- Relu
- MaxPool
- Reshape (nchw -> nc)
- FC
- Dropout
- Softmax

