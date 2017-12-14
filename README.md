# Instant

Instant is DNN inference library written in C++.
Instant is released under MIT Licence.

## Goal
- DNN Inference
- C++
- ONNX support
- Easy to use.

# Requirement

- MKL-DNN Library

# Build

```
sh retrieve_data.sh
mkdir build && cd build
cmake ..
make
```

# Run VGG16 example

```
example/vgg16_example
```

# Current supported nodes
- Conv (2D)
- Relu
- MaxPool
- Reshape (nchw -> nc)
- FC
- Dropout
- Softmax

