# Instant

Instant is DNN inference library.

Instant is written in C++14.

# Build

```
sh retrieve_data.sh
mkdir build && cd build
cmake ..
make
```

# Run VGG16

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

## Goal
- Easy to use.
- MKL-DNN
