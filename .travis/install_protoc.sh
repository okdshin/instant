#!/usr/bin/env sh
if [ ! -d "$HOME/protoc/lib" ]; then
    wget https://github.com/google/protobuf/archive/v2.6.1.tar.gz -O protobuf-2.6.1.tar.gz
    tar -xzvf protobuf-2.6.1.tar.gz
    cd protobuf-2.6.1
    ./autogen.sh
    ./configure --prefix=$HOME/protoc
    make && make install
    cd ..
else
    echo "Using cached directory."
fi
