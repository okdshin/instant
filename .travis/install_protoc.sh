#!/usr/bin/env sh
if [ ! -d "$HOME/protobuf/lib" ]; then
    wget https://protobuf.googlecode.com/files/protobuf-2.6.1.tar.gz
    tar -xzvf protobuf-2.6.1.tar.gz
    cd protobuf-2.6.1 && ./configure --prefix=$HOME/protobuf && make && make install
else
    echo "Using cached directory."
fi
