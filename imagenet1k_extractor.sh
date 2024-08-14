#!/bin/bash

find . -name "*.tar" | while read NAME ; do
    TARGET_DIR="../imagenet/train/${NAME%.tar}"
    if [ ! -d "$TARGET_DIR" ]; then
        mkdir -p "$TARGET_DIR"
        tar -xf "$NAME" -C "$TARGET_DIR"
        echo "$NAME"
    else
        echo "Skipping extraction for $NAME: Target directory already exists."
    fi
done
