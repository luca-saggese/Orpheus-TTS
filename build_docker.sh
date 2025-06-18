#!/bin/bash
docker build -t orpheus .
echo "✅ Build completata!"
echo "👉 Per eseguire il container usa:"
echo "docker run --rm -it --gpus all -p 8083:8080 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /data/huggingface:/huggingface orpheus bash"
