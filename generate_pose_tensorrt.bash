#!/bin/bash
# This script generates a TensorRT engine from pretrained FastPose model.

# download link to onnx model
onnx_url=$1

# float16=fp16 or float32=fp32 
precision=$2

#TensorRT batch_size, 8 is tested on Jetson TX2
batch_size=$3
tensorrt_model_name="fast_pose_"$precision"_b"$batch_size".trt"
workspace=0
if (( $batch_size > 1 )); then
	workspace=4096
else
	workspace=2000
fi

wget $1 -O "models/data/model_static.onnx"

/usr/src/tensorrt/bin/trtexec --onnx=models/data/model_static.onnx --saveEngine=models/data/$tensorrt_model_name --$precision --batch=$batch_size --workspace=$workspace

