> **Galliot now hosts all Neuralet’s content and expertise gained in three years of work and completing high-quality applications, mainly in Computer Vision and Deep Learning. [More Info](https://galliot.us/blog/neuralet-migrated-to-galliot/)**

# Galliot Adaptive Pose Estimation

This is the Galliot's Adaptive Pose Estimation repository. With this module you can run specialized pose estimation models with the help of [Galliot's Adaptive Learning API](https://api.neuralet.io/).

## Supported Devices

* X86 devices with Nvidia GPUs
* Nvidia Jetson family are supported. (JetPack 4.4)\
For running pose estimator you should have [Docker](https://docs.docker.com/get-docker/) and [Nvidia Docker Toolkit](https://github.com/NVIDIA/nvidia-docker) on your system

## Getting Started

1. Start an Adaptive Learning Job in [API's website](https://api.neuralet.io/) for your specific video and download the specialized model.

Note that your models categories must contains **person** class.

2. Clone this repository:

```
git clone --recurse-submodules https://github.com/galliot-us/adaptive-pose-estimation.git
cd adaptive-pose-estimation
```

3-1. Build and run docker container on X86 with GPU:
```
docker build -f x86-gpu.Dockerfile -t "galliot/pose-estimation:latest-x86_64_gpu" .
docker run -it --gpus all -v "$PWD":/repo galliot/pose-estimation:latest-x86_64_gpu
```

3-2. Build and run docker container on Jetson devices:
```
docker build -f jetson.Dockerfile -t "galliot/pose-estimation:latest-jetson" .
docker run --runtime nvidia --entrypoint bash --privileged -it -v $PWD/:/repo galliot/pose-estimation:latest-jetson
```
* Make sure you run the following bash script inside the Jetson container to export TRT model.
  ```
  # bash generate_pose_tensorrt.bash [ONNX FILE URL] [Stored on FLOAT16(fp16)/ FLOAT32(fp32)] [BATCH_SIZE]
  bash generate_pose_tensorrt.bash https://media.githubusercontent.com/media/galliot-us/models/master/ONNX/fastpose/fastpose_resnet50_256_192_tf.onnx fp16 8
  ```
NOTE: For Jetson TX2 batch size 8 is recommended. For other devices you can test multiple batch sizes to find the optimum one.

4. Start Inference:

```
python3 inference/inference.py --device DEVICE --input_video INPUT_VIDEO                                
                    --out_dir OUT_DIR
                    --detector_model_path DETECTOR_MODEL_PATH                                                                                                      
                    --label_map LABEL_MAP                                                                                                                                  
                    --detector_threshold DETECTOR_THRESHOLD                                                                                                             
                    --detector_input_width DETECTOR_INPUT_WIDTH                                                                                                           
                    --detector_input_height DETECTOR_INPUT_HEIGHT                                                                                                          
                    --pose_input_width POSE_INPUT_WIDTH
                    --pose_input_height POSE_INPUT_HEIGHT
                    --heatmap_width HEATMAP_WIDTH
                    --heatmap_height HEATMAP_HEIGHT
                    --out_width OUT_WIDTH
                    --out_height OUT_HEIGHT
                    --batch_size BATCH_SIZE
                    --trt_model_path TRT_MODEL_PATH



```
```
optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE       supports x86 and jetson
  --input_video INPUT_VIDEO
                        input video path
  --out_dir OUT_DIR     directory to store output video
  --detector_model_path DETECTOR_MODEL_PATH
                        path to the detector model files, if not provided the
                        default COCO models will be used
  --label_map LABEL_MAP
                        path to the label map file
  --detector_threshold DETECTOR_THRESHOLD
                        detection's score threshold
  --detector_input_width DETECTOR_INPUT_WIDTH
                        width of the detector's input
  --detector_input_height DETECTOR_INPUT_HEIGHT
                        height of the detector's input
  --pose_input_width POSE_INPUT_WIDTH
                        width of the pose estimator's input
  --pose_input_height POSE_INPUT_HEIGHT
                        height of the pose estomator's input
  --heatmap_width HEATMAP_WIDTH
                        width of the pose haetmap
  --heatmap_height HEATMAP_HEIGHT
                        height of the pose heatmap
  --out_width OUT_WIDTH
                        width of the output video
  --out_height OUT_HEIGHT
                        height of the output video
  --batch_size BATCH_SIZE
                        the trt model batch size (works only for Jetson devices)
  --trt_model_path TRT_MODEL_PATH
                        the path of trt model (works only for Jetson devices)

```
Note: pass your specialized model to the `detecor_model_path` argument and its `label_map.pbtxt` file to the `label_map` argument. Otherwise the default COCO model will be used.
