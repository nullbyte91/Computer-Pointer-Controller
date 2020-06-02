#!/bin/bash

dModel="./mo_model/"

sDownload="/opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py"

fFaceDetection="face-detection-adas-binary-0001"
fHeadPosition="head-pose-estimation-adas-0001"
fLandMark="facial-landmarks-35-adas-0002"
fGazeEstimation="gaze-estimation-adas-0002"

#Main starts
#Create a model directory
mkdir -p ${dModel}

# Download face-detection-adas-binary-0001
python3 ${sDownload} --name ${fFaceDetection} -o ${dModel}

# Download head-pose-estimation-adas-0001
python3 ${sDownload} --name ${fHeadPosition} -o ${dModel}

# Download facial-landmarks-35-adas-0002
python3 ${sDownload} --name ${fLandMark} -o ${dModel}

# Download gaze-estimation-adas-0002
python3 ${sDownload} --name ${fGazeEstimation} -o ${dModel}

echo "Download Done"