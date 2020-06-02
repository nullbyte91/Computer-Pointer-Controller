#!/usr/bin/env python3
import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore
from utils.ie_module import Module
import numpy as np
import cv2

class GazeEestimator(Module):
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self, model):
        super(GazeEestimator, self).__init__(model)
        assert len(model.inputs) == 3, "Expected 1 input blob"
        assert len(model.outputs) == 1, "Expected 1 output blob"
        # For gaze estimation model has three input blobs
        # 1. right_eye_image
        # 2. head_pose_angles
        # 3. left_eye_image

        self.input_blob = [] 
        self.input_shape = []
        
        for inputs in model.inputs:
            self.input_blob.append(inputs)
            self.input_shape.append(model.inputs[inputs].shape)
        self.output_blob = next(iter(model.outputs))
    
    def enqueue(self, head_pose, right_eye, left_eye):
        return super(GazeEestimator, self).enqueue({'left_eye_image': left_eye,
                                                    'right_eye_image': right_eye,
                                                    'head_pose_angles': head_pose})

    def start_async(self, headPosition, right_eye_image, left_eye_image):
        head_pose = [headPosition.head_position_x, 
                    headPosition.head_position_y, 
                    headPosition.head_position_z]

        head_pose = np.array([head_pose])
        head_pose = head_pose.flatten()

        left_eye = cv2.resize(left_eye_image, (60, 60), interpolation = cv2.INTER_AREA)
        left_eye = np.moveaxis(left_eye, -1, 0)

        right_eye = cv2.resize(right_eye_image, (60, 60), interpolation = cv2.INTER_AREA)
        right_eye = np.moveaxis(right_eye, -1, 0)
        self.enqueue(head_pose, right_eye, left_eye)
        
    def get_gazevector(self):
        outputs = self.get_outputs()
        return outputs


        
        