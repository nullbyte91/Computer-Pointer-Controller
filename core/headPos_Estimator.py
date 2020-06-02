#!/usr/bin/env python3
import os
import sys
import logging as log
import argparse 
import cv2
from math import cos, sin, pi
from utils.ie_module import Module
from utils.helper import cut_rois, resize_input

class HeadPosEstimator(Module):
    class Result:
        def __init__(self,output):
            self.head_position_x = output["angle_y_fc"][0] #Yaw
            self.head_position_y = output["angle_p_fc"][0] #Pitch
            self.head_position_z = output["angle_r_fc"][0] #Roll

    def __init__(self, model):
        super(HeadPosEstimator, self).__init__(model)
        assert len(model.inputs) == 1, "Expected 1 input blob"
        assert len(model.outputs) == 3, "Expected 1 output blob"
        self.input_blob = next(iter(model.inputs))
        self.output_blob = next(iter(model.outputs))
        self.input_shape = model.inputs[self.input_blob].shape

    def preprocess(self, frame, rois):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        inputs = cut_rois(frame, rois)
        inputs = [resize_input(input, self.input_shape) for input in inputs]
        return inputs
    
    def enqueue(self, input):
        return super(HeadPosEstimator, self).enqueue({self.input_blob: input})
    
    def start_async(self, frame, rois):
        inputs = self.preprocess(frame, rois)
        for input in inputs:
            self.enqueue(input)

    def get_headposition(self):
        outputs = self.get_outputs()
        return HeadPosEstimator.Result(outputs[0])


