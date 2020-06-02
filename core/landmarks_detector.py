import numpy as np

from utils.helper import cut_rois, resize_input
from utils.ie_module import Module

class LandmarksDetector(Module):
    POINTS_NUMBER = 5

    class Result:
        def __init__(self, outputs):
            self.points = outputs
            p = lambda i: self[i]
            self.left_eye = p(0)
            self.right_eye = p(1)
            self.nose_tip = p(2)
            self.left_lip_corner = p(3)
            self.right_lip_corner = p(4)
        def __getitem__(self, idx):
            return self.points[idx]

        def get_array(self):
            return np.array(self.points, dtype=np.float64)

    def __init__(self, model):
        super(LandmarksDetector, self).__init__(model)

        self.update = False
        assert len(model.inputs) == 1, "Expected 1 input blob"
        assert len(model.outputs) == 1, "Expected 1 output blob"
        self.input_blob = next(iter(model.inputs))
        self.output_blob = next(iter(model.outputs))
        self.input_shape = model.inputs[self.input_blob].shape

    def preprocess(self, frame, rois):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        inputs = cut_rois(frame, rois)
        
        inputs = [resize_input(input, self.input_shape) for input in inputs]
        return inputs

    def enqueue(self, input):
        return super(LandmarksDetector, self).enqueue({self.input_blob: input})

    def start_async(self, frame, rois):
        inputs = self.preprocess(frame, rois)
        for input in inputs:
            self.enqueue(input)

    def get_landmarks(self):
        outputs = self.get_outputs()
        results = [LandmarksDetector.Result(out[self.output_blob].reshape((-1, 2))) \
                      for out in outputs]
        return results
