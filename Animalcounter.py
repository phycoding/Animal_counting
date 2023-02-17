import numpy as np
import torch, detectron2
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import matplotlib
matplotlib.use('TKAgg')
     

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


class AnimalCounter:
    def __init__(self):
        self.predictor = self.load_model(0.5)
        self

    @staticmethod
    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)

        # Check if video file is opened successfully
        if (cap.isOpened()== False): 
            raise("Error opening video stream or file")
        else:
            return cap

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_model(self, threshold):
        cfg = get_cfg()

        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml")
        predictor = DefaultPredictor(cfg)
        return predictor

    def get_output_image(self, image):
        outputs = self.predictor(image)
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        return v.get_image()[:, :, ::-1]

    def check_version():
        TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
        CUDA_VERSION = torch.__version__.split("+")[-1]
        print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
        print("detectron2:", detectron2.__version__)