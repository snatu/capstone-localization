import argparse
import numpy as np
import torch
import json
import pprint
from PIL import Image, ImageDraw
import spacy 


# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("json")

    parser.add_argument(
        '--vcr_dir',
        default='images/',
        help='directory with all of the VCR image data, contains, e.g., movieclips_Lethal_Weapon')

    parser.add_argument(
        '--vg_dir',
        default='images/',
        help='directory with visual genome data, contains VG_100K and VG_100K_2')

    args = parser.parse_args()
    return args

def url2filepath(args, url):
    if 'VG_' in url:
        return args.vg_dir + '/'.join(url.split('/')[-2:])
    else:
        # http://s3-us-west-2.amazonaws.com/ai2-rowanz/vcr1images/lsmdc_3023_DISTRICT_9/3023_DISTRICT_9_01.21.02.808-01.21.16.722@5.jpg
        if 'vcr1images' in args.vcr_dir:
            return args.vcr_dir + '/'.join(url.split('/')[-2:])
        else:
            return args.vcr_dir + '/'.join(url.split('/')[-3:])


def detect (predictor, url, obj_classes):
    im = cv2.imread(url)
    #cv2_imshow(im)

    
    outputs = predictor(im)
    pred = outputs["instances"].pred_classes.tolist()



    pred = [obj_classes[0] for i in pred]

    return pred

def noun_extractor (nlp, clue):
    doc = nlp(clue)
    noun = []
    for chunk in doc.noun_chunks:
        noun.append(chunk.root.text)
    return noun

def common (noun, im_obj):
    if (set(noun) & set(im_obj)):
        return True
    return False

def main():

    args = parse_args()
    
    
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    
    obj_classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes

    f = open(args.json)
    data = json.load(f)

    
    nlp = spacy.load("en_core_web_sm")
    score = 0
    count = 0

    incorrect = []
    for i in data:
        url = url2filepath(args, i["inputs"]["image"]["url"])
        clue = i["inputs"]["clue"]

        noun = noun_extractor(nlp, clue)
        im_obj = detect(predictor, url, obj_classes)

        if common (noun, im_obj):
            score+=1
        else:
            incorrect.append((i["inputs"]["image"]["url"], i["instance_id"]))
        count+=1
        break

    print(incorrect)
    print(score)

    

        



if __name__ == '__main__':
    main()
    
