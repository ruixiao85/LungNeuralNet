"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018/
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a new model starting from ImageNet weights
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet
    # Train a new model starting from specific weights file
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5
    # Resume training a model that you had trained earlier
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last
    # Generate submission file
    python3 nucleus.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to serve as a validation set.
VAL_IMAGE_IDS = [
'37_3-WT-WT-SM_001',
'37_3-WT-WT-SM_001---Copy',
'42_4-WT-ZZ-RA_001 - Copy',
'42_4-WT-ZZ-RA_001 - Copy - Copy',
'42_4-WT-ZZ-RA_001---Copy',
'42_4-WT-ZZ-RA_001---Copy - Copy',
'43_1 KO ZZ RA_002 - Copy',
'43_1 KO ZZ RA_002 - Copy - Copy'

# '024435_2017-09-07 13_54_54_#5003#9025#2420#3188#635#1403#',
# '024435_2017-09-07 13_54_54_#5003#9025#2420#3188#6352#7120#',
# '024435_2017-09-07 13_54_54_#5003#9025#2420#3188#6987#7755#',
# '024435_2017-09-07 13_54_54_#5003#9025#2420#3188#7622#8390#',
# '024435_2017-09-07 13_54_54_#5003#9025#2420#3188#8257#9025#',
# '024435_2017-09-07 13_54_54_#5003#9025#3025#3793#0#768#',
# '024435_2017-09-07 13_54_54_#5003#9025#3025#3793#1270#2038#',
# '024435_2017-09-07 13_54_54_#5003#9025#3025#3793#1905#2673#',
# '024435_2017-09-07 13_54_54_#5003#9025#3025#3793#2541#3309#',
# '024435_2017-09-07 13_54_54_#5003#9025#3025#3793#3176#3944#',
# '024435_2017-09-07 13_54_54_#5003#9025#3025#3793#3811#4579#',
# '024435_2017-09-07 13_54_54_#5003#9025#3025#3793#4446#5214#',
# '024435_2017-09-07 13_54_54_#5003#9025#3025#3793#5081#5849#',
# '024435_2017-09-07 13_54_54_#5003#9025#3025#3793#5716#6484#',
# '024435_2017-09-07 13_54_54_#5003#9025#3025#3793#635#1403#',
# '024435_2017-09-07 13_54_54_#5003#9025#3025#3793#6352#7120#',
# '024435_2017-09-07 13_54_54_#5003#9025#3630#4398#0#768#',
# '024435_2017-09-07 13_54_54_#5003#9025#3630#4398#1270#2038#',
# '024435_2017-09-07 13_54_54_#5003#9025#3630#4398#1905#2673#',
# '024435_2017-09-07 13_54_54_#5003#9025#3630#4398#2541#3309#',
# '024435_2017-09-07 13_54_54_#5003#9025#3630#4398#3176#3944#',
# '024435_2017-09-07 13_54_54_#5003#9025#3630#4398#635#1403#',
# '024435_2017-09-07 13_54_54_#5003#9025#4235#5003#0#768#',
# '024435_2017-09-07 13_54_54_#5003#9025#4235#5003#1270#2038#',
# '024435_2017-09-07 13_54_54_#5003#9025#4235#5003#1905#2673#',
# '024435_2017-09-07 13_54_54_#5003#9025#4235#5003#635#1403#',
# '024435_2017-09-07 13_54_54_#5003#9025#605#1373#1270#2038#',
# '024435_2017-09-07 13_54_54_#5003#9025#605#1373#1905#2673#',
# '024435_2017-09-07 13_54_54_#5003#9025#605#1373#2541#3309#',
# '024435_2017-09-07 13_54_54_#5003#9025#605#1373#3176#3944#',
# '024435_2017-09-07 13_54_54_#5003#9025#605#1373#3811#4579#',
# '024435_2017-09-07 13_54_54_#5003#9025#605#1373#4446#5214#',
# '024435_2017-09-07 13_54_54_#5003#9025#605#1373#5081#5849#',
# '024435_2017-09-07 13_54_54_#5003#9025#605#1373#5716#6484#',
# '024435_2017-09-07 13_54_54_#5003#9025#605#1373#635#1403#',
# '024435_2017-09-07 13_54_54_#5003#9025#605#1373#6352#7120#',
# '024435_2017-09-07 13_54_54_#5003#9025#605#1373#6987#7755#',
# '024435_2017-09-07 13_54_54_#5003#9025#605#1373#7622#8390#',
# '024435_2017-09-07 13_54_54_#5003#9025#605#1373#8257#9025#'
]


############################################################
#  Configurations
############################################################

class NucleusConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "nucleus"
    # mask rcnn default 0.02 matterport default 0.001
    LEARNING_RATE=0.001
    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background + nucleus

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = (657 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    BACKBONE = "resnet50"
    # BACKBONE = "resnet101"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128) # already reduced

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 2000

    # threshold to determine fore/back-ground
    DETECTION_MASK_THRESHOLD=0.35

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training default 256
    # RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold. default 256
    # TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


class NucleusInferenceConfig(NucleusConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    # IMAGE_RESIZE_MODE = "crop" # pad still dividable by 64
    IMAGE_RESIZE_MODE = "pad64" # pad still dividable by 64
    # IMAGE_RESIZE_MODE = "none"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


############################################################
#  Dataset
############################################################

class NucleusDataset(utils.Dataset):
    def __init__(self, origin, targets):
        super(self.__class__,self).__init__()
        self.original=origin
        self.targets=targets

    def load_nucleus(self,subset_dir,subset):
        """Load a subset of the nuclei dataset.
        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        # self.add_class("nucleus", 1, "nucleus")
        for i, tgt in enumerate(self.targets.split(',')):
            self.add_class('nucleus', i+1, tgt)

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        # dataset_dir = os.path.join(dataset_dir, subset)
        if subset == "val":
            image_ids = VAL_IMAGE_IDS
        else:
            # Get image ids from directory names
            image_ids = next(os.walk(subset_dir))[1]
            image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))

        # Add images
        for image_id in image_ids:
            self.add_image(
                "nucleus",
                image_id=image_id,
                path=os.path.join(subset_dir,image_id,self.original,"{}.jpg".format(image_id)))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        class_ids=[]
        mask = None
        for i, tgt in enumerate(self.targets.split(',')):
            # Get mask directory from image path
            mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), tgt)
            # Read mask files from image
            for f in next(os.walk(mask_dir))[2]:
                if f.endswith(".jpg"):
                    m = skimage.io.imread(os.path.join(mask_dir, f),as_grey=True).astype(np.bool)[...,np.newaxis]
                    # print(np.shape(m))
                    mask=m if mask is None else np.concatenate((mask, m),axis=-1)
                    class_ids.append(i+1)
        # Handle occlusions
        # occlusion=np.logical_not(mask[:,:,-1]).astype(np.uint8)
        # for i in range(count-2,-1,-1):
        #     mask[:,:,i]=mask[:,:,i]*occlusion
        #     occlusion=np.logical_and(occlusion,np.logical_not(mask[:,:,i]))
        return mask, np.array(class_ids).astype(np.int32)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        # return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == 'nucleus':
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model,subset_dir,args):
    """Train the model."""
    # Training dataset.
    dataset_train = NucleusDataset(args.original,args.targets)
    dataset_train.load_nucleus(subset_dir,'train')
    dataset_train.prepare()

    # Validation dataset
    dataset_val = NucleusDataset(args.original,args.targets)
    dataset_val.load_nucleus(subset_dir,"val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=5, # 20
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10, # 40
                augmentation=augmentation,
                layers='all')

############################################################
#  Detection
############################################################

def detect(model,subset_dir,args):
    """Run detection on images in the given directory."""
    print("Running on {}".format(subset_dir))
    # res_dir=os.path.join(subset_dir,'result')
    res_dir=os.path.join(args.dataset,'result_'+args.subset)

    # Create directory
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(res_dir, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = NucleusDataset(args.original,args.targets)
    dataset.load_nucleus(subset_dir,'detect')
    dataset.prepare()
    # Load over images
    submission = []
    header='file,lym,mono,pmn' # dataset.class_names[1],dataset.class_names[2],dataset.class_names[3]
    submission.append(header); print(header)
    import collections
    for image_id in dataset.image_ids:
        image = dataset.load_image(image_id)
        r = model.detect([image], verbose=0)[0]
        source_id = dataset.image_info[image_id]["id"]
        countmap=collections.Counter(r['class_ids'])
        result=','.join([source_id,str(countmap.get(1,0)),str(countmap.get(2,0)),str(countmap.get(3,0))])
        submission.append(result); print(result)
        # Save image with masks
        visualize.display_instances(
                image, r['rois'], r['masks'], r['class_ids'],
                dataset.class_names, r['scores'],
                show_bbox=False, show_mask=False,
                title=' '.join(["Predictions",header,result]))
        plt.savefig("{}/{}.png".format(submit_dir, source_id))

    # Save to csv file
    submission = "ImageId,\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)


def move_file_structure(ext=".jpg"):
    for filename in os.listdir(subset_dir):
        if filename.endswith(ext):
            print(filename)
            filenoext=filename.replace(ext,'')
            os.makedirs(os.path.join(subset_dir,filenoext))
            os.makedirs(os.path.join(subset_dir,filenoext,'images'))
            os.rename(os.path.join(subset_dir,filename),os.path.join(subset_dir,filenoext,'images',filename))


############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Mask R-CNN for nuclei counting and segmentation')
    # parser.add_argument("command", metavar="<command>", help="'train' or 'detect'")
    parser.add_argument("--command",
                        # default='train',
                        default='detect',
                        metavar="<command>", help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False, default='D:/kaggle_nucleus',
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--subset', required=False, default='10xkyle', #'Original+LYM+MONO+PMN', # '40xpred', #'10xkyle',
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run training/prediction on")
    parser.add_argument('--original', required=False, default='images',
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run training/prediction on")
    parser.add_argument('--targets', required=False, default='LYM,MONO,PMN', #LYM,MONO,PMN
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run training/prediction on")
    parser.add_argument('--weights', required=False, default='last', # 'imagenet', #
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco' 'imagenet'")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    logs_dir=os.path.join(args.dataset,'logs')
    print("Logs: ", logs_dir)

    # Create model
    if args.command == "train":
        config= NucleusConfig()
        model = modellib.MaskRCNN(mode="training", config=config, model_dir= logs_dir)
    else:
        config=NucleusInferenceConfig()
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir= logs_dir)
    config.display()

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = 'D:\weights.h5'
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching number of classes
        model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    subset_dir=os.path.join(args.dataset,args.subset)
    if args.command == "train":
        train(model, subset_dir, args)
    elif args.command == "detect":
        move_file_structure('.jpg')
        detect(model, subset_dir, args)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))