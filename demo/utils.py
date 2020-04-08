import os
import sys
import time

import cv2
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms, models

# Used to import SIMPLECNNModel from a directory at the same leave of the demo one.
PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from models.SimpleCNNModel import SimpleCNNModel


def extract_roi(frame, mask, bbox):
    """ Extract a Region of Interest (ROI) from a frame and remove its background

    Args:
        frame: Full RGB image
        mask: Mask containing the segmentation of the object. Used to remove the background.
        bbox([x, y, w, h]): (x, y) = top left corner of the bbox, w = width of the bbox and h = height of the bbox.

    Returns:
        roi: RGB image of the object in the ROI with its background removed (black)
    """
    # Get bbox dimensions
    x, y, w, h = bbox

    # Crop the mask according to the ROI
    roi_mask = mask[y : y + h, x : x + w]
    # Create the inverse of the roi_mask
    roi_mask_inv = cv2.bitwise_not(roi_mask)

    # Crop the ROI from the frame
    roi_frame = frame[y : y + h, x : x + w, ::]
    # Remove the background in the frame
    roi_frame = cv2.bitwise_and(roi_frame, roi_frame, mask=roi_mask)

    # Create the background (color: black)
    roi_background = np.ones(roi_frame.shape, dtype=np.uint8) * 0
    #  Remove the object from the background
    roi_background = cv2.bitwise_and(roi_background, roi_background, mask=roi_mask_inv)

    # Combine the frame and the background
    roi = cv2.add(roi_background, roi_frame)

    return roi


class SimpleObjectSegmentation:
    """ Simple Object Segmentation Technique

    The segmentation technique is rather simple.
    It compares the current frame with a reference 
    frame and extract the biggest area with some differences.

    """

    def __init__(self, reference):
        """ Initialize the SimpleObjectSegmentation
        
        Args:
            reference: RGB image to use as the reference image
        """
        self.update_reference(reference)

    def update_reference(self, reference):
        """ Update the reference frame
        
        Args:
            reference: RGB image to use as the new reference image.
        """
        self.reference = reference.copy()

    def detect_object(self, frame, min_contourArea: int = 2500):
        """ Detect the biggest object
        Detect the biggest object in the current frame compared to the reference frame.

        Args:
            frame: RGB image of the current frame
            min_contourArea (int: 2500): minimum area the biggest contour should be to be detected

        Returns:
            mask/out: if the detection is successful returns the mask of the detection. Otherwise, returns the binary image used to make the detection for debugging purposes.
            bbox/None: if the detection is successful returns the bbox around the detection ((x, y) = top left corner of the bbox, w = width of the bbox and h = height of the bbox). Otherwise, returns None.
        """
        # Absolute difference between the background image and the current frame
        out = cv2.absdiff(self.reference, frame)

        # Thresholding of the difference
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = cv2.GaussianBlur(out, (5, 5), 0)
        out = cv2.threshold(out, 50, 255, cv2.THRESH_BINARY)[1]

        # Remove small detections and fill the holes in the bigger ones
        out = cv2.dilate(out, None, iterations=1)
        out = cv2.erode(out, None, iterations=1)
        out = cv2.dilate(out, None, iterations=2)

        # Extract the contours
        contours, hierarchy = cv2.findContours(
            out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Extract the biggest contour
        if contours:
            biggest_contour = max(contours, key=cv2.contourArea)

            #  Check if the area of the detection is big enough
            if cv2.contourArea(biggest_contour) >= min_contourArea:
                # Extract the bounding box around the detection
                x, y, w, h = cv2.boundingRect(biggest_contour)

                # Create the mask of the detection based on its contour
                mask = np.full((frame.shape[0], frame.shape[1]), 0, dtype=np.uint8)
                cv2.fillPoly(mask, pts=[biggest_contour], color=(255, 255, 255))

                return mask, [x, y, w, h]

        # Return None if no contour was detected or the contour was too small
        return out, None


class FashionClassifier:
    """ Wrapper for classification models trained on the Fashion-MNIST dataset.

    """

    def __init__(self, model_name: str, weights_path: str, classes: dict):
        """ Initialise the classifier
        
        Args:
            model (str): name of the model to load
            weights_path (str): filepath to the weights file for the model
            classes (dict): dictionary containing all the classes (e.g. {"0": "label_0", "1": "label_1",...})

        """
        # --- Model ---
        self.model_name = model_name
        self.weights_path = weights_path
        self.use_cuda = torch.cuda.is_available()
        self.classes = classes

        # Load the model
        self.load_model()

        # To compute the probabilities
        self.softmax = nn.Softmax(dim=1)

        self.input_size = 28

        # Create the preprocessing transformations
        list_inference_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]

        # In case the image needs 3 channels to be inputed in the model
        if self.num_input_channels > 1:
            duplicate_channel = transforms.Lambda(
                lambda x: x.repeat(self.num_input_channels, 1, 1)
            )
            list_inference_transforms.append(duplicate_channel)

        self.inference_transforms = transforms.Compose(list_inference_transforms)

    def load_model(self):
        """ Load the model according to some parameters

        """
        print(
            f"Loading {self.model_name} model with the weights from '{self.weights_path}'..."
        )

        # --- MODEL ---
        if self.model_name == "SimpleCNNModel":
            self.model = SimpleCNNModel()
            self.num_input_channels = 1

        elif self.model_name == "ResNet18":
            self.model = models.resnet18(pretrained=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, len(self.classes))
            self.num_input_channels = 3

        # Load the checkpoint containing the weights of the model
        if self.use_cuda:
            torch.cuda.benchmark = True
            self.model = self.model.cuda()
            checkpoint = torch.load(self.weights_path)
        else:
            checkpoint = torch.load(self.weights_path, map_location=torch.device("cpu"))

        # Load the trained weights in the model
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Get the best accuracy from the checkpoint
        best_accuracy = checkpoint["best_accuracy"]

        print(
            f"{self.model_name} model loaded successfully. Its accuracy on the Fashion-MNIST test dataset is {100 * best_accuracy:05.3f}."
        )

    def preprocess_input(self, image):
        """ Preprocess the image using opencv transformations only

        Args:
            image: RGB image
        
        Returns:
            input_image: grayscale image
        """
        # Resize the image for the input size
        (image_height, image_width) = image.shape[:2]

        ratio = float(self.input_size) / max([image_height, image_width])

        input_height = int(image_height * ratio)
        input_width = int(image_width * ratio)

        input_image = cv2.resize(image, (input_width, input_height))

        # Pad the image in a square with black borders
        delta_width = self.input_size - input_width
        delta_height = self.input_size - input_height
        top, bottom = delta_height // 2, delta_height - (delta_height // 2)
        left, right = delta_width // 2, delta_width - (delta_width // 2)

        color = [0, 0, 0]
        input_image = cv2.copyMakeBorder(
            input_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )

        # Convert the image to grayscale
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        return input_image

    def inference(self, image):
        """ Inference

        Classify the image using the model

        Args:
            image: RGB image to input in the model

        Returns:
             input_image: output of preprocess_input()
             confidence (list): Ordered list of the confidence of the prediction (Bigger first)
             prediction_name (list): Ordered list of the classes name (One with the biggest confidence first)
             inference_time (float): time used for the inference in seconds
        """
        # Preprocess the image
        input_image = self.preprocess_input(image)

        with torch.no_grad():
            # Use the custom transformer to convert the input to a tensor
            input_tensor = self.inference_transforms(input_image)[None, :]

            # Put the input image on the GPU is available
            if self.use_cuda:
                input_tensor = input_tensor.cuda()

            # Start the timer for calculation the inference time
            start_time = time.time()

            # Inference
            out = self.model(input_tensor)

            # Use softmax to convert the output to probability and ordered the results
            confidence, prediction = self.softmax(out).topk(dim=1, k=10)

            # Stop the inference timer
            inference_time = time.time() - start_time

            # Convert the confidence and prediction tensors to lists
            confidence = confidence[0, :].tolist()
            prediction = prediction[0, :].tolist()

            # Convert the classes id into their name
            prediction_name = [self.classes[str(x)] for x in prediction]

        return input_image, confidence, prediction_name, inference_time
