import argparse
import json
import os
import statistics
import time

import cv2
import numpy as np

from utils import SimpleObjectSegmentation, FashionClassifier, extract_roi

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=["SimpleCNNModel", "ResNet18"],
    help="Name of the model to train.",
)
parser.add_argument(
    "--weights_path",
    type=str,
    required=True,
    help="Path of the weights to load for the model.",
)
parser.add_argument(
    "--device",
    type=int,
    default="0",
    help="ID of the device to use for the realtime video capture.",
)
parser.add_argument(
    "--display_input",
    help="Display the input of the model in the top left corner.",
    action="store_true",
)
parser.add_argument(
    "--display_fps",
    help="Display the FPS in the top right corner.",
    action="store_true",
)
parser.add_argument(
    "--path_classes",
    default=os.path.join("..", "models", "classes.json"),
    type=str,
    help="Path to the json containing the classes of the Fashion MNIST dataset.",
)
args = parser.parse_args()

if __name__ == "__main__":

    # Print instructions
    print(f"------------- FASHION CLASSIFIER -------------")
    print(f" -> Press 'q' to quit")
    print(
        f" -> Press 'n' to reinitialize the background image for the simple object segmentation algorithm."
    )

    # Import the classes
    with open(args.path_classes) as json_file:
        classes = json.load(json_file)

    # Color use to draw and write
    color = (0, 255, 0)

    # list of the inference time
    list_inference_time = []

    # create the webcam device
    webcam = cv2.VideoCapture(args.device)

    # Get the first frame as reference for the object segmentation algorithm
    if webcam.isOpened():
        _, first_frame = webcam.read()

    # Instanciate the class responsible for the object detection and segmentation
    detector = SimpleObjectSegmentation(first_frame)

    # Instanciate the class responsible for the classifying the fashion objects
    fashion_classifier = FashionClassifier(
        model_name=args.model, weights_path=args.weights_path, classes=classes
    )

    # Initialize the flag to update the reference frame of the detector
    update_detector_reference = False

    # Start the capturing and processing loop
    while True:
        # Start the timer for the fps computation
        start_time = time.time()

        # Capture frame-by-frame
        _, frame = webcam.read()

        # Update the reference frame is the flag is set
        if update_detector_reference:
            detector.update_reference(frame)
            update_detector_reference = False
            print("Updated the reference image of the detector.")

        # Use the detector to grab the mask and the bounding-box of the biggest object in the frame
        mask, bbox = detector.detect_object(frame)

        #  If an object was found
        if bbox is not None:
            # --- EXTRACTION ---
            # Extract the roi around the object and remove its background using the mask
            roi = extract_roi(frame, mask, bbox)

            # --- CLASSIFICATION ---
            (
                input_image,
                confidence,
                prediction,
                processing_time,
            ) = fashion_classifier.inference(roi)

            # Add the inference time to the list of inference time
            list_inference_time.append(processing_time)

            # --- DISPLAY ---
            x, y, w, h = bbox
            # Draw the bounding box around the ROI
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Write the confidence and predicted class on top of the bounding box
            cv2.putText(
                frame,
                f"{prediction[0]} {confidence[0]:.2f}",
                (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            # Display the input image in the top left corner
            if args.display_input:
                # Resize the input image
                scale = 3
                (input_height, input_width) = input_image.shape[:2]
                input_image_to_display = cv2.resize(
                    input_image, (input_width * scale, input_height * scale)
                )

                # Convert the input image to 3 channels
                input_image_rgb = cv2.cvtColor(
                    input_image_to_display, cv2.COLOR_GRAY2BGR
                )

                # Display the input image on top of the current frame
                frame[
                    : input_height * scale, : input_width * scale, :
                ] = input_image_rgb

        # Display the FPS counter
        if args.display_fps:
            fps = 1 / (time.time() - start_time)
            cv2.putText(
                frame,
                f"{fps:05.2f} FPS",
                (frame.shape[1] - 90, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # Show the current frame with the other information
        cv2.imshow("Fashion Classifier", frame)

        # Wait for the different keys
        c = cv2.waitKey(1)

        if c == ord("q"):
            print(
                f"Average inference time for the {args.model} model over {len(list_inference_time)} inferences is {statistics.mean(list_inference_time)*1000:.3f}ms."
            )
            break
        elif c == ord("n"):
            update_detector_reference = True

    # When everything done, release the webcam and destroy the window
    webcam.release()
    cv2.destroyAllWindows()
