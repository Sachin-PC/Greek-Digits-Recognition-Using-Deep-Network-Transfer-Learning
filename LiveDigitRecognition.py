# Sachin Palahalli Chandrakumar
#  Code to implement live digits recognition from the model trained to recognize digits

from MyNetwork import MyNetwork

import cv2
import torch
from torchvision import datasets, transforms
import numpy as np


def transform_frame(frame, transform):
    """
        Method to tranfer frame to the requirements
        Parameters:
        frame : input frame
        transform : transform object
    """
    # greyscale_image = frame.convert("L")
    # inverted_image = Image.eval(greyscale_image, lambda x: 255 - x)
    # resized_image = TF.resize(inverted_image, (28, 28))
    greyscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inverted_image = 255 - greyscale_image
    resized_image = cv2.resize(inverted_image, (28, 28))
    image_tensor = transform(resized_image).unsqueeze(
        0)
    return image_tensor


def live_recognition(model, videoPath):
    """
        Method to implement live recognition of digits
        Parameters:
        model : model
        videoPath : video path
    """
    cap = cv2.VideoCapture(videoPath)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    while True:
        ret, frame = cap.read()
        print("ret = ",ret)
        if not ret:
            break
        frame_tensor = transform_frame(frame, transform)
        with torch.no_grad():
            predicted_output = model(frame_tensor)
            predicted_label = np.argmax(predicted_output)
            cv2.putText(frame, f"Predicted Label: {predicted_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 255, 0), 2)
            cv2.imshow('Live Digit Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# main method to implement live recognition
if __name__ == "__main__":
    test_batch_size = 1000
    model = MyNetwork()
    model_path = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth " \
                 "Semester/PRCV/Projects/5/Project5/models/mainModel/model.pth"
    data_path = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth " \
                "Semester/PRCV/Projects/5/utils/data/"
    model.load_state_dict(torch.load(model_path))

    videoPath = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth " \
                "Semester/PRCV/Projects/5/Project5/Video/d2.mp4"
    live_recognition(model, videoPath)

    print(model)
