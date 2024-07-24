"""
Facial Emotions Recognition System
Author: Wajahat Ullah
This file can be used for inference using the pretrained model, which
incorporated CLIP encoder to extract features and then classifies facial
expression using a fully connected layer. This model achieved about 65%
accuracy on the test data.
"""
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from model import FERModel


PRETRAINED_MODEL = 'models/fer2013_model.pth'


def load_model(model_path, num_classes=7):
    """This model loads the custom defined FERModel.

    Args:
        model_path (str): path to the pretrained model
        num_classes (int, optional): number of classes in the dataset.
        defaults to 7.

    Returns:
        FERModel: Custom FERModel
    """
    model = FERModel(num_classes)
    state_dict = torch.load(model_path,
                            map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image):
    """Preprocess an image so that it can be used by the FERModel for inference

    Args:
        image (ndarray): a grayscale image

    Returns:
        tensor: preprocessed image
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return transform(image).unsqueeze(0)


def main():
    """Program entry point
    """
    model = load_model(PRETRAINED_MODEL)

    # We will detect faces using Haarcascades
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Class lables
    emotions = ['angry', 'disgust', 'fear',
                'happy', 'sad', 'surprise', 'neutral']

    # Inference on video feed
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 10)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_tensor = preprocess_image(face)

            with torch.no_grad():
                output = model(face_tensor)
                _, predicted = torch.max(output, 1)
                emotion = emotions[predicted.item()]

            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow('Facial Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
