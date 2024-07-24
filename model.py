import torch
from torch import nn
from transformers import CLIPProcessor, CLIPModel


class FERModel(nn.Module):
    """
    A Facial Emotion Recognition Model which combines OpenAI CLIP encoder
    to extract features with a Dense layer for classification
    """
    def __init__(self, num_classes, device='cpu'):
        """Model constructor

        Args:
            num_classes (int): number of classes
            device (str, optional): device to use for inference. Defaults to 'cpu'.
        """
        super(FERModel, self).__init__()
        self.clip_model = (CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32").vision_model).to(device)
        self.classifier = (nn.Linear(768, num_classes)).to(
            device)  # CLIP's output dimension is 768
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """Performs a forward pass

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        with torch.no_grad():
            features = self.clip_model(
                x).last_hidden_state[:, 0, :]  # Use CLS token
        output = self.classifier(features)
        return output
