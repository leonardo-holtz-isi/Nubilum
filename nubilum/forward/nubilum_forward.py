#!/usr/bin/env python3

import torch
from torch import Tensor


class InstanceWrappedModel(torch.nn.Module):
    """
    This model wrapper is dedicated to be used in instance segmentation models.
    The forward function receives as additional forward arguments the ground truth
    instances labels and one specific instance label.
    Any attribute function executed using this wrapped model will calculate attributes over
    the points that have its ground truth instance label equal to the label of interest.
    The idea is to understand how the model classify the individual objects in the scene,
    checking which points contribute with the result and how the group of points of the
    instance of interest are influenced by other points from other objects in the scene.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self,
                point_coords: Tensor,
                point_colors: Tensor,
                instance_labels: Tensor,
                label_of_interest: int,
                output_scores_key_name: str = None) -> Tensor:
        """
        Calculates only the logits for the points whose instance labels is
        the label of interest. It returns a tensor with the logits of the
        instance points for each class.
        Each index of all tensors must identify the exact same point.

        Args:
            point_coords (Tensor): Model's Input Feature. Tensor containing the points coordinates.
            point_colors (Tensor): Model's Input Feature. Tensor containing the points colors.
            instance_labels (Tensor): Additional forward argument. Tensor containing the GT points
            instance labels.
            label_of_interest (int): Additional forward argument. The instance label to be verified.
            output_scores_key_name (str, optional): Additional forward argument. Indicates the key
            for the semantic predictions scores if the output is a dict. Defaults to None if the
            Output is already a tensor with the semantic predictions.

        Returns:
            Tensor: A tensor with the logits of the instance points for each class.
        """

        output = self.model(point_coords, point_colors)
        instance_mask = instance_labels == label_of_interest

        if output_scores_key_name is not None:
            scores = output[output_scores_key_name]
        else:
            scores = output

        instance_scores = scores[instance_mask]
        return torch.sum(instance_scores, dim=0).reshape(1, -1)


class PointWrappedModel(torch.nn.Module):
    """
    Model wrapper dedicated to find attributes related to a unique point.
    It permits to understands what other points influence in the classification
    of a point of interest.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, point_coords: Tensor, point_colors: Tensor,
                point_of_interest: int, output_scores_key_name: str = None) -> Tensor:
        """
        Calculates the logits of only one point, for each class.

        Args:
            point_coords (Tensor): Model's Input Feature. Tensor containing the points coordinates.
            point_colors (Tensor): Model's Input Feature. Tensor containing the points colors.
            point_of_interest (int): Additional forward argument. The point to be verified.
            output_scores_key_name (str, optional): Additional forward argument. Indicates the key
            for the semantic predictions scores if the output is a dict.
            Defaults to None if the Output is already a tensor with the semantic predictions.

        Returns:
            Tensor: A tensor with the logits of the point of interest for each class.
        """
        output = self.model(point_coords, point_colors)

        if output_scores_key_name is not None:
            scores = output[output_scores_key_name]
        else:
            scores = output

        return scores[point_of_interest].reshape(1, -1)


class SummarizedWrappedModel(torch.nn.Module):
    """
    Model wrapper dedicated to understand the prediction of the entire point cloud.
    Since attributions methods are made for classification tasks and not segmentation
    tasks, the forward function of this model wrapper tries to summarize the
    classifications of all points by summing the logit of the final prediction of each
    point. For more details, check
    https://captum.ai/tutorials/Segmentation_Interpret#Interpreting-with-Captum
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, point_coords: Tensor, point_colors: Tensor,
                output_scores_key_name: str = None) -> Tensor:
        """
        Sums the logits of each class. The logit of the ith class of a point j is
        only used in the sum if, and only if, the class i was the class predicted
        for the point j.

        Args:
            point_coords (Tensor): Model's Input Feature. Tensor containing the points coordinates.
            point_colors (Tensor): Model's Input Feature. Tensor containing the points colors.
            output_scores_key_name (str, optional): Additional forward argument. Indicates the key
            for the semantic predictions scores if the output is a dict.
            Defaults to None if the Output is already a tensor with the semantic predictions.

        Returns:
            Tensor: A tensor with a sum of all predicted logits for each class.
        """
        output = self.model(point_coords, point_colors)

        if output_scores_key_name is not None:
            scores = output[output_scores_key_name]
        else:
            scores = output

        num_points = scores.size(0)
        num_classes = scores.size(1)
        max_out = torch.max(scores, 1)
        argmax_classes = max_out.indices    # Class index for each point

        mask = torch.zeros(num_points, num_classes).to("cuda:0")
        mask[torch.arange(num_points), argmax_classes] = 1  # Boolean mask

        classification_scores = scores * mask   # contains only the scores of the predicted classes
        score_classes_sum = classification_scores.sum(dim=0)

        return score_classes_sum.reshape(1, -1)
