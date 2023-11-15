#!/usr/bin/env python3

from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch import Tensor, stack
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from k3d.colormaps import matplotlib_color_maps
from matplotlib import pyplot
import plotly.express as px

import k3d

__all__ = ['plotly_red_custom_colorscale', 'k3d_red_custom_colorscale',
           'create_k3d_category_20_discrete_map',
           'check_tensor_shapes', 'show_poi', 'sum_point_attributes',
           'create_baseline_point_cloud', 'show_point_cloud',
           'show_point_cloud_classification_k3d', 'show_point_cloud_classification_plotly',
           'explain_plotly', 'explain_k3d']

"""
Custom Colorscales
"""

# Plotly Custom Colorscale for test
plotly_red_custom_colorscale = [[0, 'seashell'],        # Lightest color for smallest value
                                [0.25, 'lightcoral'],   # Lighter color for small values
                                [0.5, 'coral'],         # Med color for intermediate values
                                [0.75, 'sienna'],       # Darker color for high values
                                [1, 'crimson']]         # Darkest color for highest value


# K3D Custom Colorscale for test:
# These color scales structure describe a list with one or more of the following pattern:
# A number from the interval [0, 1], describing a part from the scale,
# followed by the rgb color value in that part of the scale

k3d_red_custom_colorscale = [0.0, 1.0, 0.63, 0.48]
k3d_red_custom_colorscale.extend([0.1666, 1.0, 0.63, 0.48])

k3d_red_custom_colorscale.extend([0.1667, 0.87, 0.50, 0.38])
k3d_red_custom_colorscale.extend([0.3333, 0.87, 0.50, 0.38])

k3d_red_custom_colorscale.extend([0.3334, 0.73, 0.38, 0.28])
k3d_red_custom_colorscale.extend([0.5, 0.73, 0.38, 0.28])

k3d_red_custom_colorscale.extend([0.5001, 0.61, 0.26, 0.19])
k3d_red_custom_colorscale.extend([0.6666, 0.61, 0.26, 0.19])

k3d_red_custom_colorscale.extend([0.6667, 0.48, 0.15, 0.10])
k3d_red_custom_colorscale.extend([0.8333, 0.48, 0.15, 0.10])

k3d_red_custom_colorscale.extend([0.8334, 0.36, 0.0, 0.0])
k3d_red_custom_colorscale.extend([1.0, 0.36, 0.0, 0.0])


def create_k3d_category_20_discrete_map():
    """
    Creates a similar color scale to the discrete category 20, presented in matplotlib.
    Unfortunately, K3D doesn't support discrete color scales.

    Returns:
        A color scale to be used in some plots with numerical data.
    """

    N = 20
    colormap = pyplot.get_cmap(name='tab20', lut=None)
    colormap.colors
    intervals = np.linspace(0, 1, N + 1)

    colorscale = []

    for i, color in enumerate(colormap.colors):
        colorscale.extend([intervals[i], color[0], color[1], color[2]])
        colorscale.extend([intervals[i + 1] + 0.00000000000000001, color[0], color[1], color[2]])

    return colorscale


def check_tensor_shapes(tensors) -> bool:
    """Checks if the tensors inside a iterable object, like a list or tuple,
    have the same shape

    Args:
        tensors: Iterable object containing tensors.

    Returns:
        bool: True if tensors have the same shape, False otherwise.
    """

    reference_shape = tensors[0].shape

    for tensor in tensors[1:]:
        if tensor.shape != reference_shape:
            return False

    return True


def show_poi(poi_index: int, coords: Tensor) -> None:
    """
    Shows the point cloud with the point of interest in evidence

    Args:
        poi_index (int): Index of the point of interest
        coords (Tensor): Coordinates of each point.
    """

    np_coords = coords.detach().cpu().numpy()
    num_points = np_coords.shape[0]
    colors = np.ones(num_points)
    colors[poi_index] = 0.0

    fig = k3d.plot(grid_visible=False)

    fig += k3d.points(positions=np_coords,
                      shader='3d',
                      color_map=matplotlib_color_maps.Coolwarm,
                      attribute=colors,
                      color_range=[0.0, 1.0],
                      point_sizes=[0.03 if color == 1.0 else 0.08 for color in colors],
                      name="Point of interest")
    fig.display()


def sum_point_attributes(attributes: TensorOrTupleOfTensorsGeneric,
                         target_dim: int = -1) -> Tensor:
    """
    Performs an element-wise summation over the attributes, followed by a sum of the
    elements in the target dimension in the resulted tensor.

    It's useful to aggregate attribution for any kind of point features.

    Args:
        attributes (TensorOrTupleOfTensorsGeneric): Tuple of tensors that describes
        each point attributes. The tensors must have the same shape.
        target_dim (int, optional): Target dimension where the last sum will occour.
        Defaults to -1, the last dimension.

    Returns:
        Tensor: Sum of all tensors element-wise and with the last dimension added.
    """

    # TODO: Check if this is useless, since the type in the parameters probably already
    # guarantees that 'attributes' is a tuple of tensors.
    assert isinstance(attributes, tuple), \
        "Parameter 'attributes' must be a tuple of tensors."

    assert len(attributes) > 0, \
        "'attributes' tuple must contain at least one tensor."

    assert check_tensor_shapes(attributes), \
        "Attributes must have the same shape."

    assert len(attributes[0].shape) > 1, \
        "Attributes shapes must have at least two dimensions."

    return (stack(attributes).sum(dim=0)).sum(axis=target_dim)


def create_baseline_point_cloud(input_coords: Tensor) -> Tuple[Tensor]:
    """
    Baseline based on a cubic uniform point distribution.
    The colors returned will be all black.
    The volume used uses the same min and max bounds of the coordinates.
    If we can't perfectly divide the number of points in the rectangular volume,
    we add the remaining points randomly trough the volume.

    Args:
        input_coords (Tensor): Coordinates of the input in a size (N, 3).

    Returns:
        tuple(Tensor): Tuple containing the coordinates of the baseline points
        coordinates and its colors.
    """

    # Retrieve the maximum and minimum bounds
    max_values, _ = torch.max(input_coords, dim=0)
    min_values, _ = torch.min(input_coords, dim=0)

    x_min, y_min, z_min = min_values.tolist()
    x_max, y_max, z_max = max_values.tolist()

    # Retrive the number of points in input
    n_points = input_coords.size()[0]

    # Define colors as 0
    baseline_colors = torch.zeros(n_points, 3, requires_grad=True)

    # Defining grids for the volume
    grid_size = int(round(n_points ** (1 / 3.0)))

    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    z_grid = np.linspace(z_min, z_max, grid_size)

    x_points, y_points, z_points = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

    points_np = np.vstack((x_points.flatten(), y_points.flatten(), z_points.flatten())).T

    num_grid_points = points_np.shape[0]

    # Generate additional random points if necessary
    if num_grid_points < n_points:
        remaining_points = n_points - num_grid_points
        random_points = np.random.uniform(low=[x_min, y_min, z_min],
                                          high=[x_max, y_max, z_max],
                                          size=(remaining_points, 3))
        points_np = np.concatenate((points_np, random_points), axis=0)

    baseline_coords = torch.tensor(points_np, requires_grad=True)

    return (baseline_coords, baseline_colors)


def show_point_cloud(coords: Tensor, colors: Tensor, size: float = 0.1) -> None:
    """
    Plots the Point Cloud using K3D plot library.

    Args:
        coords (Tensor): Coordinates of the points, in the shape (N, 3)
        colors (Tensor): Colors of the points, in the shape (N, 3). It assumes that the colors
        are in the RGB format with interval [0, 255]
        size (float, optional): Points size in the plot. Defaults to 0.1.
    """

    rgb = colors.cpu().detach().numpy().astype(np.uint32)
    colors_hex = (rgb[:, 0] << 16) + (rgb[:, 1] << 8) + (rgb[:, 2])
    np_coords = coords.cpu().detach().numpy()

    plot = k3d.plot(grid_visible=False)
    plot += k3d.points(np_coords, colors_hex, point_size=size, shader="simple", name="Point Cloud")
    plot.display()


def show_point_cloud_classification_k3d(coords: Tensor, classifications: Tensor,
                                        size: float = 0.1) -> None:
    """
    Plots the classfication of each point using K3D.

    It's not capable to hold extra information, but it has a better performance than
    show_point_cloud_classification_plotly.
    Recomended to use when plotly's performance spoils the 3D interaction.

    Args:
        coords (Tensor): Coordinates of the points, in the shape (N, 3)
    """

    np_coords = coords.cpu().detach().numpy()
    np_class = classifications.cpu().detach().numpy()

    plot = k3d.plot(grid_visible=False)
    plot += k3d.points(np_coords,
                       shader='flat',
                       attribute=np_class,
                       point_size=size,
                       color_map=create_k3d_category_20_discrete_map(),
                       color_range=[np.min(np_class), np.max(np_class)],
                       name="Point Classifications")
    plot.display()


def show_point_cloud_classification_plotly(coords: Tensor, classifications: Tensor,
                                           instance_labels: Tensor = None,
                                           classes_dict: dict = None, size: float = 0.5) -> None:
    """
    Plots the classficiation of each point using Plotly.
    It can hold extra information such as instance labels and predictions meanings.

    Args:
        coords (Tensor): Coordinates of the points, in the shape (N, 3)
        classifications (Tensor): The predictions indices for each point
        instance_labels (Tensor, optional): Object instance labels for each point.
        The order of the points must be the same as the coordinates and classifications.
        Defaults to None.
        classes_dict (dict, optional): Dictionary containing the meaning of each prediction index.
        Defaults to None.
        size (float, optional): Points sizes in the plot. Defaults to 0.1.
    """
    size
    np_coords = coords.cpu().detach().numpy()
    np_class = classifications.cpu().detach().numpy().astype(np.int)

    if (instance_labels is not None):
        instance_labels.cpu().detach().numpy().astype(np.int)

    hover = dict(X=np_coords[:, 0],
                 Y=np_coords[:, 1],
                 Z=np_coords[:, 2],
                 Class=[classes_dict[class_num] for class_num in np_class],
                 Instance_Index=instance_labels,
                 Point_Num=[i for i in range(len(instance_labels))])

    hover_df = pd.DataFrame(hover)

    fig = px.scatter_3d(data_frame=hover_df,
                        x="X",
                        y="Y",
                        z="Z",
                        color="Class",
                        opacity=1.0,
                        hover_data=["Class", "Instance_Index", "Point_Num"])

    # Change size directly through here to avoid white outlines in the points
    for data in fig.data:
        data['marker']['size'] = size
    # The size change must not modify the legend
    fig.update_layout(legend= {'itemsizing': 'constant'})

    fig.show()


def explain_plotly(attributes: Tensor, coords: Tensor,
                   template_name: str = 'simple_white') -> None:
    """
    Plots the point cloud with its attributes values using Plotly.
    Useful for an thorough analysis of the points attribute values, but it has a poor interaction
    and scene understanding thanks to Plotly's point rendering.

    Args:
        attributes (TensorOrTupleOfTensorsGeneric): Attributes for each point.
        coords (Tensor): Coordinates of each point.
        template_name (str, optional): The template style to be used for the plot.
        Defaults to 'simple_white'.
    """

    np_coords = coords.detach().cpu().numpy()

    np_attr = attributes.detach().cpu().numpy()
    # normalized_attr = (np_attr - np.min(np_attr)) / (np.max(np_attr) - np.min(np_attr))

    fig = px.scatter_3d(x=np_coords[:, 0], y=np_coords[:, 1], z=np_coords[:, 2],
                        color=np_attr,
                        opacity=1.0,
                        range_color=[np.min(np_attr), np.max(np_attr)],
                        template='simple_white')

    fig.show()


def explain_k3d(attributes: Tensor, coords: Tensor, attribute_name=None) -> None:
    """
    Plots the point cloud with its attributes values using K3D.
    It doesn't offer the exactly value of the attributes but its performance and scene
    understanding are better than Plotly's explanation.

    Args:
        attributes (TensorOrTupleOfTensorsGeneric): Attributes for each point.
        coords (Tensor): Coordinates of each point.
        attribute_name (str, optional): Name of the point data in the plot. Defaults to None.
    """

    if attribute_name is None:
        attribute_name = "Attributes"

    fig = k3d.plot(grid_visible=False)

    min_size = 0.07
    max_size = 0.1

    np_attr = attributes.detach().cpu().numpy()

    sizes = (np_attr - np.min(np_attr)) / (np.max(np_attr) - np.min(np_attr)) * \
            (max_size - min_size) + min_size

    fig += k3d.points(positions=coords,
                      shader='3d',
                      color_map=k3d.paraview_color_maps.Viridis_matplotlib,
                      attribute=np_attr,
                      color_range=[np.min(np_attr), np.max(np_attr)],
                      point_sizes=sizes,
                      name=attribute_name)
    fig.display()
