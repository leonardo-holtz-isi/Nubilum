#!/usr/bin/env python3

# TODO: For now, the functions are mantaining an "look before you leap" (LBYL) pattern,
# over the "it's easier to ask for forgiveness than permission" (EAFP) pattern.
# This will be updated in a later version.

from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch import Tensor, stack
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from k3d.colormaps import matplotlib_color_maps
from matplotlib import pyplot
import plotly.express as px
import colorcet as cc
import math

import k3d

__all__ = ['plotly_red_custom_colorscale', 'k3d_red_custom_colorscale',
           'create_k3d_category_20_discrete_map',
           'check_tensor_shapes', 'show_poi', 'sum_point_attributions',
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
        `colorscale` (list): A color scale to be used in some plots with numerical data.
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
    """
    Checks if the tensors inside a iterable object have the same shape

    Args:
        `tensors`: Iterable object containing tensors.

    Returns:
        `tensor_shapes_ok` (bool): True if tensors have the same shape, False otherwise.
    """

    reference_shape = tensors[0].shape

    for tensor in tensors[1:]:
        if tensor.shape != reference_shape:
            return False

    return True


def show_poi(poi_index: int, np_coords: np.array) -> None:
    """
    Shows the point cloud with the point of interest in evidence

    Args:
        `poi_index` (int): Index of the point of interest.
        
        `np_coords` (numpy.array): Coordinates of each point.
    """

    # Types and shapes verifications
    if not isinstance(np_coords, np.array):
        raise TypeError("The points coordinates must be in numpy array format.")
    if np_coords.shape[1] != 3:
        raise ValueError("The points coordinates must in shape (X,3), where the three values \
            correspond to the x, y and z coordinates of each point.")
    if not isinstance(poi_index, int):
        raise TypeError("Point of interest must be of type 'int'")

    num_points = np_coords.shape[0]
    colors = np.ones(num_points)
    colors[poi_index] = 0.0

    fig = k3d.plot(grid_visible=False)

    fig += k3d.points(positions=np_coords,
                      shader='3d',
                      color_map=matplotlib_color_maps.Coolwarm,
                      attribution=colors,
                      color_range=[0.0, 1.0],
                      point_sizes=[0.03 if color == 1.0 else 0.08 for color in colors],
                      name="Point of interest")
    fig.display()


def sum_point_attributions(attributions: TensorOrTupleOfTensorsGeneric,
                         target_dim: int = -1) -> Tensor:
    """
    Performs an element-wise summation over the attributions, followed by a sum of the
    elements in the target dimension in the resulted tensor.

    It's useful to aggregate attribution for any kind of point features.

    Args:
        `attributions` (TensorOrTupleOfTensorsGeneric): Tuple of tensors that describes
        each point attributions. The tensors must have the same shape.

        `target_dim` (int, optional): Target dimension where the last sum will occour.
        Defaults to -1, the last dimension.

    Returns:
        `attributions_sum` (Tensor): Sum of all tensors element-wise and with the last dimension added.
    """

    if not isinstance(attributions, TensorOrTupleOfTensorsGeneric):
        raise TypeError("Parameter 'attributions' must be a tuple of tensors.")

    if len(attributions) == 0:
        raise ValueError("'attributions' tuple must contain at least one tensor.")

    if not check_tensor_shapes(attributions):
        raise ValueError("attributions must have the same shape.")

    if len(attributions[0].shape) == 0:
        raise ValueError("attributions shapes must have at least two dimensions.")

    return (stack(attributions).sum(dim=0)).sum(axis=target_dim)


def create_baseline_point_cloud(input_coords: Tensor) -> Tuple[Tensor]:
    """
    Baseline based on a cubic uniform point distribution.
    The colors returned will be all black.
    The volume used uses the same min and max bounds of the coordinates.
    If we can't perfectly divide the number of points in the rectangular volume,
    we add the remaining points randomly trough the volume.

    Args:
        `input_coords` (Tensor): Coordinates of the input in a size (N, 3).

    Returns:
        `baseline` (Tensor): Tuple containing the coordinates of the baseline points
        coordinates and its colors.
    """

    # Types and shapes verifications
    if not isinstance(input_coords, Tensor):
        raise TypeError("The points coordinates must be in Tensor format.")
    if input_coords.shape[1] != 3:
        raise ValueError("The points coordinates must in shape (X,3), where the three values \
            correspond to the x, y and z coordinates of each point.")

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
    grid_size = int(math.floor(n_points ** (1 / 3.0)))

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

    baseline_coords = torch.tensor(points_np, dtype=torch.float32, requires_grad=True)

    return (baseline_coords, baseline_colors)


def log_scale_attributions(attributions: np.array, signed: bool = False) -> np.array:
    """
    Apply a logarithmic scale over attributions.

    Args:
        `attributions` (numpy.array): Attribution values.

        `signed` (bool, optional): Mantains the attributions signs after the scale if True.
        Defaults to False.

    Returns:
        `log_attributions` (numpy.array): Scaled attributions
    """

    if not isinstance(attributions, np.array):
        raise TypeError("'attributions' must be in numpy array format")
    if not isinstance(signed, bool):
        raise TypeError("'attributions' must be in numpy array format")

    log_attr = np.log10(np.abs(attributions) + 1e-10)
    if signed: # to verify the positive and negative contributions
        signs = np.sign(attributions)
        # Reverts the abs procedure done to calculate the log before
        signed_log_attr = signs * (log_attr + abs(np.min(log_attr)))
        return signed_log_attr
    else:
        return log_attr

def log_scale_attr_zero_mask(attributions: np.array, signed: bool = False) -> np.array:
    """
    Apply a logarithmic scale over attributions.
    Attributions with zero value are substituted with NaN values. This is useful to
    plot attributions, since NaN values will have a gray color. Useful to check the
    receptive field of a specific region.

    Args:
        `attributions` (np.array): Attribution values.

        `signed` (bool, optional): Mantains the attributions signs after the scale if True.
        Defaults to False.

    Returns:
        `log_attributions` (numpy.array): Scaled attributions with NaN values
        where there is no attribution
    """
    
    if not isinstance(attributions, np.array):
        raise TypeError("'attributions' must be in numpy array format")
    if not isinstance(signed, bool):
        raise TypeError("'attributions' must be in numpy array format")

    zero_mask = attributions == 0

    # Apply the log scale to non-zero values
    log_attr = np.log10(np.abs(attributions[~zero_mask]) + 1e-10)

    # Verify the positive and negative contributions
    if signed:
        signs = np.sign(attributions)
        signed_log_attr = signs * (log_attr_with_nan + abs(np.nanmin(log_attr_with_nan)))
        log_attr = signed_log_attr

    # Replace values equal to zero with NaN
    log_attr_with_nan = np.empty_like(attributions, dtype=float)
    log_attr_with_nan[zero_mask] = np.nan
    log_attr_with_nan[~zero_mask] = log_attr

    return log_attr_with_nan

def join_attributions(np_attr1: np.array, np_attr2: np.array) -> np.array:
    """
    Execute an union of the maximum magnitudes from two attribution values.

    Args:
        `np_attr1` (np.array): First attributions values

        `np_attr2` (np.array): Second attributions values

    Returns:
        `joined_attributes` (np.array): Union of the max magnitude values from both attributions.
    """

    if not isinstance(np_attr1, np.array):
        raise TypeError("'np_attr1' must be in numpy array format.")
    if not isinstance(np_attr2, np.array):
        raise TypeError("'np_attr2' must be in numpy array format.")
    if np_attr1.shape != np_attr1.shape:
        raise ValueError("Both attributions must have the same shape.")

    final_attr = []

    # We need to execute an element-wise for-loop to mantain the signs
    for i in range(len(np_attr1)):
        if abs(np_attr1[i]) >= abs(np_attr2[i]):
            final_attr.append(np_attr1[i])
        else:
            final_attr.append(np_attr2[i])
    return np.array(final_attr)


# Plot functions

def show_point_cloud(np_coords: np.array, np_colors: np.array, size: float = 0.1) -> None:
    """
    Plots the Point Cloud using K3D plot library.

    Args:
        `np_coords` (numpy.array): Coordinates of the points, in the shape (N, 3)

        `np_colors` (numpy.array): Colors of the points, in the shape (N, 3).
        It assumes that the colors are in the RGB format with interval [0, 255]

        `size` (float, optional): Points size in the plot. Defaults to 0.1.
    """

    # Types and shapes verifications
    if not isinstance(np_coords, np.array):
        raise TypeError("The points coordinates must be in numpy array format.")
    if np_coords.shape[1] != 3:
        raise ValueError("The points coordinates must in shape (X,3), where the three values \
            correspond to the x, y and z coordinates of each point.")
    if not isinstance(size, float):
        raise TypeError("'size' must be a float.")

    rgb = np_colors.astype(np.uint32)
    colors_hex = (rgb[:, 0] << 16) + (rgb[:, 1] << 8) + (rgb[:, 2])

    plot = k3d.plot(grid_visible=False)
    plot += k3d.points(np_coords, colors_hex, point_size=size, shader="simple", name="Point Cloud")
    plot.display()


def show_point_cloud_classification_k3d(np_coords: np.array, np_class: np.array,
                                        size: float = 0.1) -> None:
    """
    Plots the classfication of each point using K3D.

    It's not capable to hold extra information, but it has a better performance than
    show_point_cloud_classification_plotly.
    Recomended to use when plotly's performance spoils the 3D interaction.

    Args:
        `np_coords` (numpy.array): Coordinates of the points, in the shape (N, 3)

        `np_class` (numpy.array): The predictions indices for each point.

        `size` (float, optional): Points sizes in the plot. Defaults to 0.1.
    """

    # Types and shapes verifications
    if not isinstance(np_coords, np.array):
        raise TypeError("The points coordinates must be in numpy array format.")
    if np_coords.shape[1] != 3:
        raise ValueError("The points coordinates must in shape (X,3), where the three values \
            correspond to the x, y and z coordinates of each point.")
    if not isinstance(np_class, np.array):
        raise TypeError("The points classifications must be in numpy array format.")
    if np_coords.shape[0] != np_class.shape[0]:
        raise ValueError("Coordinates and classifications should have the same amount of points")
    if not isinstance(size, float):
        raise TypeError("'size' must be a float.")

    plot = k3d.plot(grid_visible=False)
    plot += k3d.points(np_coords,
                       shader='flat',
                       attribution=np_class,
                       point_size=size,
                       color_map=create_k3d_category_20_discrete_map(),
                       color_range=[np.min(np_class), np.max(np_class)],
                       name="Point Classifications")
    plot.display()


def show_point_cloud_classification_plotly(np_coords: np.array, np_class: np.array,
                                           instance_labels: np.array = None,
                                           classes_dict: dict = None, size: float = 0.5,
                                           additional_hover_info: dict = None,
                                           save_html: bool = False,
                                           save_name: str = './default_fig.html') -> None:
    """
    Plots the classficiation of each point using Plotly.
    It can hold extra information such as instance labels and predictions meanings.

    Args:
        `np_coords` (numpy.array): Coordinates of the points, in the shape (N, 3).

        `np_class` (numpy.array): The predictions indices for each point.

        `instance_labels` (numpy.array, optional): Object instance labels for each point.
        The order of the points must be the same as the coordinates and classifications.
        Defaults to None.

        `classes_dict` (dict, optional): Dictionary containing the meaning of each prediction index.
        Defaults to None.

        `size` (float, optional): Points sizes in the plot. Defaults to 0.5.

        `additional_hover_info` (dict, optional): Dictionary containing additional information
        about the points. Each information must be in numpy format and with the same
        size as the `np_class` parameter. Defaults to None.

        `save_html` (bool, optional): Should the plotly figure be saved in a html file for later
        visualization. Defaults to False.

        `save_name` (str, optional): The name of the html file to be created if `save_html` is
        True. Defaults to './default_fig.html'.
    """

    # Types and shapes verifications
    if not isinstance(np_coords, np.array):
        raise TypeError("The points coordinates must be in numpy array format.")
    if np_coords.shape[1] != 3:
        raise ValueError("The points coordinates must in shape (X,3), where the three values \
            correspond to the x, y and z coordinates of each point.")
    if not isinstance(np_class, np.array):
        raise TypeError("The points classifications must be in numpy array format.")
    if np_coords.shape[0] != np_class.shape[0]:
        raise ValueError("Coordinates and classifications should have the same amount of points")
    if instance_labels is not None and not isinstance(instance_labels, np.array):
        raise TypeError("The points instance labels must be in numpy array format.")
    if instance_labels is not None and instance_labels.shape != np_class.shape:
        raise ValueError("'instance_labels' does not have the same shape as the classifications.")
    if classes_dict is not None and not isinstance(classes_dict, dict):
        raise TypeError("'classes_dict' must be a dictionary.")
    if not isinstance(size, float):
        raise TypeError("'size' must be a float.")
    if additional_hover_info is not None and not isinstance(additional_hover_info, dict):
        raise TypeError("'additional_hover_info' must be a dictionary.")
    for i, info in enumerate(additional_hover_info):
        if not isinstance(info, np.array):
            raise TypeError("The {}th element from additional_hover_info must be in numpy \
                array format.".format(i))
        if info.shape != np_class.shape:
            raise ValueError("The {}th element from additional_hover_info does not \
            have the same shape as the classifications.".format(i))
    if not isinstance(save_html, bool):
        raise TypeError("'save_html' must be a bool.")
    if not isinstance(save_name, bool):
        raise TypeError("'save_name' must be a str.")

    # Ensure int type for the classifications
    np_class = np_class.astype(np.int)

    # Creates a dictionary to be transformed in a dataframe later.
    # The dataframe is a better structure to be used in the plot function.
    # This is called 'hover' because it is the hover data to be showed when the cursor
    # is over a specific point.
    hover_data_names = ["Class", "Point_Num"]
    hover = dict(X=np_coords[:, 0],
                 Y=np_coords[:, 1],
                 Z=np_coords[:, 2],
                 Class=[classes_dict[class_num] for class_num in np_class],
                 Point_Num=[i for i in range(len(instance_labels))])

    # Adds additional data to the dictionary according to the presence of optional data
    if additional_hover_info is not None:
        hover.update(additional_hover_info)
        hover_data_names = hover_data_names + list(additional_hover_info.keys())

    if (instance_labels is not None):
        instance_labels = instance_labels.cpu().detach().numpy().astype(np.int)
        hover["Instance_Index"] = instance_labels
        hover_data_names.append("Instance_Index")

    hover_df = pd.DataFrame(hover)

    # Plotly default color set for discrete sequence data is limited.
    # Datasets that contains a great number of classes can't be represented
    # by Plotly's default color set and repeated colors will be used to represent
    # different classes. Colorcet lib contains a robust and diverse set of
    # colors and it is used instead.
    color_scale = []
    for color in cc.glasbey_bw_minc_20_maxl_70:
        hex_color = "#{:02X}{:02X}{:02X}".format(math.floor(color[0] * 255),
                                                 math.floor(color[1] * 255),
                                                 math.floor(color[2] * 255))
        color_scale.append(hex_color)

    fig = px.scatter_3d(data_frame=hover_df,
                        x="X",
                        y="Y",
                        z="Z",
                        color="Class",
                        color_discrete_sequence=color_scale,
                        opacity=1.0,
                        hover_data=hover_data_names)

    # Change size directly through here to avoid white outlines in the points
    for data in fig.data:
        data['marker']['size'] = size

    # The size change must not modify the legend
    fig.update_layout(legend= {'itemsizing': 'constant'})

    # Ensure that the rendering will be with the correct aspect (not flattened)
    fig.update_layout(scene_aspectmode='data')

    # Saves the figure to a new html file
    if save_html:
        fig.write_html(save_name)

    fig.show()


def explain_plotly(attributions: Tensor, coords: Tensor,
                   original_attributions: Tensor = None,
                   template_name: str = 'simple_white',
                   size: float = 1.5) -> None:
    """
    Plots the point cloud with its attributions values using Plotly.
    Useful for an thorough analysis of the points attribution values, but it has a poor interaction
    and scene understanding thanks to Plotly's point rendering.

    Args:
        attributions (TensorOrTupleOfTensorsGeneric): attributions for each point.
        coords (Tensor): Coordinates of each point.
        template_name (str, optional): The template style to be used for the plot.
        Defaults to 'simple_white'.
        size (float, optional): Size of the points to be rendered.
    """

    np_coords = coords.detach().cpu().numpy()

    np_attr = attributions.detach().cpu().numpy()
    if original_attributions is not None:
        np_orig_attr = original_attributions.detach().cpu().numpy()
    else:
        np_orig_attr = None

    hover = dict(X=np_coords[:, 0],
                 Y=np_coords[:, 1],
                 Z=np_coords[:, 2],
                 Attr=np_attr,
                 Original_Attr=np_orig_attr,
                 Point_Num=[i for i in range(len(np_attr))])

    hover_df = pd.DataFrame(hover)

    fig = px.scatter_3d(data_frame=hover_df,
                        x="X", y="Y", z="Z",
                        color="Attr",
                        opacity=1.0,
                        range_color=[np.min(np_attr), np.max(np_attr)],
                        template='simple_white',
                        hover_data=["Attr", "Original_Attr", "Point_Num"])

    for data in fig.data:
        data['marker']['size'] = size

    # The size change must not modify the legend
    fig.update_layout(legend= {'itemsizing': 'constant'})
    # Ensure that the rendering will be with the correct aspect (not flattened)
    fig.update_layout(scene_aspectmode='data')

    fig.show()


def explain_k3d(attributions: Tensor, coords: Tensor, attribution_name=None, size: float = 0.05) -> None:
    """
    Plots the point cloud with its attributions values using K3D.
    It doesn't offer the exactly value of the attributions but its performance and scene
    understanding are better than Plotly's explanation.

    Args:
        attributions (TensorOrTupleOfTensorsGeneric): attributions for each point.
        coords (Tensor): Coordinates of each point.
        attribution_name (str, optional): Name of the point data in the plot. Defaults to None.
        size (float, optional): Size of the points to be rendered.
    """

    if attribution_name is None:
        attribution_name = "attributions"

    fig = k3d.plot(grid_visible=False)

    np_attr = attributions.detach().cpu().numpy()

    fig += k3d.points(positions=coords,
                      shader='3d',
                      color_map=k3d.paraview_color_maps.Viridis_matplotlib,
                      attribution=np_attr,
                      color_range=[np.min(np_attr), np.max(np_attr)],
                      point_size=size,
                      name=attribution_name)
    fig.display()
