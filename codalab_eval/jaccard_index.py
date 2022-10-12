"""Numpy Implementation of the Jaccard Index (mIoU).
This implementation is designed to work stand-alone. Please feel free to copy
this file and the corresponding unit-test to your project.
"""

import collections
import numpy as np

_EPSILON = 1e-15


def _compute_miou(conf):
    """Helper function to compute the (m)IoU score based on confusion matrix.
    Args:
      conf (np.ndarray): The confusion amtrix as numpy array.
    Returns:
      Tuple[float, np.ndarray]: The mIoU score and the IoU score per class..
    """
    intersections = conf.diagonal()
    fps = conf.sum(axis=0) - intersections
    fns = conf.sum(axis=1) - intersections
    unions = intersections + fps + fns

    num_classes = np.count_nonzero(unions)
    ious = (
            intersections.astype(np.double) /
            np.maximum(unions, _EPSILON).astype(np.double))
    return np.sum(ious) / num_classes, ious


class JaccardIndexMetric(object):
    """Metric class for the Jaccard Index (mIoU).
    The metric computes the airthmetic mean of the per class IoU.
    Example usage:
    jaccard_obj = jaccard_index.JaccardIndexMetric(
      num_classes, ignore_label)
    jaccard_obj.update_state(y_true_multi_class_1, y_pred_multi_class_1)
    jaccard_obj.update_state(y_true_multi_class_2, y_pred_multi_class_2)
    ...
    result = jaccard_obj.result()
    """

    def __init__(self, num_classes, ignore_label):
        """Initialization of the Jaccard Index metric.
        Args:
          num_classes: Number of classes in the dataset as an integer.
          ignore_label: The class id to be ignored in evaluation as an integer or
            integer tensor.
        """
        self._num_classes = num_classes
        self._ignore_label = ignore_label

        if ignore_label >= num_classes:
            self._confusion_matrix_size = num_classes + 1
            self._include_indices = np.arange(self._num_classes)
        else:
            self._confusion_matrix_size = num_classes
            self._include_indices = np.array(
                [i for i in range(num_classes) if i != self._ignore_label])

        self._iou_confusion_matrix_per_cube = collections.OrderedDict()
        self._cube_length = collections.OrderedDict()
        self._label_bit_shift = 32
        self._bit_mask = (2 ** self._label_bit_shift) - 1

    def update_state(self, y_true_multi_class,
                     y_pred_multi_class,
                     cube_id=0):
        """Accumulates the semantic change segmentation statistics.
        Args:
          y_true_multi_class: The ground-truth semantic label map for a cube frame.
          y_pred_multi_class: The predicted label map for a cube frame.
          cube_id: The optional ID of the sequence the frames belong to. When no
            cube ID is given, all frames are considered to belong to the same
            cube (default: 0).
        """
        semantic_label = y_true_multi_class.astype(np.int64)
        semantic_prediction = y_pred_multi_class.astype(np.int64)

        # Check if the ignore value is outside the range [0, num_classes]. If yes,
        # map `_ignore_label` to `_num_classes`, so it can be used to create the
        # confusion matrix.
        if self._ignore_label > self._num_classes:
            semantic_label = np.where(semantic_label != self._ignore_label,
                                      semantic_label, self._num_classes)
            semantic_prediction = np.where(semantic_prediction != self._ignore_label,
                                           semantic_prediction, self._num_classes)

        if cube_id not in self._iou_confusion_matrix_per_cube:
            self._iou_confusion_matrix_per_cube[cube_id] = np.zeros(
                (self._confusion_matrix_size, self._confusion_matrix_size),
                dtype=np.int64)
            self._cube_length[cube_id] = 0

        # Standard IoU:
        idxs = (np.reshape(semantic_label, [-1]) <<
                self._label_bit_shift) + np.reshape(semantic_prediction, [-1])
        unique_idxs, counts = np.unique(idxs, return_counts=True)
        self._iou_confusion_matrix_per_cube[cube_id][
            unique_idxs >> self._label_bit_shift,
            unique_idxs & self._bit_mask] += counts

        self._cube_length[cube_id] += 1

    def result(self):
        """Computes the Jaccard Index metric.
        Returns:
          A dictionary containing:
            - 'mIoU': The mean IoU score.
            - 'IoU_per_class': The IoU scores per class.
            - 'mIoU_per_cube': A list of mIoU per cube.
            - 'Id_per_cube': A list of string-type cube Ids to map list index to
                cube.
            - 'Length_per_cube': A list of the length of each cube.
        """
        id_per_cube = [''] * len(self._cube_length)
        mIoU_per_cube = [''] * len(self._cube_length)

        # Compute IoU scores.
        # The rows correspond to ground-truth and the columns to predictions.
        # Remove fp from confusion matrix for the void/ignore class.
        total_iou_confusion = np.zeros(
            (self._confusion_matrix_size, self._confusion_matrix_size),
            dtype=np.int64)

        for index, cube_id in enumerate(self._iou_confusion_matrix_per_cube):
            id_per_cube[index] = cube_id
            iou_confusion = self._iou_confusion_matrix_per_cube[cube_id]
            removal_matrix = np.zeros_like(iou_confusion)
            removal_matrix[self._include_indices, :] = 1.0
            iou_confusion *= removal_matrix
            total_iou_confusion += iou_confusion

            mIoU_per_cube[index], _ = _compute_miou(iou_confusion)

        mIoU, iou_per_class = _compute_miou(total_iou_confusion)
        iou_per_class = iou_per_class[self._include_indices]

        return {
            'mIoU': mIoU,
            'IoU_per_class': iou_per_class,
            'ID_per_cube': id_per_cube,
            'mIoU_per_cube': mIoU_per_cube,
            'Length_per_cube': list(self._cube_length.values()),
        }

    def reset_states(self):
        """Resets all states that accumulated data."""
        self._iou_confusion_matrix_per_cube = collections.OrderedDict()
        self._cube_length = collections.OrderedDict()