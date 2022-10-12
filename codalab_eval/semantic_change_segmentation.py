"""Numpy Implementation of the Semantic Change Segmentation (SDS).
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
      float: The mIoU score.
    """
    intersections = conf.diagonal()
    fps = conf.sum(axis=0) - intersections
    fns = conf.sum(axis=1) - intersections
    unions = intersections + fps + fns

    num_classes = np.count_nonzero(intersections + fns)
    ious = (
            intersections.astype(np.double) /
            np.maximum(unions, _EPSILON).astype(np.double))
    return np.sum(ious) / num_classes


class SCSMetric(object):
    """Metric class for the Semantic Change Segmentation (SDS).
      Please see the following paper for more details about the metric:

        A. Toker, L. Kondmann, M. Weber, M. Eisenberger, A. Camero, J. Hu,
        A. Hoderlein, C. Senaras, T. Davis, D. Cremers, G. Marchisio, X. Zhu,
        L. Leal-Taixe, DynamicEarthNet: Daily Multi-Spectral Satellite Dataset for
        Semantic Change Segmentation, CVPR 2022.

      The metric computes the airthmetic mean of two terms.
      - Binary Change (BC): This term measures the quality of the prediction
          regarding which pixels did change.
      - Semantic Change (SC): This term measure the quality of the semantic
          predictions for the pixels that did change.
      Example usage:
      scs_obj = semantic_change_segmentation.SCSMetric(
        num_classes, ignore_label)
      scs_obj.update_state(y_true_binary_1, y_true_multi_class_1,
        y_pred_binary_1, y_pred_multi_class_1)
      scs_obj.update_state(y_true_binary_2, y_true_multi_class_2,
        y_pred_binary_2, y_pred_multi_class_2)
      ...
      result = scs_obj.result()
    """

    def __init__(self, num_classes, ignore_label):
        """Initialization of the SCS metric.
        Args:
            num_classes: Number of classes in the dataset as an integer.
            ignore_label: The class id to be ignored in evaluation as an integer or integer tensor.
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

        self._sc_iou_confusion_matrix_per_cube = collections.OrderedDict()
        self._binary_iou_per_cube = collections.OrderedDict()
        self._cube_length = collections.OrderedDict()
        self._label_bit_shift = 32
        self._bit_mask = (2 ** self._label_bit_shift) - 1

    def update_state(self, y_true_binary, y_true_multi_class,
                     y_pred_binary, y_pred_multi_class,
                     cube_id=0):
        """Accumulates the semantic change segmentation statistics.
        Args:
          y_true_binary: The ground-truth binary change label map for a cube frame.
            The expected format is np.bool_ with `True` for change.
          y_true_multi_class: The ground-truth semantic label map for a cube frame.
          y_pred_binary: The predicted binary change label map for a cube frame.
            The expected format is np.bool_ with `True` for change.
          y_pred_multi_class: The predicted label map for a cube frame.
          cube_id: The optional ID of the sequence the frames belong to. When no
            cube ID is given, all frames are considered to belong to the same
            cube (default: 0).
        """
        y_true_binary = y_true_binary.astype(np.bool_)
        y_pred_binary = y_pred_binary.astype(np.bool_)

        semantic_label = y_true_multi_class.astype(np.int64)
        semantic_prediction = y_pred_multi_class.astype(np.int64)

        if self._ignore_label > self._num_classes:
            semantic_label = np.where(semantic_label != self._ignore_label,
                                      semantic_label, self._num_classes)
            semantic_prediction = np.where(semantic_prediction != self._ignore_label,
                                           semantic_prediction, self._num_classes)

        # Limit semantic predictions to ground-truth change.
        semantic_change_label = semantic_label[y_true_binary == True]
        semantic_change_prediction = semantic_prediction[y_true_binary == True]

        if cube_id not in self._sc_iou_confusion_matrix_per_cube:
            self._sc_iou_confusion_matrix_per_cube[cube_id] = np.zeros(
                (self._confusion_matrix_size, self._confusion_matrix_size),
                dtype=np.int64)
            self._binary_iou_per_cube[cube_id] = np.zeros((2, 2), dtype=np.int64)

            self._cube_length[cube_id] = 0

        # Semantic Change:
        idxs = (np.reshape(semantic_change_label, [-1]) <<
                self._label_bit_shift) + np.reshape(semantic_change_prediction, [-1])
        unique_idxs, counts = np.unique(idxs, return_counts=True)
        self._sc_iou_confusion_matrix_per_cube[cube_id][
            unique_idxs >> self._label_bit_shift,
            unique_idxs & self._bit_mask] += counts

        self._cube_length[cube_id] += 1

        # Binary Change
        idxs = np.stack([
            np.reshape(y_true_binary, [-1]),
            np.reshape(y_pred_binary, [-1])
        ], axis=0).astype(np.int32)
        np.add.at(self._binary_iou_per_cube[cube_id], tuple(idxs), 1)

    def result(self):
        """Computes the semantic change segmentation metric.
        Returns:
          A dictionary containing:
            - 'SCS': The total semantic change segmentation score.
            - 'BC': The total binary change score.
            - 'SC': The total semantic change score.
            - 'SCS_per_cube': A list of SCS scores per cube.
            - 'BC_per_cube': A list of BC scores per cube.
            - 'SC_per_cube': A list of SC scores per cube.
            - 'Id_per_cube': A list of string-type cube Ids to map list index to
                cube.
            - 'Length_per_cube': A list of the length of each cube.
        """
        bc_per_cube = [0] * len(self._binary_iou_per_cube)
        sc_per_cube = [0] * len(self._binary_iou_per_cube)
        id_per_cube = [''] * len(self._binary_iou_per_cube)

        # Compute IoU scores.
        # The rows correspond to ground-truth and the columns to predictions.
        # Remove fp from confusion matrix for the void/ignore class.
        total_change_confusion = np.zeros(
            (self._confusion_matrix_size, self._confusion_matrix_size),
            dtype=np.int64)

        total_binary_confusion = np.zeros((2, 2), np.int64)
        for index, cube_id in enumerate(self._sc_iou_confusion_matrix_per_cube):
            id_per_cube[index] = cube_id
            change_confusion = self._sc_iou_confusion_matrix_per_cube[cube_id]
            removal_matrix = np.zeros_like(change_confusion)
            removal_matrix[self._include_indices, :] = 1.0
            change_confusion *= removal_matrix
            total_change_confusion += change_confusion
            sc_per_cube[index] = _compute_miou(change_confusion)

            binary_confusion = self._binary_iou_per_cube[cube_id]
            total_binary_confusion += binary_confusion

            tps = binary_confusion[1, 1]
            fps = binary_confusion[0, 1]
            fns = binary_confusion[1, 0]
            union = tps + fps + fns
            bc_per_cube[index] = tps.astype(np.double) / np.maximum(union, 1e-15).astype(np.double)

        sc = _compute_miou(total_change_confusion)

        tps = total_binary_confusion[1, 1]
        fps = total_binary_confusion[0, 1]
        fns = total_binary_confusion[1, 0]
        union = tps + fps + fns
        bc = tps.astype(np.double) / np.maximum(union, _EPSILON).astype(np.double)

        scs = 0.5 * (bc + sc)
        scs_per_cube = 0.5 * (np.array(bc_per_cube) + np.array(sc_per_cube))

        return {
            'SCS': scs,
            'BC': bc,
            'SC': sc,
            'SCS_per_cube': scs_per_cube,
            'BC_per_cube': bc_per_cube,
            'SC_per_cube': sc_per_cube,
            'ID_per_cube': id_per_cube,
            'Length_per_cube': list(self._cube_length.values()),
        }

    def reset_states(self):
        """Resets all states that accumulated data."""
        self._sc_iou_confusion_matrix_per_cube = collections.OrderedDict()
        self._binary_iou_per_cube = collections.OrderedDict()
        self._cube_length = collections.OrderedDict()