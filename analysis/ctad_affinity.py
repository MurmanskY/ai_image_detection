"""Nonparametric calibration utilities for cross-tokenizer affinity scores."""

import numpy as np


def lower_tail_surprisal(q_values, reference_qerr):
    """Return per-tokenizer lower-tail surprisal for each image.

    ``q_values`` has shape ``(n_images, n_tokenizers)``. ``reference_qerr``
    contains one real-only reference array per tokenizer. The add-one
    correction prevents zero empirical probabilities.
    """
    values = np.asarray(q_values, dtype=float)
    if (values.ndim != 2 or len(reference_qerr) == 0 or
            values.shape[1] != len(reference_qerr)):
        raise ValueError("q_values must be (n_images, n_tokenizers)")
    if not np.isfinite(values).all():
        raise ValueError("q_values must contain only finite values")

    surprisal = np.empty_like(values)
    for index, reference in enumerate(reference_qerr):
        reference = np.asarray(reference, dtype=float)
        if reference.ndim != 1 or reference.size == 0:
            raise ValueError("each tokenizer needs a non-empty 1D reference")
        if not np.isfinite(reference).all():
            raise ValueError("reference_qerr must contain only finite values")
        sorted_reference = np.sort(reference)
        count = np.searchsorted(sorted_reference, values[:, index], side="right")
        tail_probability = (count + 1) / (reference.size + 1)
        surprisal[:, index] = -np.log(tail_probability)
    return surprisal


def affinity_score(q_values, reference_qerr):
    """Aggregate sparse affinity evidence using the maximum tail surprisal."""
    return lower_tail_surprisal(q_values, reference_qerr).max(axis=1)


def conformal_upper_tail_p(calibration_scores, scores):
    """Compute finite-sample upper-tail conformal p-values.

    The calibration scores must be computed with a reference model fitted on
    data independent of both calibration and test samples.
    """
    calibration = np.asarray(calibration_scores, dtype=float)
    values = np.asarray(scores, dtype=float)
    if calibration.ndim != 1 or calibration.size == 0:
        raise ValueError("calibration_scores must be a non-empty 1D array")
    if values.ndim != 1:
        raise ValueError("scores must be a 1D array")
    if not np.isfinite(calibration).all() or not np.isfinite(values).all():
        raise ValueError("scores must contain only finite values")
    exceedances = (calibration[:, None] >= values[None, :]).sum(axis=0)
    return (exceedances + 1) / (calibration.size + 1)
