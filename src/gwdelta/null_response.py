from dataclasses import dataclass

import numpy as np

from .array_backend import infer_backend_from_array
from .fd_response import link_fd_response, second_generation_xyz_from_links


@dataclass(frozen=True)
class XYZNullResponse:
    frequency_hz: np.ndarray
    weights: np.ndarray
    valid: np.ndarray

    def apply(self, xyz_spectra):
        xp = infer_backend_from_array(xyz_spectra).xp
        data = xp.asarray(xyz_spectra)
        if data.shape != (3, len(self.frequency_hz)):
            raise ValueError("xyz_spectra must have shape (3, number of frequencies), ordered X,Y,Z")
        valid = xp.asarray(self.valid)
        if not bool(xp.all(xp.isfinite(data[:, valid]))):
            raise ValueError("Nonfinite XYZ spectra in valid frequency bins")
        return xp.sum(xp.asarray(self.weights).T*data, axis=0)

    def noise_psd(self, xyz_covariance):
        xp = infer_backend_from_array(xyz_covariance).xp
        covariance = xp.asarray(xyz_covariance)
        if covariance.shape != (len(self.frequency_hz), 3, 3):
            raise ValueError("xyz_covariance must have shape (number of frequencies,3,3)")
        valid = xp.asarray(self.valid)
        selected = covariance[valid]
        if not bool(xp.all(xp.isfinite(selected))):
            raise ValueError("Nonfinite covariance in valid frequency bins")
        scale = xp.max(xp.abs(selected), axis=(1, 2))
        if bool(xp.any(xp.max(xp.abs(selected-selected.conj().swapaxes(1, 2)), axis=(1, 2)) > 1e-10*scale)):
            raise ValueError("XYZ covariance must be Hermitian")
        if bool(xp.any(xp.linalg.eigvalsh(selected)[:, 0] < -1e-10*scale)):
            raise ValueError("XYZ covariance must be positive semidefinite")
        weights = xp.asarray(self.weights)
        psd = xp.einsum("fi,fij,fj->f", weights, covariance, weights.conj()).real
        return xp.where(valid, xp.maximum(psd, 0), xp.nan)


def xyz_null_response(frequency_hz, *, positions_m, lam, beta, rank_rtol=1e-8):
    frequency = np.array(frequency_hz, dtype=float, copy=True)
    positions = np.asarray(positions_m, dtype=float)
    if (frequency.ndim != 1 or frequency.size == 0 or not np.all(np.isfinite(frequency))
            or np.any(frequency < 0) or np.any(np.diff(frequency) <= 0)):
        raise ValueError("frequency_hz must be a nonempty, increasing, finite nonnegative array")
    if positions.shape != (3, 3) or not np.all(np.isfinite(positions)):
        raise ValueError("positions_m must contain three finite Cartesian spacecraft positions")
    sides = positions[1:]-positions[0]
    if np.linalg.norm(np.cross(*sides)) <= 1e-12*np.linalg.norm(sides[0])*np.linalg.norm(sides[1]):
        raise ValueError("Spacecraft positions must form a nondegenerate triangle")
    if not np.isfinite(lam) or not np.isfinite(beta) or abs(beta) > np.pi/2:
        raise ValueError("lam and beta must be finite radians, with abs(beta) <= pi/2")
    if not np.isfinite(rank_rtol) or not 0 < rank_rtol < 1:
        raise ValueError("rank_rtol must lie strictly between zero and one")
    links = link_fd_response(frequency, lam=lam, beta=beta, positions_m=positions)
    response = np.stack([np.stack(second_generation_xyz_from_links(
        frequency, value, positions_m=positions), axis=-1) for value in links], axis=-1)
    u, singular, _ = np.linalg.svd(response, full_matrices=True)
    valid = ((singular[:, 0] > singular[:, 0].max()*1e-12)
             & (singular[:, 1] > rank_rtol*singular[:, 0]))
    weights = u[:, :, -1].conj()
    pivot = np.argmax(np.abs(weights), axis=1)
    weights *= np.exp(-1j*np.angle(weights[np.arange(len(weights)), pivot]))[:, None]
    weights[~valid] = np.nan
    for array in (frequency, weights, valid):
        array.setflags(write=False)
    return XYZNullResponse(frequency, weights, valid)
