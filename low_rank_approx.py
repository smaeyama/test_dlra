#!/usr/bin/env python
# coding: utf-8

"""Low-rank representation utilities for DLRA."""

from __future__ import annotations

import numpy as np


class LowRankApprox:
    def __init__(self, nx: int, nv: int, nr: int, dtype=np.float64):
        self.nx = nx
        self.nv = nv
        self.nr = nr
        self.dtype = np.dtype(dtype)
        self.X = np.zeros((nx, nr), dtype=self.dtype)
        self.S = np.zeros((nr, nr), dtype=self.dtype)
        self.V = np.zeros((nv, nr), dtype=self.dtype)

    def init_from_full(self, f_xv: np.ndarray, dx: float, dv: float):
        U, s, Vt = np.linalg.svd(f_xv, full_matrices=False)
        r = min(self.nr, s.size)
        self.X[:, :r] = U[:, :r] / np.sqrt(dx)
        self.S[:r, :r] = np.sqrt(dx) * np.diag(s[:r]) * np.sqrt(dv)
        self.V[:, :r] = Vt[:r, :].T / np.sqrt(dv)

    def init_from_tensors(self, X: np.ndarray, S: np.ndarray, V: np.ndarray):
        if X.shape != (self.nx, self.nr):
            raise ValueError(f"X must have shape ({self.nx}, {self.nr}).")
        if S.shape != (self.nr, self.nr):
            raise ValueError(f"S must have shape ({self.nr}, {self.nr}).")
        if V.shape != (self.nv, self.nr):
            raise ValueError(f"V must have shape ({self.nv}, {self.nr}).")

        self.X = np.array(X, copy=True)
        self.S = np.array(S, copy=True)
        self.V = np.array(V, copy=True)

    def to_full(self):
        return self.X @ self.S @ self.V.T
