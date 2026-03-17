#!/usr/bin/env python
# coding: utf-8

"""DLRA solver for the linear gyrokinetic test problem."""

from __future__ import annotations

import argparse

import numpy as np
import xarray as xr

from linear_gyrokinetic import (
    GKParameters,
    build_geometry,
    complex_to_parts,
    flatten_vm,
    init_state,
    projector_splitting_step,
    state_fields_from_h,
    weighted_gram,
)
from low_rank_approx import LowRankApprox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DLRA linear gyrokinetic solver")
    parser.add_argument("--out", default="lingk_dlra.nc")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--nt", type=int, default=41)
    parser.add_argument("--nskip", type=int, default=10)
    parser.add_argument("--nz", type=int, default=24)
    parser.add_argument("--nv", type=int, default=8)
    parser.add_argument("--nm", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ky", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = GKParameters(
        nz=args.nz,
        nv=args.nv,
        nm=args.nm,
        dt=args.dt,
        dt_out=args.dt * args.nskip,
        seed=args.seed,
        ky=args.ky,
        beta=args.beta,
    )
    geom = build_geometry(params)
    hk, fk, pk, ak = init_state(params, geom)

    h_mat = flatten_vm(hk)
    max_rank = min(h_mat.shape)
    if args.rank > max_rank:
        raise ValueError(f"rank must be <= min(2*nz, 2*nv*(nm+1)*ns) = {max_rank} (got {args.rank})")

    h_lra = LowRankApprox(geom.nz_tot, geom.nvm, args.rank, dtype=np.complex128)
    h_lra.init_from_full(h_mat, dx=geom.dz, dv=geom.vm_weight)

    nsave = (args.nt - 1) // args.nskip + 1
    time = 0.0

    x_all = np.zeros((nsave, geom.nz_tot, args.rank), dtype=np.complex128)
    s_all = np.zeros((nsave, args.rank, args.rank), dtype=np.complex128)
    v_all = np.zeros((nsave, geom.nvm, args.rank), dtype=np.complex128)
    f_all = np.zeros((nsave, geom.nz_tot, geom.nv_tot, geom.nm_tot, params.ns), dtype=np.complex128)
    phi_all = np.zeros((nsave, geom.nz_tot), dtype=np.complex128)
    a_all = np.zeros((nsave, geom.nz_tot), dtype=np.complex128)

    for isave in range(nsave):
        hk = h_lra.to_full().reshape(geom.nz_tot, geom.nv_tot, geom.nm_tot, params.ns)
        fk, pk, ak = state_fields_from_h(hk, geom)

        x_all[isave] = h_lra.X
        s_all[isave] = h_lra.S
        v_all[isave] = h_lra.V
        f_all[isave] = fk
        phi_all[isave] = pk
        a_all[isave] = ak

        for _ in range(args.nskip):
            h_lra = projector_splitting_step(h_lra, args.dt, geom)
            time += args.dt

    x_real, x_imag = complex_to_parts(x_all)
    s_real, s_imag = complex_to_parts(s_all)
    v_real, v_imag = complex_to_parts(v_all)
    f_real, f_imag = complex_to_parts(f_all)
    phi_real, phi_imag = complex_to_parts(phi_all)
    a_real, a_imag = complex_to_parts(a_all)

    x_gram = np.stack([weighted_gram(x_all[i], geom.dz) for i in range(nsave)], axis=0)
    v_gram = np.stack([weighted_gram(v_all[i], geom.vm_weight) for i in range(nsave)], axis=0)
    x_gram_real, x_gram_imag = complex_to_parts(x_gram)
    v_gram_real, v_gram_imag = complex_to_parts(v_gram)

    ds = xr.Dataset(
        data_vars={
            "X_real": (("time", "z", "rank"), x_real),
            "X_imag": (("time", "z", "rank"), x_imag),
            "S_real": (("time", "rank", "rank2"), s_real),
            "S_imag": (("time", "rank", "rank2"), s_imag),
            "V_real": (("time", "vm", "rank"), v_real),
            "V_imag": (("time", "vm", "rank"), v_imag),
            "f_real": (("time", "z", "vl", "mu", "species"), f_real),
            "f_imag": (("time", "z", "vl", "mu", "species"), f_imag),
            "phi_real": (("time", "z"), phi_real),
            "phi_imag": (("time", "z"), phi_imag),
            "A_real": (("time", "z"), a_real),
            "A_imag": (("time", "z"), a_imag),
            "X_gram_real": (("time", "rank", "rank2"), x_gram_real),
            "X_gram_imag": (("time", "rank", "rank2"), x_gram_imag),
            "V_gram_real": (("time", "rank", "rank2"), v_gram_real),
            "V_gram_imag": (("time", "rank", "rank2"), v_gram_imag),
        },
        coords={
            "time": np.arange(nsave) * params.dt_out,
            "z": geom.zz,
            "vl": geom.vl,
            "mu": geom.mu,
            "species": np.arange(params.ns),
            "rank": np.arange(args.rank),
            "rank2": np.arange(args.rank),
            "vm": np.arange(geom.nvm),
        },
        attrs={
            "dt": args.dt,
            "nt": args.nt,
            "nskip": args.nskip,
            "rank": args.rank,
            "nz": args.nz,
            "nv": args.nv,
            "nm": args.nm,
            "seed": args.seed,
            "ky": args.ky,
            "beta": args.beta,
        },
    )
    ds.to_netcdf(args.out)
    print(f"[lingk-dlra] wrote dataset: {args.out}")


if __name__ == "__main__":
    main()
