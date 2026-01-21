#!/usr/bin/env python3
"""
Stage-1 folded cascode sizing script.

Uses gm/Id lookup tables (nch_1v.mat, nch_2v.mat, pch_1v.mat, pch_2v.mat)
via look_up.py to size M1–M12 of a single-ended folded cascode op-amp stage.

"""

import numpy as np
from look_up import (
    importdata,
    look_up_basic,
    look_up_vs_gm_id,
    look_up_vgs_vs_gm_id,
)

# ---------------------------------------------------------------------
# Helper: snap a requested L to the closest L available in the LUT
# ---------------------------------------------------------------------


def nearest_L(device, L_desired):
    """
    Given a device LUT dict (nch_x or pch_x) and a desired channel length,
    return the closest valid L on that LUT grid.
    """
    L_grid = np.array(device["L"])
    idx = np.argmin(np.abs(L_grid - L_desired))
    return float(L_grid[idx])


# ---------------------------------------------------------------------
# Helper: size ONE transistor given gm/Id, branch current, L, VDS, VSB
# ---------------------------------------------------------------------


def size_device(device, name, gm_id_target, I_branch, L, vds_opt, vsb=0.0):
    """
    device   : gm/Id lookup dict (e.g., nch_2, pch_2)
    name     : string, just for printing
    gm_id_target : target gm/Id (1/V)
    I_branch : desired drain current in this device (A)
    L        : channel length (um, same units as in .mat) – assumed valid
    vds_opt  : VDS bias used for sizing (V)
    vsb      : VSB bias (V)

    Returns dict with W, L, gm, ro, VGS, VOV, etc.
    """

    # 1) VGS from gm/Id (use look_up_vgs_vs_gm_id)
    vgs = float(
        look_up_vgs_vs_gm_id(
            device,
            gm_id_target,
            vds=vds_opt,
            vsb=vsb,
            l=L,
        )
    )

    # 2) ID/W from gm/Id
    Id_per_W = float(
        look_up_vs_gm_id(
            device,
            "ID_W",
            gm_id_target,
            vds=vds_opt,
            vsb=vsb,
            l=L,
        )
    )  # A / µm

    # 3) Width from required branch current
    W = I_branch / Id_per_W

    # 4) gds/ID → ro
    gds_per_Id = float(
        look_up_vs_gm_id(
            device,
            "GDS_ID",
            gm_id_target,
            vds=vds_opt,
            vsb=vsb,
            l=L,
        )
    )  # (S / A) = 1 / V
    ro = 1.0 / (gds_per_Id * I_branch)

    # 5) gm from gm/Id
    gm = gm_id_target * I_branch

    # 6) VT and VOV
    vt = float(look_up_basic(device, "VT", vgs, vds_opt, vsb, L))
    vov = vgs - vt

    return {
        "name": name,
        "W": W,
        "L": L,
        "Id": I_branch,
        "gm": gm,
        "ro": ro,
        "gm_id": gm_id_target,
        "vgs": vgs,
        "vt": vt,
        "vov": vov,
    }


# ---------------------------------------------------------------------
# Stage-1 folded cascode sizing
# ---------------------------------------------------------------------


def size_stage1_folded(params, nch_1, nch_2, pch_1, pch_2):
    """
    Size M1–M12 of a single-ended folded cascode stage.

    We treat:
      M1, M2 : NMOS input diff pair (nch_2)
      M3, M4 : bottom NMOS cascodes (nch_2)
      M5, M6 : folded NMOS devices (nch_2)
      M7, M8 : PMOS cascodes (pch_2)
      M9, M10: PMOS loads (pch_2)
      M11    : NMOS tail device (nch_2)
      M12    : NMOS current-mirror device for tail (nch_2)
    """

    VDDH = params["VDDH"]
    Iref = params["Iref"]

    # gm targets from week_1.py (sanity)
    gm1_target = params["gm1_target"]  # overall input gm (both sides)
    gm2_target = params["gm2_target"]  # 2nd-stage gm (used later)

    # gm/Id targets
    gm_id_in = params["gm_id_in"]          # M1, M2
    gm_id_tail = params["gm_id_tail"]      # M11, M12
    gm_id_fold = params["gm_id_fold"]      # M5, M6
    gm_id_casc_n = params["gm_id_casc_n"]  # M3, M4
    gm_id_casc_p = params["gm_id_casc_p"]  # M7, M8
    gm_id_load_p = params["gm_id_load_p"]  # M9, M10

    # Length *targets* (µm) from your params…
    L_in_target = params["L_in"]
    L_tail_target = params["L_tail"]
    L_fold_target = params["L_fold"]
    L_casc_n_target = params["L_casc_n"]
    L_casc_p_target = params["L_casc_p"]
    L_load_p_target = params["L_load_p"]

    # Snap each L to the nearest legal grid for that device
    L_in = nearest_L(nch_2, L_in_target)
    L_tail = nearest_L(nch_2, L_tail_target)
    L_fold = nearest_L(nch_2, L_fold_target)
    L_casc_n = nearest_L(nch_2, L_casc_n_target)
    L_casc_p = nearest_L(pch_2, L_casc_p_target)
    L_load_p = nearest_L(pch_2, L_load_p_target)

    # Approximate VDS biases for sizing (V)
    vds_in = params["vds_in"]
    vds_tail = params["vds_tail"]
    vds_fold = params["vds_fold"]
    vds_casc_n = params["vds_casc_n"]
    vds_casc_p = params["vds_casc_p"]
    vds_load_p = params["vds_load_p"]

    # -----------------------------------------------------------------
    # Currents in each branch
    # -----------------------------------------------------------------
    # Tail current (M11) = Iref
    I_tail = Iref

    # Each side of the diff pair gets half the tail current
    I_M1 = I_tail / 2.0
    I_M2 = I_tail / 2.0

    # Each folded device (M5, M6) carries one input-branch current
    I_M5 = I_M1
    I_M6 = I_M2

    # Loads and cascodes above carry same current as folded devices
    I_M3 = I_M5
    I_M4 = I_M6
    I_M7 = I_M5
    I_M8 = I_M6
    I_M9 = I_M5
    I_M10 = I_M6

    # Tail mirror (M12) has the same current as M11 (1:1 mirror here)
    I_M11 = I_tail
    I_M12 = I_tail

    results = {}

    # ------------------ Input pair M1, M2 --------------------------------
    results["M1"] = size_device(
        nch_2, "M1 (in+)", gm_id_in, I_M1, L_in, vds_in, vsb=0.0
    )
    results["M2"] = size_device(
        nch_2, "M2 (in-)", gm_id_in, I_M2, L_in, vds_in, vsb=0.0
    )

    # ------------------ Bottom cascodes M3, M4 ---------------------------
    results["M3"] = size_device(
        nch_2, "M3 (bottom cascode)", gm_id_casc_n, I_M3, L_casc_n, vds_casc_n, vsb=0.0
    )
    results["M4"] = size_device(
        nch_2, "M4 (bottom cascode)", gm_id_casc_n, I_M4, L_casc_n, vds_casc_n, vsb=0.0
    )

    # ------------------ Folded devices M5, M6 ----------------------------
    results["M5"] = size_device(
        nch_2, "M5 (folded)", gm_id_fold, I_M5, L_fold, vds_fold, vsb=0.0
    )
    results["M6"] = size_device(
        nch_2, "M6 (folded)", gm_id_fold, I_M6, L_fold, vds_fold, vsb=0.0
    )

    # ------------------ PMOS cascodes M7, M8 -----------------------------
    # For PMOS in your gm/Id LUT, we still use VDS = VDDH - node voltages,
    # but here we just bias with vds_casc_p as a positive magnitude.
    results["M7"] = size_device(
        pch_2, "M7 (PMOS cascode)", gm_id_casc_p, I_M7, L_casc_p, vds_casc_p, vsb=0.0
    )
    results["M8"] = size_device(
        pch_2, "M8 (PMOS cascode)", gm_id_casc_p, I_M8, L_casc_p, vds_casc_p, vsb=0.0
    )

    # ------------------ PMOS loads M9, M10 -------------------------------
    results["M9"] = size_device(
        pch_2, "M9 (PMOS load)", gm_id_load_p, I_M9, L_load_p, vds_load_p, vsb=0.0
    )
    results["M10"] = size_device(
        pch_2, "M10 (PMOS load)", gm_id_load_p, I_M10, L_load_p, vds_load_p, vsb=0.0
    )

    # ------------------ Tail devices M11, M12 ----------------------------
    results["M11"] = size_device(
        nch_2, "M11 (tail)", gm_id_tail, I_M11, L_tail, vds_tail, vsb=0.0
    )
    results["M12"] = size_device(
        nch_2, "M12 (tail mirror)", gm_id_tail, I_M12, L_tail, vds_tail, vsb=0.0
    )

    return results


def print_stage1_results(results, VDDH, Iref):
    print(
        f"=== Stage-1 folded cascode sizing "
        f"(VDDH = {VDDH:.2f} V, Iref = {Iref*1e6:.1f} µA) ==="
    )
    for key in sorted(results.keys(), key=lambda k: int(k[1:])):  # sort by M#
        r = results[key]
        print(
            f"{r['name']:18s}  W = {r['W']:.3f}  L = {r['L']:.3f}  "
            f"Id = {r['Id']*1e6:6.2f} µA  gm = {r['gm']*1e6:6.2f} µS  "
            f"ro = {r['ro']:.1f} Ω  VGS = {r['vgs']:.3f} V  VOV = {r['vov']:.3f} V"
        )


# ---------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------


def main():
    # Load gm/Id tables
    nch_1 = importdata("nch_1v.mat")
    nch_2 = importdata("nch_2v.mat")
    pch_1 = importdata("pch_1v.mat")
    pch_2 = importdata("pch_2v.mat")

    # System-level parameters from week_1.py
    params = {
        "VDDH": 1.8,
        "Iref": 19.2e-6,
        # From your week_1 script (adjust if you re-run it):
        "gm1_target": 1.15e-4,   # S
        "gm2_target": 1.859e-3,  # S
        # gm/Id choices (1/V) – tweak as you like
        "gm_id_in": 10.0,
        "gm_id_tail": 17.0,
        "gm_id_fold": 12.0,
        "gm_id_casc_n": 10.0,
        "gm_id_casc_p": 10.0,
        "gm_id_load_p": 8.0,
        # Length targets in µm (will be snapped to LUT grids)
        "L_in": 0.12,
        "L_tail": 0.50,
        "L_fold": 0.12,
        "L_casc_n": 0.12,
        "L_casc_p": 0.12,
        "L_load_p": 0.12,
        # Bias VDS guesses for sizing (rough but OK for gm/Id design)
        "vds_in": 0.3,
        "vds_tail": 0.3,
        "vds_fold": 0.3,
        "vds_casc_n": 0.3,
        "vds_casc_p": 0.3,
        "vds_load_p": 0.3,
    }

    results = size_stage1_folded(params, nch_1, nch_2, pch_1, pch_2)
    print_stage1_results(results, params["VDDH"], params["Iref"])


if __name__ == "__main__":
    main()
