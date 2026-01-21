#!/usr/bin/env python3
"""
Stage-1 folded cascode sizing + bias helper for EE140 LCD driver project.

- Uses gm/Id lookup tables (nch_1v.mat, nch_2v.mat, pch_1v.mat, pch_2v.mat)
  via look_up.py
- Sizes M1–M12 (folded cascode, single-ended).
- Computes a consistent DC bias stack that runs off VDDH = 1.8 V.
- Outputs:
    * Node voltages (tail, Nd, Nn, Vout, Nt, VDDH)
    * Gate bias voltages Vbias1..4, Vbias_tail
    * Device-level summary (W, L, Id, gm, ro, VGS, VOV)

IMPORTANT DEVICE-TYPE GUIDELINE FOR YOUR DESIGN:
  - M1, M2 (input NMOS)      : ***nmos2v***
  - M3, M4 (bottom cascodes) : nmos1v or nmos2v (terminals < ~1.1 V)
  - M5, M6 (folded NMOS)     : nmos1v or nmos2v (terminals < ~1.1 V)
  - M7–M10 (top PMOS stack)  : ***pmos2v*** (they see VDDH = 1.8 V)
  - M11, M12 (tail + mirror) : nmos2v (safest)

You will still confirm all |VGS|, |VDS| < 1.8 V in Cadence.
"""

import numpy as np
from look_up import (
    importdata,
    look_up_basic,
    look_up_vs_gm_id,
    look_up_vgs_vs_gm_id,
)

# ---------------------------------------------------------------------
# Helper: snap L to closest L in LUT
# ---------------------------------------------------------------------


def nearest_L(device, L_desired):
    L_grid = np.array(device["L"])
    idx = np.argmin(np.abs(L_grid - L_desired))
    return float(L_grid[idx])


# ---------------------------------------------------------------------
# Helper: size one MOS using gm/Id
# ---------------------------------------------------------------------


def size_device(device, name, gm_id_target, I_branch, L, vds_opt, vsb=0.0):
    """
    device   : LUT dict (nch_2, pch_2, nch_1, pch_1)
    name     : for printing only
    gm_id_target : gm/Id target (1/V)
    I_branch : drain current (A)
    L        : channel length (um, same units as LUT)
    vds_opt  : bias VDS used for lookup (V)
    vsb      : VSB (V)

    Returns dict with W, L, Id, gm, ro, VGS, VOV, VT.
    """

    # 1) VGS from gm/Id
    vgs = float(
        look_up_vgs_vs_gm_id(
            device,
            gm_id_target,
            vds=vds_opt,
            vsb=vsb,
            l=L,
        )
    )

    # 2) Id/W from gm/Id
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

    W = I_branch / Id_per_W  # µm

    # 3) gds/Id → ro
    gds_per_Id = float(
        look_up_vs_gm_id(
            device,
            "GDS_ID",
            gm_id_target,
            vds=vds_opt,
            vsb=vsb,
            l=L,
        )
    )  # (S/A) = 1/V
    ro = 1.0 / (gds_per_Id * I_branch)

    # 4) gm from gm/Id
    gm = gm_id_target * I_branch

    # 5) VT, VOV
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
# Sizing + biasing of the folded cascode stack
# ---------------------------------------------------------------------


def size_and_bias_stage1(params, nch_1, nch_2, pch_1, pch_2):
    """
    Returns:
        results : dict of device sizing (M1..M12)
        bias    : dict with node voltages and gate biases
    """
    VDDH = params["VDDH"]
    VSS = params["VSS"]
    Iref = params["Iref"]
    margin = params["Vds_margin"]

    # gm/Id targets
    gm_id_in = params["gm_id_in"]
    gm_id_tail = params["gm_id_tail"]
    gm_id_fold = params["gm_id_fold"]
    gm_id_casc_n = params["gm_id_casc_n"]
    gm_id_casc_p = params["gm_id_casc_p"]
    gm_id_load_p = params["gm_id_load_p"]

    # Length targets (µm), snapped to LUT grid
    L_in = nearest_L(nch_2, params["L_in"])
    L_tail = nearest_L(nch_2, params["L_tail"])
    L_fold = nearest_L(nch_2, params["L_fold"])
    L_casc_n = nearest_L(nch_2, params["L_casc_n"])
    L_casc_p = nearest_L(pch_2, params["L_casc_p"])
    L_load_p = nearest_L(pch_2, params["L_load_p"])

    # Approximate VDS biases for LUT
    vds_in = params["vds_in"]
    vds_tail = params["vds_tail"]
    vds_fold = params["vds_fold"]
    vds_casc_n = params["vds_casc_n"]
    vds_casc_p = params["vds_casc_p"]
    vds_load_p = params["vds_load_p"]

    # -----------------------------------------------------------------
    # Branch currents
    # -----------------------------------------------------------------
    Itail = Iref             # total tail
    I_side = Itail / 2.0     # per side (M1, M2)
    I_M1 = I_side
    I_M2 = I_side
    I_M5 = I_side
    I_M6 = I_side
    I_M3 = I_side
    I_M4 = I_side
    I_M7 = I_side
    I_M8 = I_side
    I_M9 = I_side
    I_M10 = I_side
    I_M11 = Itail
    I_M12 = Itail

    results = {}

    # Input NMOS pair (nmos2v)
    results["M1"] = size_device(
        nch_2, "M1 (in+)", gm_id_in, I_M1, L_in, vds_in, vsb=0.0
    )
    results["M2"] = size_device(
        nch_2, "M2 (in-)", gm_id_in, I_M2, L_in, vds_in, vsb=0.0
    )

    # Bottom cascodes (NMOS)
    results["M3"] = size_device(
        nch_2, "M3 (bottom cascode)", gm_id_casc_n, I_M3, L_casc_n, vds_casc_n, vsb=0.0
    )
    results["M4"] = size_device(
        nch_2, "M4 (bottom cascode)", gm_id_casc_n, I_M4, L_casc_n, vds_casc_n, vsb=0.0
    )

    # Folded NMOS
    results["M5"] = size_device(
        nch_2, "M5 (folded)", gm_id_fold, I_M5, L_fold, vds_fold, vsb=0.0
    )
    results["M6"] = size_device(
        nch_2, "M6 (folded)", gm_id_fold, I_M6, L_fold, vds_fold, vsb=0.0
    )

    # PMOS cascodes (top middle)
    results["M7"] = size_device(
        pch_2,
        "M7 (PMOS cascode)",
        gm_id_casc_p,
        I_M7,
        L_casc_p,
        vds_casc_p,
        vsb=0.0,
    )
    results["M8"] = size_device(
        pch_2,
        "M8 (PMOS cascode)",
        gm_id_casc_p,
        I_M8,
        L_casc_p,
        vds_casc_p,
        vsb=0.0,
    )

    # PMOS loads (very top, to VDDH)
    results["M9"] = size_device(
        pch_2,
        "M9 (PMOS load)",
        gm_id_load_p,
        I_M9,
        L_load_p,
        vds_load_p,
        vsb=0.0,
    )
    results["M10"] = size_device(
        pch_2,
        "M10 (PMOS load)",
        gm_id_load_p,
        I_M10,
        L_load_p,
        vds_load_p,
        vsb=0.0,
    )

    # Tail + mirror (NMOS)
    results["M11"] = size_device(
        nch_2, "M11 (tail)", gm_id_tail, I_M11, L_tail, vds_tail, vsb=0.0
    )
    results["M12"] = size_device(
        nch_2, "M12 (tail mirror)", gm_id_tail, I_M12, L_tail, vds_tail, vsb=0.0
    )

    # -----------------------------------------------------------------
    # Compute a self-consistent stack of node voltages
    #
    # Node order (bottom to top):
    #   VSS (0)
    #   Vtail     : sources of M1/M2
    #   Nd        : drains of M1/M2, sources of M3/M4
    #   Nn        : drains of M3/M4, sources of M5/M6
    #   Vout_cm   : drains of M5/M6, drains of M7/M8 (output node)
    #   Nt        : sources of M7/M8, drains of M9/M10
    #   VDDH
    #
    # Saturation constraints (with margin):
    #   Nd    >= Vtail   + vov1 + margin
    #   Nn    >= Nd      + vov3 + margin
    #   Vout  >= Nn      + vov5 + margin
    #   Nt    >= Vout    + vov7 + margin
    #   Nt    <= VDDH - (vov9 + margin)
    #
    # We choose Nt in its allowed range, then back-calculate others.
    # -----------------------------------------------------------------

    vov1 = results["M1"]["vov"]
    vov3 = results["M3"]["vov"]
    vov5 = results["M5"]["vov"]
    vov7 = results["M7"]["vov"]
    vov9 = results["M9"]["vov"]

    # Start with a conservative minimum stack from bottom, assuming small Vtail
    Vtail_guess = params["Vtail_guess"]

    Nd_min = Vtail_guess + vov1 + margin
    Nn_min = Nd_min + vov3 + margin
    Vout_min = Nn_min + vov5 + margin
    Nt_min = Vout_min + vov7 + margin
    Nt_max = VDDH - (vov9 + margin)

    if Nt_min >= Nt_max:
        print("WARNING: Stack too tall for VDDH with current VOVs + margin.")
        print(f"  Nt_min = {Nt_min:.3f} V, Nt_max = {Nt_max:.3f} V")
        print("  Consider reducing currents (higher gm/Id) or Vds_margin.")
        Nt = (Nt_min + Nt_max) / 2.0
    else:
        # Choose Nt in the middle of allowed interval for headroom
        Nt = 0.5 * (Nt_min + Nt_max)

    # Now back-calculate the consistent node voltages
    Vout = Nt - (vov7 + margin)
    Nn = Vout - (vov5 + margin)
    Nd = Nn - (vov3 + margin)
    Vtail = Nd - (vov1 + margin)

    # Tail device saturation check
    vov_tail = results["M11"]["vov"]
    if (Vtail - VSS) < (vov_tail + margin):
        print("WARNING: Tail device may not be well in saturation.")
        print(
            f"  Vtail={Vtail:.3f} V, required >= {vov_tail + margin:.3f} V "
            "(VOV_tail + margin)"
        )

    # -----------------------------------------------------------------
    # Gate bias voltages from VGS (NMOS) / VSG (PMOS)
    # -----------------------------------------------------------------
    vgs1 = results["M1"]["vgs"]
    vgs3 = results["M3"]["vgs"]
    vgs5 = results["M5"]["vgs"]
    vgs7 = results["M7"]["vgs"]  # treat as VSG
    vgs9 = results["M9"]["vgs"]  # treat as VSG
    vgs_tail = results["M11"]["vgs"]

    # Input common-mode (for both Vin+ and Vin-)
    Vin_cm = Vtail + vgs1

    # NMOS cascode bias (gate of M3/M4)
    Vbias1 = Nd + vgs3

    # Folded NMOS bias (gate of M5/M6)
    Vbias2 = Nn + vgs5

    # PMOS cascode bias (gate of M7/M8); VSG = vgs7, so Vg = Vs - VSG
    Vbias3 = Nt - vgs7

    # PMOS load bias (gate of M9/M10); Vs = VDDH
    Vbias4 = VDDH - vgs9

    # Tail bias (gate of M11/M12); source at VSS
    Vbias_tail = VSS + vgs_tail

    bias = {
        "Vtail": Vtail,
        "Nd": Nd,
        "Nn": Nn,
        "Vout_cm": Vout,
        "Nt": Nt,
        "VDDH": VDDH,
        "Vin_cm": Vin_cm,
        "Vbias1_bottom_casc_n": Vbias1,
        "Vbias2_fold_n": Vbias2,
        "Vbias3_casc_p": Vbias3,
        "Vbias4_load_p": Vbias4,
        "Vbias_tail": Vbias_tail,
    }

    return results, bias


# ---------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------


def print_results(results, bias):
    print("=== Stage-1 Folded Cascode – Device Summary ===")
    for key in sorted(results.keys(), key=lambda k: int(k[1:])):  # sort by M#
        r = results[key]
        print(
            f"{r['name']:18s}  "
            f"W = {r['W']:.3f} um  L = {r['L']:.3f} um  "
            f"Id = {r['Id']*1e6:6.2f} uA  gm = {r['gm']*1e6:6.2f} uS  "
            f"ro = {r['ro']:.1f} ohm  VGS = {r['vgs']:.3f} V  VOV = {r['vov']:.3f} V"
        )

    print("\n=== Node Voltages (DC, approx) ===")
    print(f"VSS      = 0.000 V")
    print(f"Vtail    = {bias['Vtail']:.3f} V  (sources of M1/M2)")
    print(f"Nd       = {bias['Nd']:.3f} V  (drains M1/M2, sources M3/M4)")
    print(f"Nn       = {bias['Nn']:.3f} V  (drains M3/M4, sources M5/M6)")
    print(f"Vout_cm  = {bias['Vout_cm']:.3f} V  (output node)")
    print(f"Nt       = {bias['Nt']:.3f} V  (sources of M7/M8, drains M9/M10)")
    print(f"VDDH     = {bias['VDDH']:.3f} V")

    print("\n=== Gate Bias Voltages ===")
    print(f"Vin_cm (for Vin+, Vin-)         = {bias['Vin_cm']:.3f} V")
    print(f"Vbias1 – bottom NMOS cascodes  = {bias['Vbias1_bottom_casc_n']:.3f} V")
    print(f"Vbias2 – folded NMOS (M5/M6)   = {bias['Vbias2_fold_n']:.3f} V")
    print(f"Vbias3 – PMOS cascodes (M7/M8) = {bias['Vbias3_casc_p']:.3f} V")
    print(f"Vbias4 – PMOS loads (M9/M10)   = {bias['Vbias4_load_p']:.3f} V")
    print(f"Vbias_tail – tail source (M11) = {bias['Vbias_tail']:.3f} V")
    print("\nUse these as the DC values for V1..V4 (and tail) in your Cadence schematic.")


# ---------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------


def main():
    # Load gm/Id tables
    nch_1 = importdata("nch_1v.mat")
    nch_2 = importdata("nch_2v.mat")
    pch_1 = importdata("pch_1v.mat")
    pch_2 = importdata("pch_2v.mat")

    # System-level and sizing parameters
    params = {
        "VSS": 0.0,
        "VDDH": 1.8,
        "Iref": 19.2e-6,  # tail current
        # gm/Id choices (1/V) – tweak as desired
        "gm_id_in": 10.0,
        "gm_id_tail": 17.0,
        "gm_id_fold": 12.0,
        "gm_id_casc_n": 10.0,
        "gm_id_casc_p": 10.0,
        "gm_id_load_p": 8.0,
        # Length targets (um) – will be snapped to LUT grid
        "L_in": 0.12,
        "L_tail": 0.50,
        "L_fold": 0.12,
        "L_casc_n": 0.12,
        "L_casc_p": 0.12,
        "L_load_p": 0.12,
        # VDS guesses for LUT (V)
        "vds_in": 0.3,
        "vds_tail": 0.3,
        "vds_fold": 0.3,
        "vds_casc_n": 0.3,
        "vds_casc_p": 0.3,
        "vds_load_p": 0.3,
        # Stack design
        "Vds_margin": 0.05,   # extra headroom beyond VOV
        "Vtail_guess": 0.20,  # initial guess for Vtail (bottom source node)
    }

    results, bias = size_and_bias_stage1(params, nch_1, nch_2, pch_1, pch_2)
    print_results(results, bias)


if __name__ == "__main__":
    main()
