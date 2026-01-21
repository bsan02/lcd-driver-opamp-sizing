#!/usr/bin/env python3
"""
Telescopic cascode sizing + bias:

- Loads gm/Id LUTs via look_up.py (nch_2v.mat, pch_2v.mat, etc.)
- "Back-solves" gm/Id that matches your *actual* schematic W (given Id, L, VDS)
- Reports device metrics (W, L, Id, gm, ro, VGS, VOV, gm/Id)
- Computes a consistent DC headroom stack and produces bias voltages:
    * Vin_cm (input common-mode)
    * VBN (NMOS cascode gate bias)
    * VBP_cas (PMOS cascode gate bias)
    * VBP_load (PMOS load/mirror gate bias)
    * VTAIL_bias (tail current source gate bias)

- Top PMOS loads (pmos2v):        L=0.25u, W=249.89u  (M6, M2)
- PMOS cascodes (pmos2v):         L=0.25u, W=249.89u  (M7, M3)
- NMOS cascodes (nmos2v):         L=0.50u, W=16.00u   (M8, M4)
- Input NMOS pair (nmos2v):       L=0.50u, W=13.15u   (M9, M5)
- Bias mirror devices (nmos2v):   L=0.50u, W=0.60u (M1 diode), W=1.10u (M0 mirror)

"""

import numpy as np
from look_up import (
    importdata,
    look_up_basic,
    look_up_vs_gm_id,
    look_up_vgs_vs_gm_id,
)

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def nearest_L(device, L_desired):
    L_grid = np.array(device["L"])
    idx = np.argmin(np.abs(L_grid - L_desired))
    return float(L_grid[idx])


def gm_id_from_W(device, W_target, Id_target, L, vds_opt, vsb=0.0,
                 gm_id_min=4.0, gm_id_max=25.0, npts=800):
  
    gm_id_grid = np.linspace(gm_id_min, gm_id_max, npts)

    idw = np.array([
        look_up_vs_gm_id(device, "ID_W", g, vds=vds_opt, vsb=vsb, l=L)
        for g in gm_id_grid
    ], dtype=float)  # A/um

    W_fit = Id_target / idw
    err = np.abs(W_fit - W_target)
    k = int(np.argmin(err))

    gm_id_best = float(gm_id_grid[k])
    idw_best = float(idw[k])
    W_best = float(W_fit[k])
    err_frac = float((W_best - W_target) / W_target)

    vgs_best = float(
        look_up_vgs_vs_gm_id(device, gm_id_best, vds=vds_opt, vsb=vsb, l=L)
    )

    return gm_id_best, vgs_best, idw_best, W_best, err_frac


def size_device_from_gm_id(device, name, gm_id, Id, L, vds_opt, vsb=0.0):
    """
    Standard size helper: uses gm/Id -> VGS, Id/W -> W, gds/Id -> ro, etc.
    """
    vgs = float(look_up_vgs_vs_gm_id(device, gm_id, vds=vds_opt, vsb=vsb, l=L))

    Id_per_W = float(look_up_vs_gm_id(device, "ID_W", gm_id, vds=vds_opt, vsb=vsb, l=L))
    W = Id / Id_per_W

    gds_per_Id = float(look_up_vs_gm_id(device, "GDS_ID", gm_id, vds=vds_opt, vsb=vsb, l=L))
    ro = 1.0 / (gds_per_Id * Id)

    gm = gm_id * Id
    vt = float(look_up_basic(device, "VT", vgs, vds_opt, vsb, L))
    vov = vgs - vt

    return {
        "name": name,
        "W": float(W),
        "L": float(L),
        "Id": float(Id),
        "gm": float(gm),
        "ro": float(ro),
        "gm_id": float(gm_id),
        "vgs": float(vgs),
        "vt": float(vt),
        "vov": float(vov),
    }


# ---------------------------------------------------------------------
# Stage-1 (Telescopic) sizing + bias
# ---------------------------------------------------------------------

def size_and_bias_telescopic(params, nch_2, pch_2):
    """
    Telescopic stage node order (bottom -> top):
      VSS
      Vtail     : sources of input NMOS (M9/M5)
      N1        : drains of input NMOS / sources of NMOS cascodes (M8/M4)
      Vout_cm   : drains of NMOS cascodes / drains of PMOS cascodes (M7/M3)  (this is your V_stage1)
      Ntop      : sources of PMOS cascodes / drains of PMOS loads (M6/M2)
      VDDH

    Returns:
      results: dict of device summaries
      bias:    dict of node voltages + gate biases
    """
    VSS  = params["VSS"]
    VDDH = params["VDDH"]
    margin = params["Vds_margin"]

    # Currents
    Itail = params["Itail"]          # total tail current
    I_side = Itail / 2.0             # per branch (left/right)

    # Snap L to LUT grid
    L_in   = nearest_L(nch_2, params["L_in"])
    L_ncas = nearest_L(nch_2, params["L_ncas"])
    L_pcas = nearest_L(pch_2, params["L_pcas"])
    L_pload= nearest_L(pch_2, params["L_pload"])
    L_bias = nearest_L(nch_2, params["L_bias"])
    L_tail = nearest_L(nch_2, params["L_tail"])

    # VDS guesses for LUT usage
    vds_in    = params["vds_in"]
    vds_ncas  = params["vds_ncas"]
    vds_pcas  = params["vds_pcas"]
    vds_pload = params["vds_pload"]
    vds_bias  = params["vds_bias"]
    vds_tail  = params["vds_tail"]

    results = {}

    # Input NMOS pair (M9/M5)
    gm_id_in, vgs_in, _, W_fit_in, err_in = gm_id_from_W(
        nch_2, params["W_in"], I_side, L_in, vds_in
    )
    # NMOS cascodes (M8/M4)
    gm_id_ncas, vgs_ncas, _, W_fit_ncas, err_ncas = gm_id_from_W(
        nch_2, params["W_ncas"], I_side, L_ncas, vds_ncas
    )
    # PMOS cascodes (M7/M3)
    gm_id_pcas, vgs_pcas, _, W_fit_pcas, err_pcas = gm_id_from_W(
        pch_2, params["W_pcas"], I_side, L_pcas, vds_pcas
    )
    # PMOS loads (M6/M2)
    gm_id_pload, vgs_pload, _, W_fit_pload, err_pload = gm_id_from_W(
        pch_2, params["W_pload"], I_side, L_pload, vds_pload
    )
    # Bias mirror devices (M1 diode, M0 mirror)
    gm_id_m1, vgs_m1, _, W_fit_m1, err_m1 = gm_id_from_W(
        nch_2, params["W_M1_diode"], params["I_bias_ref"], L_bias, vds_bias
    )
    gm_id_m0, vgs_m0, _, W_fit_m0, err_m0 = gm_id_from_W(
        nch_2, params["W_M0_mirror"], params["I_bias_ref"], L_bias, vds_bias
    )
    # Tail device (if you want it explicit)
    gm_id_tail, vgs_tail, _, W_fit_tail, err_tail = gm_id_from_W(
        nch_2, params["W_tail"], Itail, L_tail, vds_tail
    )

    # -----------------------------------------------------------------
    # 2) Create device summaries (these will be *very close* to your W targets)
    # -----------------------------------------------------------------
    results["M9"] = size_device_from_gm_id(nch_2, "M9 (NMOS in, left)",  gm_id_in,   I_side, L_in,   vds_in)
    results["M5"] = size_device_from_gm_id(nch_2, "M5 (NMOS in, right)", gm_id_in,   I_side, L_in,   vds_in)

    results["M8"] = size_device_from_gm_id(nch_2, "M8 (NMOS cascode, left)",  gm_id_ncas, I_side, L_ncas, vds_ncas)
    results["M4"] = size_device_from_gm_id(nch_2, "M4 (NMOS cascode, right)", gm_id_ncas, I_side, L_ncas, vds_ncas)

    results["M7"] = size_device_from_gm_id(pch_2, "M7 (PMOS cascode, left)",  gm_id_pcas, I_side, L_pcas, vds_pcas)
    results["M3"] = size_device_from_gm_id(pch_2, "M3 (PMOS cascode, right)", gm_id_pcas, I_side, L_pcas, vds_pcas)

    results["M6"] = size_device_from_gm_id(pch_2, "M6 (PMOS load/mirror, left)",  gm_id_pload, I_side, L_pload, vds_pload)
    results["M2"] = size_device_from_gm_id(pch_2, "M2 (PMOS load/mirror, right)", gm_id_pload, I_side, L_pload, vds_pload)

    results["M1"] = size_device_from_gm_id(nch_2, "M1 (NMOS diode bias)", gm_id_m1, params["I_bias_ref"], L_bias, vds_bias)
    results["M0"] = size_device_from_gm_id(nch_2, "M0 (NMOS mirror bias)", gm_id_m0, params["I_bias_ref"], L_bias, vds_bias)

    results["Mtail"] = size_device_from_gm_id(nch_2, "Mtail (tail current source)", gm_id_tail, Itail, L_tail, vds_tail)

    # -----------------------------------------------------------------
    # 3) Bias stack (headroom-consistent) for telescopic
    # -----------------------------------------------------------------
    vov_in   = results["M9"]["vov"]
    vov_ncas = results["M8"]["vov"]
    vov_pcas = results["M7"]["vov"]
    vov_pload= results["M6"]["vov"]
    vov_tail = results["Mtail"]["vov"]

    # Choose a Vtail that keeps tail in saturation
    Vtail_min = VSS + (vov_tail + margin)
    Vtail = max(params["Vtail_guess"], Vtail_min)

    # NMOS side constraints
    N1_min = Vtail + (vov_in + margin)
    Vout_min = N1_min + (vov_ncas + margin)

    # PMOS side constraints:
    # For PMOS cascode: Ntop - Vout >= vov_pcas + margin
    # For PMOS load:    VDDH - Ntop >= vov_pload + margin  -> Ntop <= VDDH - (vov_pload + margin)
    # So Ntop must satisfy: Ntop >= Vout + (vov_pcas + margin) and Ntop <= VDDH - (vov_pload + margin)
    Ntop_max = VDDH - (vov_pload + margin)

    # Pick Ntop (headroom-friendly)
    Ntop = min(params["Ntop_guess"], Ntop_max)

    # Now enforce PMOS cascode sat: Vout <= Ntop - (vov_pcas + margin)
    Vout_max_from_pcas = Ntop - (vov_pcas + margin)

    # Choose Vout within [Vout_min, Vout_max_from_pcas]
    if Vout_min >= Vout_max_from_pcas:
        print("WARNING: Telescopic stack too tall with current VOVs/margin.")
        print(f"  Vout_min = {Vout_min:.3f} V, Vout_max_from_pcas = {Vout_max_from_pcas:.3f} V")
        Vout = 0.5 * (Vout_min + Vout_max_from_pcas)
    else:
        Vout = 0.5 * (Vout_min + Vout_max_from_pcas)

    # Back out N1 from NMOS cascode constraint
    N1 = Vout - (vov_ncas + margin)

    # -----------------------------------------------------------------
    # 4) Gate biases (use VGS for NMOS, VSG for PMOS)
    # -----------------------------------------------------------------
    # NMOS input common-mode
    Vin_cm = Vtail + results["M9"]["vgs"]

    # NMOS cascode gate bias: Vg = Vs + VGS (source is N1)
    VBN = N1 + results["M8"]["vgs"]

    # PMOS cascode gate bias: Vg = Vs - VSG (here vgs stored is VSG magnitude from LUT)
    VBP_cas = Ntop - results["M7"]["vgs"]

    # PMOS load gate bias: source at VDDH -> Vg = VDDH - VSG
    VBP_load = VDDH - results["M6"]["vgs"]

    # Tail bias: source at VSS -> Vg = VSS + VGS
    VTAIL_bias = VSS + results["Mtail"]["vgs"]

    bias = {
        "VSS": VSS,
        "VDDH": VDDH,
        "Vtail": Vtail,
        "N1": N1,
        "Vout_cm": Vout,     # this is V_stage1
        "Ntop": Ntop,
        "Vin_cm": Vin_cm,
        "VBN_ncas": VBN,
        "VBP_pcas": VBP_cas,
        "VBP_pload": VBP_load,
        "VTAIL_bias": VTAIL_bias,
        "match_errors": {
            "in_W_err_frac": err_in,
            "ncas_W_err_frac": err_ncas,
            "pcas_W_err_frac": err_pcas,
            "pload_W_err_frac": err_pload,
            "M1_W_err_frac": err_m1,
            "M0_W_err_frac": err_m0,
            "tail_W_err_frac": err_tail,
        }
    }

    return results, bias

#----------------------------------------------------------------

def print_results(results, bias):
    print("=== Stage-1 Telescopic Cascode – Device Summary ===")
    order = ["M1","M0","Mtail","M9","M5","M8","M4","M7","M3","M6","M2"]
    for k in order:
        r = results[k]
        print(
            f"{r['name']:28s}  "
            f"W={r['W']:.3f}um  L={r['L']:.3f}um  "
            f"Id={r['Id']*1e6:7.3f}uA  gm/Id={r['gm_id']:.2f}  "
            f"VGS(or VSG)={r['vgs']:.3f}V  VOV={r['vov']:.3f}V  "
            f"gm={r['gm']*1e6:7.2f}uS  ro={r['ro']:.1f}ohm"
        )

    print("\n=== Node Voltages (DC, approx) ===")
    print(f"VSS      = {bias['VSS']:.3f} V")
    print(f"Vtail    = {bias['Vtail']:.3f} V  (sources of input NMOS)")
    print(f"N1       = {bias['N1']:.3f} V  (between input NMOS and NMOS cascodes)")
    print(f"V_stage1 = {bias['Vout_cm']:.3f} V  (single-ended stage-1 output node)")
    print(f"Ntop     = {bias['Ntop']:.3f} V  (between PMOS cascodes and PMOS loads)")
    print(f"VDDH     = {bias['VDDH']:.3f} V")

    print("\n=== Gate Bias Voltages to use in schematic ===")
    print(f"Vin_cm (apply to Vinp/Vinn CM) = {bias['Vin_cm']:.3f} V")
    print(f"VBN (NMOS cascode gates)       = {bias['VBN_ncas']:.3f} V")
    print(f"VBP_cas (PMOS cascode gates)   = {bias['VBP_pcas']:.3f} V")
    print(f"VBP_load (PMOS load/mirror)    = {bias['VBP_pload']:.3f} V")
    print(f"VTAIL_bias (tail gate)         = {bias['VTAIL_bias']:.3f} V")

    print("\n=== W-match fractional errors (W_fit - W_target)/W_target ===")
    for k,v in bias["match_errors"].items():
        print(f"{k:18s}: {v:+.4f}")


# ---------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------

def main():
    # Load gm/Id tables (2V devices for  stage-1 stack off VDDH=1.8V)
    nch_2 = importdata("nch_2v.mat")
    pch_2 = importdata("pch_2v.mat")

    # --- Parameters: set to match your schematic + reasonable LUT bias points ---
    params = {
        "VSS": 0.0,
        "VDDH": 1.8,

        # From operating point readouts: branch currents ~35uA/side
        "Itail": 70e-6,        # total tail current (A)  -> ~35uA/side
        "I_bias_ref": 6e-6,    # bias mirror reference current (A) (M1/M0 block)

        # schematic W/L targets (um)
        "L_in":   0.50,  "W_in":   13.15,     # M9/M5
        "L_ncas": 0.50,  "W_ncas": 16.00,     # M8/M4
        "L_pcas": 0.25,  "W_pcas": 249.89,    # M7/M3
        "L_pload":0.25,  "W_pload":249.89,    # M6/M2

        # Bias mirror devices (left block)
        "L_bias": 0.50,  "W_M1_diode": 0.60,  # M1
        "W_M0_mirror": 1.10,                 # M0

        # Tail device (not clearly visible in screenshot): choose something reasonable
        "L_tail": 0.50,  "W_tail": 1.50,

        # LUT VDS guesses (keep these close to Cadence DC operating points)
        "vds_in":    0.24,   # input NMOS VDS ~ 0.24V 
        "vds_ncas":  0.42,   # NMOS cascode VDS ~ 0.42V-ish
        "vds_pcas":  0.43,   # PMOS cascode VSD magnitude ~0.43V-ish (use as "vds" to LUT)
        "vds_pload": 0.43,   # PMOS load VSD magnitude ~0.43V-ish
        "vds_bias":  0.59,   # bias mirror devices around ~0.59V
        "vds_tail":  0.44,   # tail device VDS guess

        # Headroom design knobs
        "Vds_margin": 0.05,
        "Vtail_guess": 0.20,   # start near 0.2V 
        "Ntop_guess":  1.37,  
    }

    results, bias = size_and_bias_telescopic(params, nch_2, pch_2)
    print_results(results, bias)


if __name__ == "__main__":
    main()
