# week_1.py

import numpy as np
import matplotlib.pyplot as plt


# ------------------------- Part 2a: errors, tau ------------------------- #

def part2a_error_and_tau(total_error=0.002, static_error=0.001):
    """
    2(a): split error into static & dynamic, compute max settling time and tau.

    total_error: ε_total  (≤ 0.2%)
    static_error: ε_s     (we choose 50–50 split, so 0.1%)
    """
    dynamic_error = total_error - static_error  # ε_d

    # Max settling time for full frame:
    # 60 Hz refresh, 272 rows, 340 columns (EE140 column)
    max_settling_time = (1 / 60.0) / (272 * 340)

    # First-order approximation: e^{-t/τ} = ε_d  →  τ = -T_settle / ln(ε_d)
    tau = -max_settling_time / np.log(dynamic_error)

    return {
        "total_error": total_error,
        "static_error": static_error,
        "dynamic_error": dynamic_error,
        "max_settling_time": max_settling_time,
        "tau": tau,
    }


# ---------------- Part 2b: loop gain & open-loop gain ------------------ #

def part2b_loop_gain(params, B=1/3):
    """
    2(b): from static error, compute required loop gain βA_OL(0)
          and open-loop DC gain A_OL(0).

    B is the feedback factor (≈ 1/3 for the LCD driver).
    """
    eps_s = params["static_error"]

    # For unity-step: ε_s ≈ 1 / (1 + β A_OL(0))
    # → βA_OL(0) = 1/ε_s - 1
    LG = 1 / eps_s - 1            # βA_OL(0)
    Aopen = LG / B                # A_OL(0)

    params.update({
        "B": B,
        "LG": LG,
        "Aopen": Aopen,
    })
    return params


# ---------- Part 2c: poles, gm1, gm2, R1, R2, unity BW ----------------- #

def part2c_compensation(params):
    """
    2(c): use tau to get closed-loop BW, then choose wp1, wp2, gm1, gm2,
          R1, R2 according to your design equations.
    """
    tau = params["tau"]
    LG = params["LG"]
    Aopen = params["Aopen"]

    # Closed-loop 3 dB BW (first-order approximation)
    closed_BW = 1 / tau       # rad/s (approx)

    # Dominant pole: ω_p1 ≈ ω_cl / LG
    wp1 = closed_BW / LG

    # Chosen compensation capacitor
    Cc = 1e-12  # 1 pF
    gm1 = Aopen * wp1 * Cc

    # Unity-gain frequency (of loop) from Miller gm1/Cc
    wu = gm1 / Cc

    # Load cap (pixel + explicit CL = 25 pF)
    CL = 25e-12

    # Non-dom pole from desired phase margin
    phase_margin_deg = 60.0
    wp2 = np.tan(np.deg2rad(90 - phase_margin_deg)) * wu

    gm2 = wp2 * (CL + 3e-12)

    R1R2 = Aopen / (gm2 * gm1)

    gm1R1 = 115.0
    R1 = gm1R1 / gm1
    R2 = R1R2 / R1

    params.update({
        "closed_BW": closed_BW,
        "wp1": wp1,
        "Cc": Cc,
        "gm1": gm1,
        "wu": wu,
        "CL": CL,
        "phase_margin_deg": phase_margin_deg,
        "wp2": wp2,
        "gm2": gm2,
        "R1R2": R1R2,
        "R1": R1,
        "R2": R2,
    })
    return params


# ---------------------- Bode plots for Week 1 -------------------------- #

def make_bode_plots(params):
    """
    Plot:
      - |A_OL(jω)| in dB
      - |β A_OL(jω)| in dB

    Using a simple two-pole open-loop model:
        A_OL(s) = Aopen / ((1 + s/ω_p1) (1 + s/ω_p2))
    """
    Aopen = params["Aopen"]
    B = params["B"]
    wp1 = params["wp1"]
    wp2 = params["wp2"]

    # Frequency sweep (Hz)
    f = np.logspace(2, 9, 600)  # 10^2 ... 10^9 Hz
    w = 2 * np.pi * f
    s = 1j * w

    # Open-loop transfer function
    AOL = Aopen / ((1 + s / wp1) * (1 + s / wp2))
    L = B * AOL  # loop gain βA_OL

    mag_AOL_dB = 20 * np.log10(np.abs(AOL))
    mag_L_dB = 20 * np.log10(np.abs(L))

    plt.figure()
    plt.semilogx(f, mag_AOL_dB, label="|A_OL(jω)|")
    plt.semilogx(f, mag_L_dB, "--", label="|βA_OL(jω)|")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.title("Week 1: Open-loop and Loop Gain Bode Magnitude")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------- Init  ----------------------------- #

def init_week1():
    """
    Driver function: run parts 2(a)–(c), print key values,
    and generate Bode plots.
    """
    params = part2a_error_and_tau()
    params = part2b_loop_gain(params)
    params = part2c_compensation(params)

    print("=== Week 1 System-Level Calculations ===")
    print(f"Total error ε_total      = {params['total_error']:.4f}")
    print(f"Static error ε_s         = {params['static_error']:.4f}")
    print(f"Dynamic error ε_d        = {params['dynamic_error']:.4f}")
    print(f"Max settling time T_settle = {params['max_settling_time']*1e6:.3f} µs")
    print(f"Tau (time constant) τ    = {params['tau']*1e6:.3f} µs\n")

    print(f"Feedback factor B        = {params['B']:.3f}")
    print(f"Loop gain βA_OL(0)       = {params['LG']:.2f}")
    print(f"Open-loop gain A_OL(0)   = {params['Aopen']:.2f}\n")

    print(f"Closed-loop BW (rad/s)   = {params['closed_BW']:.3e}")
    print(f"ω_p1 (dominant pole)     = {params['wp1']:.3e} rad/s")
    print(f"ω_p2 (2nd pole)          = {params['wp2']:.3e} rad/s")
    print(f"gm1                      = {params['gm1']:.3e} S")
    print(f"gm2                      = {params['gm2']:.3e} S")
    print(f"Unity frequency ω_u      = {params['wu']:.3e} rad/s")
    print(f"R1                       = {params['R1']:.3e} Ω")
    print(f"R2                       = {params['R2']:.3e} Ω")

    prod = params["gm1"] * params["gm2"] * params["R1"] * params["R2"]
    print(f"\nSanity: gm1*gm2*R1*R2    = {prod:.2f}")

    # Bode plots for A_OL and βA_OL
    make_bode_plots(params)


if __name__ == "__main__":
    init_week1()
