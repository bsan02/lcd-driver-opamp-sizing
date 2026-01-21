# Op-Amp System for LCD Driver (EE140) — Sizing & Analysis Scripts

This repository contains the sizing/analysis scripts and design artifacts for an LCD pixel-driver operational amplifier designed in a 45nm CMOS educational PDK (gpdk045). The amplifier is optimized for fast, accurate pixel updates in an active-matrix LCD load model (≈25 pF pixel capacitance behind ≈1 kΩ series resistance) and a closed-loop gain of 2.

The project combines **analog IC design** with a **software-style optimization workflow**: lookup tables (LUTs), interpolation, parameter sweeps, and constraint checks to converge on device sizes and bias points that meet gain/stability/power targets.

## What I Built

**Circuit topology**
- **Two-stage, single-ended op-amp LCD driver**
  - Stage 1: **Telescopic cascode** front-end (high intrinsic gain, bandwidth) that converts differential inputs (Vinp/Vinn) to a single-ended internal node.
  - Stage 2: **Class-AB** output stage sized for large transient drive current into the pixel load with low quiescent power.
- **Miller compensation** using a MIM capacitor to establish a dominant pole and improve stability. 

**Supplies / devices**
- Designed around **VDDH = 1.8 V** and 2V devices (nmos2v/pmos2v) in the stack, with biasing arranged to keep devices in saturation while preserving output swing headroom. 

## Software Workflow 

Analog design becomes tractable when you treat it like constrained optimization:

1. **Translate system specs → amplifier constraints**
   - Required loop gain to meet ≤0.2% total settling error, and closed-loop bandwidth implied by the settling-time target. 
2. **Use gm/Id + LUT-driven sizing**
   - LUTs map device operating points (gm/Id, VGS, VDS, ID) to sizing decisions (W/L).
   - Iterated on gm/Id to balance gain vs parasitics/stability (backing off from overly aggressive gm/Id that inflated capacitances and reduced phase margin). 
3. **Run parameter sweeps + constraint checks**
   - Scripts identify violations (headroom, stability, power) and guide the next sizing/bias iteration.
4. **Validate in Cadence**
   - AC loop gain, phase margin, CMRR/PSRR, and transient settling vs the LCD load model. 

## Key Results (Measured in Simulation)

Final design meets the LCD driver requirements with margin:

- **Open-loop gain:** ~73 dB (≈4.37 kV/V)   
- **Phase margin:** ~66°   
- **Power:** ~0.81 mW  
- **Settling time (1.4 V step):** ~520 ns 
- **Output swing:** ~0.051–1.7 V (≈1.65 Vpp) 
- **DC CMRR:** ~87 dB 
- **DC PSRR @ VDDH:** ~51.7 dB  



## Design Notes / Tradeoffs

- Telescopic cascode provides excellent gain and low-frequency rejection (CMRR/PSRR), but constrains headroom—bias placement is critical to maintain saturation while hitting output swing. 
- Class-AB output stage + Miller compensation balances fast transient drive against stability (phase margin), with a deliberate tradeoff between bandwidth and well-damped settling. 

## Context

This work was developed for UC Berkeley EE140/EE240A (Fall 2025) LCD driver amplifier design project. 
