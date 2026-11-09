DarkMind Vocoder V2 – Reference Implementation

A stateful, real‑time capable vocoder with a continuously morphing seasonal engine, a physical pendulum‑driven carrier, and proper unvoiced excitation detection. This Python module serves as an executable DSP reference model for the eventual DarkMind VST3/AU plugin.

---

Features

· Invisible block boundaries – all filters, envelopes, pendulum, and phase retain state across process_block() calls, so arbitrary buffer sizes are handled seamlessly.
· Seasonal morphing engine – continuously interpolates between Spring, Summer, Autumn, and Winter, affecting pitch, chaos, trapezoid shape, drive, spectral tilt, sensitivity, and more. Parameters evolve per‑sample within a block for smooth automation.
· Physical carrier – a simplified double pendulum drives the oscillator frequency and a soft, asymmetric trapezoid waveform. Symplectic Euler integration keeps the motion stable.
· Stateful SOS filters – uses scipy.signal.sosfilt with persistent zi state for both analysis and carrier filterbanks.
· Unvoiced excitation – estimates high‑frequency/transient ratio and injects shaped noise for consonants without turning vowels into noise.
· Band normalisation – maintains stable output levels regardless of active band count.
· Built‑in validation suite – proves block‑boundary continuity, filter tail continuation, envelope state persistence, silence handling, and extreme parameter safety.

---

Installation

```bash
pip install numpy scipy numba
```

Numba is optional – if not installed, the envelope follower falls back to a pure‑Python loop.

---

Quick Start

```python
from darkmind_vocoder_v2 import DarkMindVocoder

# Create vocoder
vocoder = DarkMindVocoder(sample_rate=48000, num_bands=32, f_min=70, f_max=12000)
vocoder.prepare()

# Process an audio block (any length)
input_block = ...  # numpy array of float64, shape (N,)
output_block = vocoder.process_block(input_block)

# Change season position (0=Spring, 1=Summer, 2=Autumn, 3=Winter)
vocoder.set_season_position(1.5)   # halfway between Summer and Autumn

# Adjust parameters
vocoder.set_parameters(attack_ms=10.0, release_ms=150.0, drive=0.25)
```

---

Validation

Run the built‑in tests to verify the implementation's correctness:

```python
python darkmind_vocoder_v2.py
```

The validation suite checks:

1. Block continuity – processing a signal in random block sizes yields the same output as processing it whole.
2. Filter state – an impulse tail continues across block boundaries.
3. Envelope state – a sustained tone doesn't cause envelope resets.
4. Silence – zero input produces (near) zero output.
5. Extreme parameters – no NaNs or infinities across all seasons.

All tests should pass (some may print warnings for informational purposes).

---

Architecture Overview

```
┌─────────────────────────┐
│         INPUT           │
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│   MODULATOR FILTERBANK  │   Stateful SOS filters
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│   ENVELOPE FOLLOWERS    │   Attack/release per band
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│   UNVOICED DETECTOR     │   HF/transient ratio → noise
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│      SEASON ENGINE      │   Continuous Spring→Summer→…
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│   DARKMIND CARRIER      │   Pendulum + phase + trapezoid
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│   CARRIER FILTERBANK    │   Stateful SOS filters
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│   BAND MODULATION       │   carrier × envelope
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│  DRIVE / SATURATION     │   tanh + gain
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│         OUTPUT          │
└─────────────────────────┘
```

Every component maintains its own persistent state across blocks.

---

Parameters

Global Parameters (DarkMindParameters)

Parameter Default Description
base_frequency 110.0 Carrier base frequency (Hz) before seasonal pitch bias
chaos_depth 0.30 Amount of pendulum‑driven frequency modulation
chaos_rate 0.45 Rate of chaos (scales coupling)
damping 0.10 Pendulum damping factor
trapezoid_softness 0.40 Blend between triangle and soft‑tanh shape
trapezoid_asymmetry 0.10 Asymmetry of the waveform
attack_ms 8.0 Envelope attack time (ms)
release_ms 120.0 Envelope release time (ms)
season_position 0.0 Cyclic season position (0=Spring, 1=Summer, 2=Autumn, 3=Winter, 4=Spring)
envelope_sensitivity 1.0 Exponent applied to envelope gain
unvoiced_mix 0.35 Amount of unvoiced excitation mixed in
drive 0.15 Saturation drive amount
output_gain 1.0 Final output gain

Seasonal Parameters (per season)

Each season defines a set of SeasonParameters:

· pitch_bias (semitones)
· chaos_depth, chaos_rate
· trapezoid_softness, trapezoid_asymmetry
· drive
· spectral_tilt_db
· formant_shift (semitones, used for carrier)
· envelope_sensitivity
· unvoiced_mix
· stereo_motion (reserved for future)

The seasonal engine smoothly interpolates these values using a smoothstep curve and applies per‑sample linear interpolation within each block.

---

Performance Notes

· Stateful SOS filtering via scipy.signal.sosfilt is fast and C‑optimised.
· The envelope follower is accelerated with Numba if available; otherwise falls back to a Python loop.
· The carrier uses a per‑sample loop for full parameter modulation – this is the main bottleneck, but for 24–32 bands it's still manageable. If real‑time on low‑end hardware is needed, consider computing carrier parameters at a control rate (e.g., every 64 samples) and interpolating.
· All arrays are pre‑allocated where possible; however, process_block does allocate output buffers internally. For maximum performance, you can pre‑allocate and reuse by modifying the class to accept an output buffer.

---

Known Limitations

· The pendulum model is a simplified double pendulum; it may not exhibit chaotic behaviour for all parameter ranges. An RK4 integration option could improve accuracy.
· The unvoiced detector is a simple high‑frequency energy ratio – it works for basic consonants but could be refined with a proper spectral flux or zero‑crossing rate.
· No stereo processing is implemented yet (stereo_motion is reserved).
· formant_shift is not yet applied to the carrier (though the parameter exists). Future work will use it to shift formant peaks in the carrier spectrum.

---

Future Work

· Add three‑stage envelope (attack/hold/release/tail) for a more expressive DarkMind character.
· Implement RK4 pendulum integration for higher fidelity.
· Apply formant shifting to the carrier filterbank.
· Add stereo widening and spatial motion based on stereo_motion.
· Port the architecture to C++/JUCE for a real plugin.
· Provide a sounddevice‑based live input/output demo.

---

License

This project is released under the MIT License. See LICENSE for details.

---

Acknowledgements

Built as a reference implementation for the DarkMind vocoder concept. Inspired by classic vocoder design and physical‑modeling synthesis.
