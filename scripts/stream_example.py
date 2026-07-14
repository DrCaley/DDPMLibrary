"""Minimal StreamDDPM usage example — the same contract as DDPM / VCNN.

    mean, uncertainty = StreamDDPM(device="auto").predict(observations, priors)

The stream model is additionally conditioned on the ocean state ~13 h and ~25 h
earlier (temporal priors). Pass them as two fields shaped like the output,
(44, 94, 2) in m/s. Here we use zeros for illustration (a warning is emitted);
in real use, supply the actual earlier fields for best quality.
"""
import numpy as np

from ddpm_library import StreamDDPM


def main():
    from ddpm_library.config import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX

    rng = np.random.default_rng(0)
    observations = [
        (rng.uniform(LAT_MIN, LAT_MAX), rng.uniform(LON_MIN, LON_MAX),
         1_700_000_000.0, rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2))
        for _ in range(40)
    ]

    # Two prior fields (lags 13 h, 25 h), each (44, 94, 2) m/s. Use real earlier
    # fields in practice; zeros here just to show the call signature.
    priors = [np.zeros((44, 94, 2), dtype=np.float32) for _ in range(2)]

    model = StreamDDPM(device="auto")

    # Single best-estimate field (uncertainty is zeros, like DDPM/VCNN).
    mean, uncertainty = model.predict(observations, priors)
    print("single field :", mean.shape, "m/s, uncertainty all-zero:",
          bool(np.all(uncertainty == 0)))

    # Ensemble → real per-cell uncertainty.
    mean40, unc40 = model.predict(observations, priors, n_draws=40)
    print("ensemble mean:", mean40.shape,
          "| mean uncertainty (m/s): %.4f" % float(np.nanmean(unc40)))


if __name__ == "__main__":
    main()
