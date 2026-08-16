# Checkpoint location

The trained weights for this model are **not duplicated here**. They live in the
installable package, which is the copy the library actually loads:

    src/ddpm_library/assets/repaint_timecond_weights.pt

They were byte-identical to the copy that used to sit in this folder
(md5 d2f837e9a1a2a5452b11fffb3849e260), so the duplicate was removed to keep the
repository from carrying the same 171 MB twice.

To run the standalone scripts in the parent folder, point them at that path, e.g.

    python run_mcg_dps_z004.py --checkpoint ../../src/ddpm_library/assets/repaint_timecond_weights.pt

Or from Python:

    from ddpm_library.config import REPAINT_WEIGHTS_PATH

Note: `results/uncertainty_validation_time_conditioned_100seeds/fields.pt` (67 MB
of saved ensemble output) was also removed — it is regenerable evaluation output,
not an input. `results.csv` and `summary.txt` are kept.
