# Pre-publication checklist

- [x] Add valid software citation metadata and the public repository URL.
- [x] Use CDDMamba as the public repository and method name.
- [ ] Confirm the repository-wide license after reviewing the UrbanSSF source
      permissions.
- [ ] Upload only the two final proposed-model checkpoints as GitHub Release
      assets; do not commit checkpoints into Git history.
- [x] Add the checkpoint release URLs to `README.md`.
- [ ] Check that the manuscript tables match every CSV in `results/`.
- [ ] Confirm whether the integrator paragraph accurately reflects the measured
      Euler/RK4 results.
- [ ] Run `python tools/release_audit.py` immediately before the first push.
- [ ] Create a `v1.0.0` release and use its permanent URL in the submission form.

