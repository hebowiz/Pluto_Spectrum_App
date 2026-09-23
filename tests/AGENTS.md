# Test organization

- Place Python tests in the subsystem they exercise: `common/`, `rtsa/`, `calibration/`, `vsa/{core,bluetooth,dect,adsb}/`, or `vsg/`.
- Put test data under `data/`, separate from Python tests. Group protocol fixtures under `data/fixtures/<protocol>/`; use `general/` for protocol-independent signals and formats.
- Human-readable validation images belong in `../docs/verification/assets/`. Only actual automated-test image inputs belong under `data/expected-images/`.
- Resolve data paths relative to the repository/test tree, not the process working directory. Use explicit protocol paths; do not pick the first filename found by recursive search.
- Before adding a fixture, check whether an existing one covers the case. Record an explicit reason when modifying or regenerating fixtures.
- Never change test conditions, expected values, or algorithms merely to make a structural reorganization pass.
- Do not split large test modules or reconstruct test cases as part of a directory-only cleanup.
- For a separately authorized responsibility split, preserve complete test/helper bodies, decorators, parameter IDs, and setup/teardown. Compare the full collected case set before and after the split.
- Keep same-directory test helper modules small and uniquely named, and import helpers explicitly. Do not import test functions from another test module.
- Run the full suite after broad test-tree moves and compare failures and collected tests with the pre-move baseline.
