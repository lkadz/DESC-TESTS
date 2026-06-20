# Benchmark configurations

Each configuration is a subdirectory containing the DESC h5 files for one
stellarator case. Select it with `--config <name>` in `benchmark_desc_bfield.py`
and `submit_desc_bfield_scan.py`.

## Layout

```
configurations/
  precise_QA/
    eq.h5       ← DESC equilibrium (required)
    coils.h5    ← DESC coil / external field (optional)
  W7-X/
    eq.h5
    coils.h5
```

The script errors with a clear message if `eq.h5` is missing. `coils.h5` is
optional — omitting it requires `--allow-missing-coils`.

## Populating the files (on the cluster)

```bash
# precise_QA (default)
cp /path/to/DESC/desc/examples/precise_QA_output.h5 \
   benchmarks/configurations/precise_QA/eq.h5
cp /path/to/DESC/tests/inputs/precise_QA_helical_coils.h5 \
   benchmarks/configurations/precise_QA/coils.h5

# W7-X
cp /path/to/DESC/desc/examples/W7-X_output.h5 \
   benchmarks/configurations/W7-X/eq.h5
cp /path/to/W7-X_coils.h5 \
   benchmarks/configurations/W7-X/coils.h5
```

## Adding a new configuration

```bash
mkdir benchmarks/configurations/<name>
cp /path/to/my_eq.h5   benchmarks/configurations/<name>/eq.h5
cp /path/to/my_coil.h5 benchmarks/configurations/<name>/coils.h5
```

That's it — it is immediately selectable via `--config <name>`. The other
benchmark parameters (grid size, padding, eps, etc.) are controlled by
command-line flags; see `python benchmarks/benchmark_desc_bfield.py --help`.
