# TT‑Metal Profiler Guide

> **Quick start**: Build with profiling enabled, run your experiment with a single command, then open the companion notebook to explore the generated `.csv`.

---

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Build & Install](#build--install)
4. [Running the Profiler](#running-the-profiler)
5. [Output Artifacts](#output-artifacts)
6. [Analysing Results](#analysing-results)
7. [Troubleshooting & FAQ](#troubleshooting--faq)

---

## Features

* **One‑line build** with all profiler hooks enabled.
* **Environment‑variable toggles**—no code changes required.
* Generates a **single comma‑separated log (`*.csv`)** ready for pandas / Excel.
* Companion **Jupyter Notebook** for rich visualisation.

---

## Prerequisites

Same as tt-metal


---

## Build & Install

```bash
./build_metal.sh -b Release \
                 --enable-profile \
                 --build-tt-train
```

The flags do the following:

| Flag               | Purpose                                                   |
| ------------------ | --------------------------------------------------------- |
| `-b Release`       | Compile with full optimisation.                           |
| `--enable-profile` | Injects profiler hooks into every kernel.                 |
| `--build-tt-train` | Builds the `tt-train` helper used by the NanoGPT example. |

After completion the relevant binaries live under `build/tt-train/`.

---

## Running the Profiler

```bash
TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_WATCHER_DEBUG_DELAY=10 \
TT_METAL_READ_DEBUG_DELAY_CORES=0,0 \
TT_METAL_WRITE_DEBUG_DELAY_CORES=0,0 \
TT_METAL_READ_DEBUG_DELAY_RISCVS=BR \
TT_METAL_WRITE_DEBUG_DELAY_RISCVS=BR \
python -m tracy -r -v -p build/tt-train/sources/examples/nano_gpt/nano_gpt
```

### What do those variables mean?

| Variable                                 | Default | Description                                                                         |
| ---------------------------------------- | ------- | ----------------------------------------------------------------------------------- |
| `TT_METAL_WATCHER_NOINLINE`              | `0`     | Forces watchdog helpers to stay **out‑of‑line** for clearer flame graphs.           |
| `TT_METAL_WATCHER_DEBUG_DELAY`           | `0` ms  | Extra delay (ms) after each kernel for debugger attachment.                         |
| `TT_METAL_READ/WRITE_DEBUG_DELAY_CORES`  | `"0,0"` | Comma‑separated *(x,y)* coordinates of cores for which to inject read/write delays. |
| `TT_METAL_READ/WRITE_DEBUG_DELAY_RISCVS` | `BR`    | RISC‑V side delays. Use `BR` for **B**oot & **R**untime, `B`, `R`, or leave empty.  |

> **Can I omit some options?** Yes—only set what you actually need; unset variables fallback to defaults. But it wasn't tested

---

## Analysing Results

Open the companion notebook and point it to your freshly‑generated CSV:

```bash
jupyter lab notebooks/profiler_results.ipynb
```

The notebook walks you through:

1. **Aggregating time‑per‑op** across the whole training step.
2. Building **heat‑maps** to spot core imbalance.
3. Surfacing the **top N kernels** by wall‑clock time.

Feel free to fork / extend the notebook for your own workflows.

---

## Troubleshooting & FAQ

| Symptom                                                              | Cause & Fix                                                                                                                   |
| -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **“Profiler unavailable”** during an *independent* out‑of‑tree build | The standalone `cmake` flow does not yet inject profiling hooks. → Use **`build_metal.sh`** with `--enable-profile`.          |
| Need custom CLI flags for nanogpt                                               | At the moment flags are *hard‑coded*—edit `main.cpp` and rebuild.                                                         |
| Empty / partial CSV                                                  | You might be dumping too infrequently. Call:<br>`ctx().get_profiler().dump_results();`<br>near your training‑loop boundaries. |
| Unknown env vars                                                     | Any unrecognised variable is silently ignored—double‑check spelling & values.                                                 |

---
