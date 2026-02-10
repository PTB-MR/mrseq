
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Coverage Bagde](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/schuenke/330a0c00b5fa35d89bbc73ea6e8d99be/raw/coverage.json)

# MRseq - MR pulse sequences using Pulseq

`MRseq` is a collection of several useful functions, kernels and scripts for creating vendor-agnostic MR pulse sequences using the open-source Pulseq format.

- **Source code:** <https://github.com/PTB-MR/mrseq>
- **Bug reports:** <https://github.com/PTB-MR/mrseq/issues>
- **Documentation:** <https://ptb-mr.github.io/mrseq/intro.html>

## Contributing

We are looking forward to your contributions via Pull-Requests.

### Installation for developers

#### Prerequisites for Windows

Before installing `MRseq` with development dependencies on Windows, you need:

1. **Visual Studio Build Tools**: The MRzeroCore dependency requires Rust compilation with Microsoft Visual C++ linker
   - Download "Build Tools for Visual Studio 2022" from https://visualstudio.microsoft.com/downloads/
   - During installation, select the "C++ build tools" workload
   - Ensure "Windows 10/11 SDK" is included
   - This is required for compiling native Rust extensions

2. **Rust toolchain** (automatically installed by MRzeroCore if not present)

#### Installation steps

1. Clone the `MRseq` repository
2. Create/select a python environment
3. Install `MRseq` in editable mode including test dependencies: ``` pip install -e ".[dev]" ```
4. Setup pre-commit hook: ``` pre-commit install ```


### Licensing and Dependencies

The core source code of `MRseq` is licensed under the **Apache-2.0 license**. 

For quality control and demonstration purposes, the notebooks in the `examples/` directories utilize [MRZero](https://github.com/MRsources/MRzero-Core) which is AGPL-licensed. The notebooks in the `examples/` folder should be treated as **AGPLv3**. 

**Important for Commercial Users:**
The AGPL dependency is listed only as an optional dev dependency. Installing this package via `pip install mrseq` **does not** install the AGPL library. Your usage of the core library remains subject to the permissive Apache-2.0 terms, unaffected by the licensing of the test suite or examples.
