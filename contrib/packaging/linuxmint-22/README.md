# Installing the Tenstorrent stack on Linux Mint 22

This document describes how to install the Tenstorrent stack on Linux Mint 22 (Ubuntu 24.04 base).

This article covers installation of the Tenstorrent software stack on Linux Mint, including kernel modules, utilities, and the TTNN neural network library.

## Prerequisites

- The host machine has an internet connection to download software packages.
- You have administrator privileges on the host machine (sudo access).
- Linux Mint 22 (based on Ubuntu 24.04) is the target environment.
- Support for Linux Mint is considered experimental at this point.

## Running the installer script

> **Note:**  
> The tt-installer script is not yet supported on Linux Mint 22.  
> Please consider using the manual installation (Option 2) or the Docker image (Option 1).

## TT-NN / TT-Metalium Manual Installation

There are two options for installing TT-Metalium:
- Option 1: From Docker Release Image (quick start)
- Option 2: From Source (developer / debug)

---
### Docker Release Image

Download the latest Docker release from our registry and run with device passthrough:

```sh
docker pull ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-24.04-release-amd64:latest-rc
docker run -it --rm --device /dev/tenstorrent ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-24.04-release-amd64:latest-rc bash
```

For more information on the Docker Release Images, visit the project's package/release pages.

You are all set! Try some TT-NN Basic Examples next:
https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/usage.html#basic-examples

---

### Source

Install from source if you are a developer who wants to be close to the metal and the source code. Recommended for running the demo models.


#### Step 0. Clone the Repository:

```bash
git clone https://github.com/tenstorrent/tt-metal.git --recurse-submodules
cd tt-metal
```

#### Step 1. Install dependencies

Run the script to install build dependeicies
```bash
./install_dependencies.sh
```

#### Step 2. Build the Library:

Option A: Using the build script (recommended)

```bash
./build_metal.sh
```

Option B: Manual CMake build

```bash
mkdir build
cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebugInfo -DCMAKE_CXX_COMPILER=<your compiler>
ninja
sudo ninja install   # or ninja install to install into the build prefix if configured
```

#### Step 4. Python environment setup

(Optional) use an existing Python environment:
```bash
export PYTHON_ENV_DIR=<path_to_your_env_directory>
```

Or create and activate the environment using the provided script:
```bash
./create_venv.sh
source python_env/bin/activate
```

If `PYTHON_ENV_DIR` is not set, the script creates a new virtual environment in `./python_env`.

## You Are All Set!

After installation and reboot, verify device nodes under /dev/tenstorrent and try running tt-metalium demo workloads either in the Docker image or from your built environment.

If you run into issues, collect logs:
- Installer log: the installer prints a temporary log path (e.g. /tmp/tenstorrent_install_XXXX/install.log)
- Kernel build logs: check /var/lib/dkms/<module>/ for DKMS build logs
- dmesg and system journal: sudo dmesg | tail -n 200 ; sudo journalctl -b --no-pager

Happy hacking!
