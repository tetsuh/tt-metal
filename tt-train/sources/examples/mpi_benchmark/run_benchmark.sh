#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Defaults (override with -m)
METAL_HOME="${TT_METAL_HOME:-/home/ttuser/git/tt-metal}"
BIN_DIR="${METAL_HOME}/tt-train/build/sources/examples/mpi_benchmark"
SSH_USER="ttuser"
SCP_OPTS="-p"    # preserve times & modes
BINARIES=(mpi_benchmark)

# Your cluster hosts, in the order MPI should assign ranks:
HOSTS=(
  "11.228.0.10"
  "11.228.0.11"
  "11.228.0.14"
  # "11.228.0.16"
)

print_usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  -h            Show this help and exit
  -m METAL_HOME Override TT_METAL_HOME (default: $METAL_HOME)
EOF
}

# parse flags
while getopts "hm:" opt; do
  case "$opt" in
    h) print_usage; exit 0 ;;
    m) METAL_HOME="$OPTARG"
       BIN_DIR="${METAL_HOME}/tt-train/build/sources/examples/mpi_benchmark"
       ;;
    *) print_usage; exit 1 ;;
  esac
done

# must have at least 2 hosts
NUM_HOSTS=${#HOSTS[@]}
if (( NUM_HOSTS < 2 )); then
  echo "ERROR: need at least 2 hosts" >&2
  exit 1
fi

# check binaries exist and are executable
for bin in "${BINARIES[@]}"; do
  if [[ ! -x "${BIN_DIR}/${bin}" ]]; then
    echo "ERROR: Missing or not executable: ${BIN_DIR}/${bin}" >&2
    exit 1
  fi
done

# build a temporary hostfile
HOSTFILE=$(mktemp)
trap 'rm -f "$HOSTFILE"' EXIT

for h in "${HOSTS[@]}"; do
  printf "%s slots=1\n" "$h" >> "$HOSTFILE"
done

# copy binaries to all remote hosts (skip first)
echo "Copying binaries to remote hosts..."
for host in "${HOSTS[@]:1}"; do
  echo " -> $host"
  ssh "${SSH_USER}@${host}" "mkdir -p '${BIN_DIR}'"
  for bin in "${BINARIES[@]}"; do
    scp ${SCP_OPTS} "${BIN_DIR}/${bin}" "${SSH_USER}@${host}:${BIN_DIR}/"
  done
done
echo "✔ Remote copy complete."

# launch MPI job, exporting TT_METAL_HOME so each rank sees it
echo "Launching MPI benchmark..."
mpirun-ulfm \
  --hostfile "$HOSTFILE" \
  -np "$NUM_HOSTS" \
  -x TT_METAL_HOME \
  "${BIN_DIR}/mpi_benchmark"

echo "✔ MPI benchmark finished."
