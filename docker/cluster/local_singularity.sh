#!/usr/bin/env bash

# Singularity executable path
SINGULARITY_BIN=/apps/spack/opt/spack/linux-rocky8-zen4/gcc-13.2.0/singularityce-4.1.0-bgiu75t2gjd6w4rbl4weloiokpgd2wii/bin/singularity

#==
# Helper functions
#==

setup_directories() {
    for dir in \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/kit" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/ov" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/pip" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/glcache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/computecache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/logs" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/data" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/documents"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo "Created directory: $dir"
        fi
    done
}

#==
# Main
#==

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Load variables
source $SCRIPT_DIR/.env.cluster
source $SCRIPT_DIR/../.env.base

# Use CLUSTER_ISAACLAB_DIR if first argument is not provided or doesn't exist
if [ -z "$1" ] || [ ! -d "$1" ]; then
    ISAACLAB_SOURCE_DIR=$CLUSTER_ISAACLAB_DIR
    echo "[INFO] Using CLUSTER_ISAACLAB_DIR: $ISAACLAB_SOURCE_DIR"
else
    ISAACLAB_SOURCE_DIR=$1
    echo "[INFO] Using provided directory: $ISAACLAB_SOURCE_DIR"
fi

# Verify the directory exists
if [ ! -d "$ISAACLAB_SOURCE_DIR" ]; then
    echo "[ERROR] Isaac Lab directory does not exist: $ISAACLAB_SOURCE_DIR"
    exit 1
fi

# Shift arguments: container profile is now $1, script is $2, etc.
shift
CONTAINER_PROFILE=$1
shift

echo "(run_singularity.py): Called on compute node from current isaaclab directory $ISAACLAB_SOURCE_DIR with container profile $CONTAINER_PROFILE and arguments ${@}"

# Set TMPDIR to scratch space
if [ -z "$TMPDIR" ]; then
    if [ -n "$PBS_JOBID" ]; then
        export TMPDIR=/scratch/$USER/tmp_${PBS_JOBID}
    else
        export TMPDIR=/scratch/$USER/tmp_$$
    fi
    mkdir -p $TMPDIR
    echo "[INFO] Created TMPDIR on scratch: $TMPDIR"
else
    echo "[INFO] Using existing TMPDIR: $TMPDIR"
fi

# Ensure TMPDIR is on scratch
if [[ ! "$TMPDIR" =~ ^/scratch ]]; then
    echo "[WARNING] TMPDIR is not on scratch. Overriding to scratch space for better performance."
    if [ -n "$PBS_JOBID" ]; then
        export TMPDIR=/scratch/$USER/tmp_${PBS_JOBID}
    else
        export TMPDIR=/scratch/$USER/tmp_$$
    fi
    mkdir -p $TMPDIR
    echo "[INFO] Set TMPDIR to: $TMPDIR"
fi

# Setup directories
setup_directories
cp -r $CLUSTER_ISAAC_SIM_CACHE_DIR $TMPDIR

# Make logs directory
mkdir -p "$ISAACLAB_SOURCE_DIR/logs"
touch "$ISAACLAB_SOURCE_DIR/logs/.keep"

# Copy isaaclab to compute node
echo "[DEBUG] Copying $ISAACLAB_SOURCE_DIR to $TMPDIR"
cp -r $ISAACLAB_SOURCE_DIR $TMPDIR
dir_name=$(basename "$ISAACLAB_SOURCE_DIR")
echo "[DEBUG] Directory name: $dir_name"
echo "[DEBUG] Target path in TMPDIR: $TMPDIR/$dir_name"

# Verify the training script exists
TRAIN_SCRIPT="$1"
if [ -f "$TMPDIR/$dir_name/$TRAIN_SCRIPT" ]; then
    echo "[INFO] Training script found: $TRAIN_SCRIPT"
else
    echo "[ERROR] Training script NOT found at: $TMPDIR/$dir_name/$TRAIN_SCRIPT"
    echo "[DEBUG] Contents of $TMPDIR/$dir_name:"
    ls -la $TMPDIR/$dir_name/ | head -20
fi

# Extract container
echo "[DEBUG] Extracting container from $CLUSTER_SIF_PATH/$CONTAINER_PROFILE.tar"
tar -xf $CLUSTER_SIF_PATH/$CONTAINER_PROFILE.tar -C $TMPDIR

# Create Isaac Sim symlink in the copied directory (REQUIRED by isaaclab.sh)
if [ ! -L "$TMPDIR/$dir_name/_isaac_sim" ]; then
    echo "[INFO] Creating _isaac_sim symlink -> $DOCKER_ISAACSIM_ROOT_PATH"
    ln -sf $DOCKER_ISAACSIM_ROOT_PATH $TMPDIR/$dir_name/_isaac_sim
fi

# Verify symlink
if [ -L "$TMPDIR/$dir_name/_isaac_sim" ]; then
    echo "[INFO] _isaac_sim symlink created: $(readlink $TMPDIR/$dir_name/_isaac_sim)"
else
    echo "[ERROR] Failed to create _isaac_sim symlink"
    exit 1
fi

# Execute with isaaclab.sh install
echo "[DEBUG] Running singularity with bind: $TMPDIR/$dir_name:/workspace/isaaclab:rw"
$SINGULARITY_BIN exec \
    -B $TMPDIR/docker-isaac-sim/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw \
    -B $TMPDIR/docker-isaac-sim/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
    -B $TMPDIR/docker-isaac-sim/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw \
    -B $TMPDIR/docker-isaac-sim/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw \
    -B $TMPDIR/docker-isaac-sim/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw \
    -B $TMPDIR/docker-isaac-sim/documents:${DOCKER_USER_HOME}/Documents:rw \
    -B $TMPDIR/$dir_name:/workspace/isaaclab:rw \
    -B $CLUSTER_ISAACLAB_DIR/logs:/workspace/isaaclab/logs:rw \
    --env ISAACLAB_PATH=/workspace/isaaclab \
    --pwd /workspace/isaaclab \
    --nv \
    $TMPDIR/$CONTAINER_PROFILE.sif \
    /bin/bash -c "
        cd /workspace/isaaclab && \
        echo 'Installing Isaac Lab...' && \
        ./isaaclab.sh --install && \
        echo 'Isaac Lab installed, running training...' && \
        ./isaaclab.sh -p ${@}
    "


# Copy cache back
rsync -azPv $TMPDIR/docker-isaac-sim $CLUSTER_ISAAC_SIM_CACHE_DIR/..

# Cleanup temporary directory if in interactive mode
if [ -z "$PBS_JOBID" ]; then
    echo "[INFO] Cleaning up TMPDIR: $TMPDIR"
    rm -rf $TMPDIR
fi

# Cleanup code copy if configured
if $REMOVE_CODE_COPY_AFTER_JOB; then
    rm -rf $ISAACLAB_SOURCE_DIR
fi

echo "(run_singularity.py): Return"

