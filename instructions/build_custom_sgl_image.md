# Building Custom Container Images on CSCS

Quick guide to customize any container image on CSCS using Enroot.

## Steps

### 1. Start a compute node
```bash
srun --partition debug --time 01:30:00 -A <your-account> --mem 460000 --pty bash
```

### 2. Create and modify container
```bash
cd /iopsstor/scratch/cscs/<username>
enroot create --name <container-name> /path/to/base/image.sqsh
enroot start --root --rw <container-name>
```

### 3. Inside container, install your packages
```bash
# Install whatever you need
apt-get update && apt-get install -y <packages>
pip install <python-packages>

# Make your modifications
# ...

# Exit when done
exit
```

### 4. Export the modified container
```bash
enroot export --output /path/to/output/custom-image.sqsh <container-name>
```

## Example: Building on SGL Image

```bash
# Create container from SGL base image
enroot create --name sgl-dev /capstor/store/cscs/swissai/infra01/container-images/sgl1/image.sqsh
enroot start --root --rw sgl-dev

# Inside container - install ML packages
apt-get update && apt-get install -y tmux htop nvtop && apt-get clean
pip install --no-cache-dir numba matplotlib seaborn evaluate pytorch-ignite

# Install Swiss AI transformers
cd /tmp && git clone https://github.com/swiss-ai/transformers.git && \
cd transformers && git checkout v4.51.1+swissai+cuda && \
pip install --no-deps --no-cache-dir --force-reinstall .

exit

# Export custom image
enroot export --output sgl-custom.sqsh sgl-dev
```

## Usage
```bash
enroot create --name my-container /path/to/custom-image.sqsh
enroot start --rw my-container
```

## Notes
- Use Enroot instead of podman/docker on CSCS (filesystem doesn't support extended attributes)
- Base images available at `/capstor/store/cscs/swissai/infra01/container-images/`
- Export to `/capstor` for long-term storage, `/iopsstor/scratch` for temporary use