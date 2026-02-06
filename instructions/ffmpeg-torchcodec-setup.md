# FFmpeg + TorchCodec Setup on CSCS

Guide to set up FFmpeg and TorchCodec for audio/video processing on compute nodes.

## Quick Setup (per job)

Add to your SLURM script or shell before running Python:

```bash
export LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/deps/ffmpeg/lib:${LD_LIBRARY_PATH}
export PATH=/capstor/store/cscs/swissai/infra01/MLLM/deps/ffmpeg/bin:${PATH}
```

### Verify

```bash
ffmpeg -version
python -c "import torchcodec; print('torchcodec OK')"
```

### SLURM example

```bash
#!/bin/bash
#SBATCH --job-name=audio_preprocess

export LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/deps/ffmpeg/lib:${LD_LIBRARY_PATH}
export PATH=/capstor/store/cscs/swissai/infra01/MLLM/deps/ffmpeg/bin:${PATH}

srun python preprocess.py
```

## What's Installed

FFmpeg 8 (N-122667), compiled from source for aarch64 (GH200):

```
/capstor/store/cscs/swissai/infra01/MLLM/deps/ffmpeg/
├── bin/       ffmpeg, ffprobe
├── lib/       libavcodec.so, libavutil.so, libavformat.so, ...
├── include/   C headers
└── share/     docs
```

## Installing TorchCodec (requires --container-writable)

Pre-built torchcodec wheels from PyPI have an ABI mismatch with the NVIDIA PyTorch build (`2.9.0a0+nv25.09`). Must build from source.

### 1. Start a compute node with writable container

```bash
srun --partition debug --time 01:30:00 -A <your-account> --mem 460000 \
     --container-writable --pty bash
```

### 2. Set FFmpeg library path

```bash
export LD_LIBRARY_PATH=/capstor/store/cscs/swissai/infra01/MLLM/deps/ffmpeg/lib:${LD_LIBRARY_PATH}
```

### 3. Build and install torchcodec from source

```bash
pip install --no-deps --no-build-isolation git+https://github.com/pytorch/torchcodec.git
```

### 4. Verify

```bash
python -c "import torchcodec; print(torchcodec.__version__)"
```

## Usage in Python

```python
# torchaudio automatically uses torchcodec as backend
import torchaudio
wav, sr = torchaudio.load("audio.wav")   # WAV
wav, sr = torchaudio.load("audio.mp3")   # MP3 (needs torchcodec + ffmpeg)
wav, sr = torchaudio.load("audio.flac")  # FLAC

# Or use soundfile for WAV/FLAC (no FFmpeg needed)
import soundfile as sf
audio, sr = sf.read("audio.wav")
```

## How FFmpeg Was Built

Compiled from source to `/capstor/store/cscs/swissai/infra01/MLLM/deps/ffmpeg/` (no sudo needed):

```bash
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
git clone https://git.ffmpeg.org/ffmpeg.git

cd nv-codec-headers
make PREFIX=/capstor/store/cscs/swissai/infra01/MLLM/deps/ffmpeg install

cd ../ffmpeg
./configure --prefix=/capstor/store/cscs/swissai/infra01/MLLM/deps/ffmpeg \
            --enable-shared --disable-static
make -j$(nproc)
make install
```

To rebuild with NVIDIA GPU acceleration (NVENC/NVDEC), add to configure:
```bash
--enable-nonfree --enable-cuda-nvcc --enable-libnpp \
--extra-cflags=-I/usr/local/cuda/include \
--extra-ldflags=-L/usr/local/cuda/lib64
```
