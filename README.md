# CUDA 3DGS Renderer

CUDA-accelerated real-time renderer for 3D Gaussian Splatting scenes. Implements the tile-based rendering pipeline from Kerbl et al. with GPU-parallel optimizations achieving 125 FPS at 2M Gaussians.

## Quick Links

- [Setup Instructions](SETUP.md) - Configure your development environment
- [Paper](paper_final.tex) - Detailed optimization analysis

## Requirements

- CUDA-capable GPU (Compute Capability 8.6+)
- CUDA Toolkit 12.1+
- CMake 3.20+
- C++20 compatible compiler
- Windows or Linux

## Initial Setup

Before building, you need to configure your development environment. See [SETUP.md](SETUP.md) for detailed instructions.

**Quick start:**
1. Copy `config.ps1.example` to `config.ps1`
2. Edit `config.ps1` with your CUDA, Visual Studio, and CMake paths
3. Follow the external library setup below

## External Libraries

This project uses several external libraries that are gitignored and must be set up separately:

- **GLFW** - Window and input management
- **GLM** - OpenGL mathematics library
- **GLAD** - OpenGL loader
- **tinyply** - PLY file loading (included in repository)

To set up the external libraries, clone them into the `external/` directory:

```bash
git clone https://github.com/glfw/glfw.git external/glfw
git clone https://github.com/g-truc/glm.git external/glm
```

For GLAD, download the OpenGL 4.1 Core profile loader and place it in `external/glad/` with the structure:
```
external/glad/
├── include/
│   ├── glad/
│   │   └── glad.h
│   └── KHR/
│       └── khrplatform.h
└── src/
    └── glad.c
```

## Source Files

This renderer is designed to work with 3D Gaussian Splatting PLY files exported from [Postshot](https://www.postshot.ai/) or compatible tools. The test datasets used in this project are from [Steam Studio's free 3DGS PLY collection](https://note.com/steam_studio/n/ne9736d94f162), specifically the cactus scene variants.

**Compatible PLY files:**
- Uncompressed PLY format exported from Postshot
- Files should contain standard 3DGS attributes (position, scale, rotation, opacity, spherical harmonics)

Place your PLY files in the `data/splats/` directory. The renderer expects PLY files with the standard 3DGS structure.

## Building

```bash
cmake -S . -B build
cmake --build build --config Release
```

The renderer executable will be at `build/Release/renderer_cuda.exe` (Windows) or `build/renderer_cuda` (Linux).

## Running

### Specifying Input Files

The PLY file path is specified as the first positional argument:

```bash
./build/Release/renderer_cuda.exe data/splats/your_scene.ply
```

If no file is specified, the renderer defaults to `data/splats/cactus_splat3_30kSteps_464k_splats.ply`.

**Examples:**
```bash
# Load a specific PLY file
./build/Release/renderer_cuda.exe data/splats/cactus_splat3_30kSteps_142k_splats.ply

# Use absolute path
./build/Release/renderer_cuda.exe C:/path/to/your/scene.ply

# Load file with custom camera settings
./build/Release/renderer_cuda.exe data/splats/scene.ply --pos 0 0 5 --yaw -90 --pitch 0 --fov 60
```

### Command-Line Options

All options come after the PLY file path:

- `--benchmark` - Run automated benchmark with scripted camera movement
- `--output <dir>` - Output directory for benchmark CSV files (default: `benchmark_data`)
- `--pos <x> <y> <z>` - Set initial camera position
- `--yaw <degrees>` - Set initial camera yaw angle
- `--pitch <degrees>` - Set initial camera pitch angle
- `--fov <degrees>` - Set field of view

## Controls

- **Mouse drag** - Rotate camera (click to capture mouse)
- **WASD** - Move camera
- **Space/Shift** - Move up/down
- **Scroll** - Adjust FOV
- **ESC** - Exit

## Benchmarking

Run automated benchmarks across all PLY files:
```powershell
powershell -ExecutionPolicy Bypass -File .\run_all_benchmarks.ps1
```

Or if execution policy is already set:
```powershell
.\run_all_benchmarks.ps1
```

Results are saved to `benchmark_data/` directory. Use `-SkipBuild` to skip rebuilding:
```powershell
.\run_all_benchmarks.ps1 -SkipBuild
```

**Note:** If you get an execution policy error, see [SETUP.md](SETUP.md) for details.

## Project Structure

```
src/
├── utils/            # Utility functions (PLY loading, math)
└── cuda/             # CUDA renderer implementation
    ├── main_cuda.cpp
    ├── cuda_renderer.cu
    └── cuda_renderer.h
```

## Performance

Optimized implementation achieves 125 FPS at 2M Gaussians (720×720 resolution) on RTX 4090. See `paper_final.tex` for detailed performance analysis and optimization techniques.
