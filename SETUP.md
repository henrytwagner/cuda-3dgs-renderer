# Setup Instructions

This document describes how to configure the project for your development environment.

## Prerequisites

Before building, ensure you have:

1. **CUDA Toolkit** (12.1+ recommended)
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Verify installation: `nvcc --version`

2. **CMake** (3.20+)
   - Download from: https://cmake.org/download/
   - Or use the bundled version in `tools/cmake-3.28.3-windows-x86_64/`

3. **C++ Compiler**
   - Windows: Visual Studio 2019/2022 with C++ build tools
   - Linux: GCC 9+ or Clang 10+

4. **External Libraries** (see README.md for setup)

## Windows Configuration

### Step 1: Locate Your Tool Paths

**CUDA Toolkit:**
```powershell
# Find CUDA installation
Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\" -Directory
```
Common location: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1`

**Visual Studio MSVC:**
```powershell
# Find MSVC compiler
Get-ChildItem "C:\Program Files\Microsoft Visual Studio\" -Recurse -Filter "cl.exe" -ErrorAction SilentlyContinue | Select-Object -First 1 DirectoryName
```
Common location: `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\<version>\bin\Hostx64\x64`

**CMake:**
- If using bundled CMake: `tools\cmake-3.28.3-windows-x86_64\bin`
- If installed system-wide: Leave empty (must be in PATH)

### Step 2: Create Configuration File

1. Copy `config.ps1.example` to `config.ps1`:
   ```powershell
   Copy-Item config.ps1.example config.ps1
   ```

2. Edit `config.ps1` and update the paths:
   ```powershell
   $CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
   $MSVC_PATH = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64"
   $CMAKE_PATH = "$PSScriptRoot\tools\cmake-3.28.3-windows-x86_64\bin"
   ```

3. **Important:** Add `config.ps1` to `.gitignore` to avoid committing your local paths:
   ```
   config.ps1
   ```

### Step 3: Verify Configuration

Test the configuration:
```powershell
.\run_all_benchmarks.ps1 -SkipBuild
```

If paths are incorrect, you'll see errors about missing executables. Update `config.ps1` accordingly.

## Linux Configuration

On Linux, the build scripts use environment variables. Set them in your shell:

```bash
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
```

Then build normally:
```bash
cmake -S . -B build
cmake --build build --config Release
```

## Alternative: Environment Variables

Instead of using `config.ps1`, you can set environment variables:

**Windows (PowerShell):**
```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
$env:PATH = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64;$env:PATH"
```

**Windows (Command Prompt):**
```cmd
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
set PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64;%PATH%
```

**Linux/Mac:**
```bash
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
```

## Running Scripts

**PowerShell Execution Policy:**

Windows may block PowerShell scripts by default. To run the benchmark script, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all_benchmarks.ps1
```

Or set the execution policy for the current user (one-time setup):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

After setting the policy, you can run scripts directly:
```powershell
.\run_all_benchmarks.ps1
```

## Troubleshooting

**"cmake not found"**
- Ensure CMake is in PATH, or set `$CMAKE_PATH` in `config.ps1`
- Verify: `cmake --version`

**"nvcc not found"**
- Check `$CUDA_PATH` points to CUDA installation directory
- Verify: `$CUDA_PATH\bin\nvcc.exe --version` (Windows) or `$CUDA_PATH/bin/nvcc --version` (Linux)

**"cl.exe not found" (Windows)**
- Check `$MSVC_PATH` points to MSVC bin directory
- Open "Developer Command Prompt for VS" to get correct environment
- Or run: `& "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"`

**Build errors about missing libraries**
- Ensure external libraries are set up (see README.md)
- Check that GLFW, GLM, and GLAD are in `external/` directory

