# Build script for requant_plugin.dll
# Usage: Run in PowerShell from this directory.

$ErrorActionPreference = "Stop"

$buildDir = Join-Path $PSScriptRoot "build"
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

Push-Location $buildDir

# Configure (Visual Studio generator)
$cudaNvcc = "E:/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/nvcc.exe"

# Clean previous cache if generator changed
if (Test-Path "$buildDir/CMakeCache.txt") {
    Remove-Item "$buildDir/CMakeCache.txt" -Force
}
if (Test-Path "$buildDir/CMakeFiles") {
    Remove-Item "$buildDir/CMakeFiles" -Recurse -Force
}

cmake .. -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CUDA_COMPILER="$cudaNvcc"

# Build
cmake --build . --config Release

Pop-Location
