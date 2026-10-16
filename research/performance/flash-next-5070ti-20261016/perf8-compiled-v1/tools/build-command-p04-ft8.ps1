$ErrorActionPreference = 'Stop'
$payload = 'C:\workspace\build\siltok-speed\p04'
if (Test-Path -LiteralPath $payload) { throw "Build output already exists: $payload" }
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
& 'C:\workspace\wt\siltok-speed\flash-perf\packaging\windows\build.ps1' `
  -PythonEmbedZip 'C:\workspace\tools\setup\python-3.12.10-embed-amd64.zip' `
  -SourceVenv 'C:\workspace\envs\siltok-speed\flash-perf-rel8' `
  -BuildVenv 'C:\workspace\envs\siltok-speed\flash-perf-build' `
  -PythonDevRoot 'C:\workspace\tools\python312' `
  -WinSWExe 'C:\workspace\tools\WinSW-x64.exe' `
  -VcRedist 'C:\workspace\tools\setup\vc_redist.x64.exe' `
  -FreetokenWheel 'C:\workspace\wt\freetoken-mats\flash-perf\artifacts\strataq2perf8-cp312-cu130-sm120\wheels\freetoken-0.1.3+speed.strataq2perf8-cp312-cp312-win_amd64.whl' `
  -KernelCacheDir 'C:\workspace\wt\freetoken-mats\flash-perf\artifacts\strataq2perf8-cp312-cu130-sm120\kernel-cache' `
  -PydDir 'C:\workspace\mats\siltok-speed\kt' `
  -H3PrebuiltDir 'C:\workspace\mats\siltok-speed\h3' `
  -VcpkgRoot 'C:\workspace\tools\vcpkg' `
  -OutDir $payload
