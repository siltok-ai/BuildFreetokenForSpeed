$ErrorActionPreference = 'Stop'
. C:\workspace\tools\bin\enter-speed.ps1 -Profile build -Native -Task flash-review
$payload = 'C:\workspace\build\siltok-speed\review1-p01'
$resolvedPayload = [IO.Path]::GetFullPath($payload)
if (-not $resolvedPayload.StartsWith('C:\workspace\build\siltok-speed\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Build output escaped task root' }
if (Test-Path -LiteralPath $resolvedPayload) { throw 'Fresh build output required' }
& C:\workspace\wt\siltok-speed\flash-perf\packaging\windows\build.ps1 `
  -PythonEmbedZip 'C:\workspace\tools\setup\python-3.12.10-embed-amd64.zip' `
  -SourceVenv 'C:\workspace\envs\siltok-speed\flash-review1-release' `
  -BuildVenv 'C:\workspace\envs\siltok-speed\flash-perf-build' `
  -PythonDevRoot 'C:\workspace\tools\python312' `
  -WinSWExe 'C:\workspace\tools\WinSW-x64.exe' `
  -VcRedist 'C:\workspace\tools\setup\vc_redist.x64.exe' `
  -FreetokenWheel 'C:\workspace\wt\freetoken-mats\flash-perf\artifacts\strataq2review1-cp312-cu130-sm120\wheels\freetoken-0.1.3+speed.strataq2review1-cp312-cp312-win_amd64.whl' `
  -KernelCacheDir 'C:\workspace\wt\freetoken-mats\flash-perf\artifacts\strataq2review1-cp312-cu130-sm120\kernel-cache' `
  -PydDir 'C:\workspace\mats\siltok-speed\kt' `
  -H3PrebuiltDir 'C:\workspace\mats\siltok-speed\h3' `
  -VcpkgRoot 'C:\workspace\tools\vcpkg' `
  -OutDir $resolvedPayload
if ($LASTEXITCODE -ne 0) { throw "Protected payload build failed: $LASTEXITCODE" }
