$ErrorActionPreference = 'Stop'
. C:\workspace\tools\bin\enter-speed.ps1 -Profile build -Native -Task flash-review2b
$payload = 'C:\workspace\build\siltok-speed\review2b-p01'
$resolvedPayload = [IO.Path]::GetFullPath($payload)
if (-not $resolvedPayload.StartsWith('C:\workspace\build\siltok-speed\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Build output escaped task root' }
if (Test-Path -LiteralPath $resolvedPayload) { throw 'Fresh build output required' }

$releasePython='C:\workspace\envs\siltok-speed\flash-review2b-release\Scripts\python.exe'
$wheel='C:\workspace\wt\freetoken-mats\flash-perf\artifacts\strataq2review2b-cp312-cu130-sm120\wheels\freetoken-0.1.3+speed.strataq2review2b-cp312-cp312-win_amd64.whl'
$cacheSource='C:\workspace\wt\freetoken-mats\flash-perf\artifacts\strataq2review2b-cp312-cu130-sm120\kernel-cache'
$cacheCopy='C:\workspace\runs\siltok-speed\explore-20261017-evening\packaging-preflight-cache-v1'
if (Test-Path -LiteralPath $cacheCopy) { throw 'Fresh preflight cache required' }
Copy-Item -LiteralPath $cacheSource -Destination $cacheCopy -Recurse
foreach ($pattern in @('cpuinfer_ext*.pyd','KTransformersOps*.pyd','vLLMMarlin*.pyd')) {
    $files=Get-ChildItem -LiteralPath C:\workspace\mats\siltok-speed\kt -Filter $pattern -File
    if (-not $files) { throw "Missing extension: $pattern" }
    $files | Get-FileHash -Algorithm SHA256 | Format-List
}
$torchCuda=(& $releasePython -I -B -c 'import torch;print(torch.version.cuda)').Trim()
$nvccText=(& "$env:CUDA_PATH\bin\nvcc.exe" --version | Out-String)
$toolkitMajor=[regex]::Match($nvccText,'release\s+(\d+)\.').Groups[1].Value
if (-not $toolkitMajor -or $toolkitMajor -ne $torchCuda.Split('.')[0]) { throw 'CUDA major mismatch' }
Write-Host "torch CUDA=$torchCuda; Toolkit=$toolkitMajor; xollvm disabled"
& $releasePython -I -B -m silitok_speed.qwen36.freetoken_release verify-wheel --wheel $wheel --pyproject C:\workspace\wt\siltok-speed\flash-perf\silitok_speed\pyproject.toml
if ($LASTEXITCODE -ne 0) { throw 'Wheel preflight failed' }
& $releasePython -I -B -m silitok_speed.qwen36.freetoken_release generate-manifest --kernel-cache $cacheCopy
if ($LASTEXITCODE -ne 0) { throw 'AOT manifest preflight failed' }
& $releasePython -I -B -m silitok_speed.qwen36.freetoken_release self-check --kernel-cache $cacheCopy --pip-check --constraint-policy C:\workspace\wt\siltok-speed\flash-perf\packaging\windows\runtime-requirements.toml
if ($LASTEXITCODE -ne 0) { throw 'Release runtime preflight failed' }

& C:\workspace\wt\siltok-speed\flash-perf\packaging\windows\build.ps1 `
  -PythonEmbedZip 'C:\workspace\tools\setup\python-3.12.10-embed-amd64.zip' `
  -SourceVenv 'C:\workspace\envs\siltok-speed\flash-review2b-release' `
  -BuildVenv 'C:\workspace\envs\siltok-speed\flash-review2b' `
  -PythonDevRoot 'C:\workspace\tools\python312' `
  -WinSWExe 'C:\workspace\tools\WinSW-x64.exe' `
  -VcRedist 'C:\workspace\tools\setup\vc_redist.x64.exe' `
  -FreetokenWheel 'C:\workspace\wt\freetoken-mats\flash-perf\artifacts\strataq2review2b-cp312-cu130-sm120\wheels\freetoken-0.1.3+speed.strataq2review2b-cp312-cp312-win_amd64.whl' `
  -KernelCacheDir 'C:\workspace\wt\freetoken-mats\flash-perf\artifacts\strataq2review2b-cp312-cu130-sm120\kernel-cache' `
  -PydDir 'C:\workspace\mats\siltok-speed\kt' `
  -H3PrebuiltDir 'C:\workspace\mats\siltok-speed\h3' `
  -VcpkgRoot 'C:\workspace\tools\vcpkg' `
  -OutDir $resolvedPayload
if ($LASTEXITCODE -ne 0) { throw "Protected payload build failed: $LASTEXITCODE" }
