$ErrorActionPreference = 'Stop'
$env:CUDA_PATH = 'C:\workspace\tools\cuda-13.0'
$env:CUDA_HOME = $env:CUDA_PATH
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = 'C:\workspace\worktrees\speed-q2-material-review'
$buildArgs = @{
    SourceVenv = 'C:\workspace\envs\speed-q2-review'
    BuildVenv = 'C:\workspace\envs\silitok-speed-cu130'
    PythonEmbedZip = 'C:\workspace\downloads\python-3.12.10-embed-amd64.zip'
    WinSWExe = 'C:\workspace\downloads\WinSW-x64-v2.12.0.exe'
    VcRedist = 'C:\workspace\downloads\vc_redist.x64.exe'
    PydDir = 'C:\workspace\artifacts\kt-cu130\pyd'
    H3PrebuiltDir = 'C:\workspace\artifacts\h3-cu130'
    FreetokenWheel = 'C:\workspace\worktrees\build-freetoken-q2-material-review\artifacts\q2opt2-cp312-cu130-sm120\wheels\freetoken-0.1.3+speed.q2opt2-cp312-cp312-win_amd64.whl'
    KernelCacheDir = 'C:\workspace\worktrees\build-freetoken-q2-material-review\artifacts\q2opt2-cp312-cu130-sm120\kernel-cache'
    OutDir = 'C:\workspace\build\q2-review-20261011'
    SkipInstaller = $true
}
if (Test-Path -LiteralPath $buildArgs.OutDir) { throw 'Use a fresh output directory' }
Push-Location -LiteralPath 'C:\workspace\worktrees\speed-q2-material-review'
try {
    & .\packaging\windows\build.ps1 @buildArgs
    if ($LASTEXITCODE -ne 0) { throw "Speed build failed: $LASTEXITCODE" }
} finally { Pop-Location }
