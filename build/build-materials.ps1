[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BuildPython,
    [Parameter(Mandatory)][string]$CudaHome,
    [Parameter(Mandatory)][string]$OutDir,
    [string]$Profile = (Join-Path $PSScriptRoot 'profiles/q2opt2.json'),
    [int]$Jobs = 2,
    [switch]$PreflightOnly
)
$ErrorActionPreference = 'Stop'
# 按目标机器探测 VS；不写死本机 workspace、venv 或 CUDA 路径。
$VsWhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$VsRoot = & $VsWhere -latest -products '*' -version '[17.0,18.0)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
if (-not $VsRoot) { throw 'VS2022 C++ Build Tools are unavailable' }
$DevCmd = Join-Path $VsRoot 'Common7\Tools\VsDevCmd.bat'
$CompilerEnvironment = & $env:ComSpec /d /s /c "call `"$DevCmd`" -arch=x64 -host_arch=x64 >nul && set"
if ($LASTEXITCODE -ne 0) { throw 'VS2022 environment initialization failed' }
foreach ($Line in $CompilerEnvironment) {
    if ($Line -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2] }
}
$env:CUDA_HOME = (Resolve-Path -LiteralPath $CudaHome).Path
$env:CUDA_PATH = $env:CUDA_HOME
$env:DISTUTILS_USE_SDK = '1'
$env:VSLANG = '1033'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MAX_JOBS = [string]$Jobs
$env:PATH = (Split-Path -Parent (Get-Command cl.exe).Source) + ';' + (Split-Path -Parent $BuildPython) + ';' + (Join-Path $env:CUDA_HOME 'bin') + ';' + $env:PATH
$ProducerArguments = @('--output', $OutDir, '--jobs', [string]$Jobs, '--profile', $Profile)
if ($PreflightOnly) { $ProducerArguments += '--preflight-only' }
& $BuildPython -X utf8 -B (Join-Path $PSScriptRoot 'build_materials.py') @ProducerArguments
if ($LASTEXITCODE -ne 0) { throw "FreeToken materials build failed ($LASTEXITCODE); retain output logs" }
