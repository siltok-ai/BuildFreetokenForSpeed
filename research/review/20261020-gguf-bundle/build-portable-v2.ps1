$ErrorActionPreference = 'Stop'
. C:\workspace\tools\bin\enter-speed.ps1 -Profile build -Native -Task flash-portable1
$SourceRoot = 'C:\workspace\wt\siltok-speed\flash-next'
$ReleaseVenv = 'C:\workspace\envs\siltok-speed\flash-portable1'
$Materials = 'C:\workspace\wt\freetoken-mats\flash-perf\artifacts\strataq2portable1-cp312-cu130-sm120'
$PayloadRoot = 'C:\workspace\build\siltok-speed\flash portable-p02'
if (Test-Path -LiteralPath $PayloadRoot) { throw 'Fresh output required' }
$Release = Get-Content (Join-Path $SourceRoot 'silitok_speed\runtimes\freetoken\release.json') -Raw | ConvertFrom-Json
$Wheel = Join-Path $Materials $Release.wheel.path
& "$SourceRoot\packaging\windows\build.ps1" `
  -PythonEmbedZip 'C:\workspace\tools\setup\python-3.12.10-embed-amd64.zip' `
  -SourceVenv $ReleaseVenv `
  -BuildVenv 'C:\workspace\envs\siltok-speed\flash-review2b' `
  -PythonDevRoot 'C:\workspace\tools\python312' `
  -WinSWExe 'C:\workspace\tools\WinSW-x64.exe' `
  -VcRedist 'C:\workspace\tools\setup\vc_redist.x64.exe' `
  -FreetokenWheel $Wheel `
  -KernelCacheDir (Join-Path $Materials 'kernel-cache') `
  -KtSftDir 'C:\workspace\wt\siltok-speed\external final\kt-sft' `
  -PydDir 'C:\workspace\mats\siltok-speed\kt' `
  -H3PrebuiltDir 'C:\workspace\mats\siltok-speed\h3' `
  -VcpkgRoot 'C:\workspace\tools\vcpkg' `
  -OutDir $PayloadRoot
if ($LASTEXITCODE) { throw "Protected build failed $LASTEXITCODE" }
$Delivery = 'C:\workspace\dist\siltok-speed\flash-portable-p02'
New-Item -ItemType Directory -Path $Delivery -ErrorAction Stop | Out-Null
Get-ChildItem -LiteralPath $PayloadRoot -Filter 'SiltokSpeed-Setup-*.exe' -File |
    Copy-Item -Destination $Delivery -ErrorAction Stop
