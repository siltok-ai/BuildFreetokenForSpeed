$ErrorActionPreference = 'Stop'
$taskRoot = 'C:\workspace\artifacts\flash-next-strata-q2-20261012'
$taskSource = Join-Path $taskRoot 'candidate-source-dev4-working\build-source'
$taskPython = 'C:\workspace\envs\speed-strata-q2-dev4\Scripts\python.exe'
$taskDevCmd = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat'
$taskCompilerEnvironment = & $env:ComSpec /d /s /c "call `"$taskDevCmd`" -arch=x64 -host_arch=x64 >nul && set"
if ($LASTEXITCODE -ne 0) { throw 'VS2022 environment initialization failed' }
foreach ($taskLine in $taskCompilerEnvironment) {
    if ($taskLine -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2] }
}
$env:CUDA_HOME = 'C:\workspace\tools\cuda-13.0'
$env:CUDA_PATH = $env:CUDA_HOME
$env:DISTUTILS_USE_SDK = '1'
$env:VSLANG = '1033'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MAX_JOBS = '1'
$env:PATH = 'C:\workspace\tools\cuda-13.0\bin\x64;C:\workspace\tools\cuda-13.0\bin;'+$env:PATH
& $taskPython -X utf8 -B (Join-Path $taskSource 'tools\build_windows_kernel_cache.py') --output (Join-Path $taskRoot 'dev4-native-aot-probe-v1') --build-dir (Join-Path $taskRoot 'dev4-native-aot-probe-v1-build') --spec freetoken__gguf_mmvq_r1 --arch 12.0 --jobs 1 --keep-build
exit $LASTEXITCODE
