[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$taskRun = 'C:\workspace\runs\siltok-speed\flash-perf-20261016'
$taskEvidence = Join-Path $taskRun 'install-i06'
$taskInstall = 'C:\workspace\build\siltok-speed\i06'
$taskInstaller = 'C:\workspace\build\siltok-speed\p03\SiltokSpeed-Setup-0.1.2.dev0+strataq2perf2.2026101613.exe'
$taskExit = 1
# 此提权入口仅安装本轮固定产物、记录归属，并停止测试服务；不建立常驻提权通道。
try {
    $taskIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
    if (-not ([Security.Principal.WindowsPrincipal]$taskIdentity).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Administrator token required' }
    if (Test-Path -LiteralPath $taskEvidence) { throw 'Preserve previous installation evidence' }
    if ((Get-FileHash -LiteralPath $taskInstaller -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'be5eccf429c262f19ef077768520286db34c0b0b7d959c0030c24b461368f8a3') { throw 'Installer identity mismatch' }
    New-Item -ItemType Directory -Path $taskEvidence | Out-Null
    Start-Transcript -LiteralPath (Join-Path $taskEvidence 'elevated.log') | Out-Null
    & C:\workspace\tools\bin\record-flash-install.ps1 -InstallId i06 -Phase before -RunRoot $taskRun
    $taskArguments = @('/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART',('/DIR='+$taskInstall),'/MODELSDIR=C:\workspace\cfg\siltok-speed\flash-accept\delivery-off-v1',('/LOG='+$taskEvidence+'\install.log'))
    $taskProcess = Start-Process -FilePath $taskInstaller -ArgumentList $taskArguments -WindowStyle Hidden -Wait -PassThru
    if ($taskProcess.ExitCode -ne 0) { throw "Installer failed: $($taskProcess.ExitCode)" }
    & C:\workspace\tools\bin\record-flash-install.ps1 -InstallId i06 -Phase installed -RunRoot $taskRun
    $taskService = Get-CimInstance Win32_Service -Filter "Name='SiltokSpeed'"
    if ($taskService.PathName -notlike '*C:\workspace\build\siltok-speed\i06\SiltokSpeed.exe*') { throw 'Wrong service owner' }
    if ($taskService.State -ne 'Stopped') { Stop-Service SiltokSpeed -Force }
    [ordered]@{status='installed_service_stopped'; observed_utc=[DateTime]::UtcNow.ToString('o'); install_path=$taskInstall; launcher_admin=$true; session_id=(Get-Process -Id $PID).SessionId; installer_exitcode=$taskProcess.ExitCode} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskEvidence 'result.json') -Encoding utf8
    $taskExit = 0
} catch {
    [ordered]@{status='failed'; observed_utc=[DateTime]::UtcNow.ToString('o'); error=$_.ToString()} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRun 'install-i06-error.json') -Encoding utf8
    Write-Error -ErrorRecord $_ -ErrorAction Continue
} finally {
    if (Test-Path -LiteralPath (Join-Path $taskEvidence 'elevated.log')) { Stop-Transcript | Out-Null }
}
exit $taskExit
