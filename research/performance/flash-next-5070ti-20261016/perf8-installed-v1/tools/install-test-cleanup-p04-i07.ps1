[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$InstallerSha256,
    [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$BundleSha256
)
$ErrorActionPreference = 'Stop'
$taskRun = 'C:\workspace\runs\siltok-speed\flash-perf-20261016'
$taskEvidence = Join-Path $taskRun 'install-i07'
$taskInstall = 'C:\workspace\build\siltok-speed\i07'
$taskExit = 1
$taskOwned = $false
# 单次、限时的安装与清理流程；只等待固定验收结果，不执行标记文件提供的命令。
try {
    $taskIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
    if (-not ([Security.Principal.WindowsPrincipal]$taskIdentity).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Administrator token required' }
    if (Test-Path -LiteralPath $taskEvidence) { throw 'Preserve prior i07 evidence' }
    $taskInstaller = @(Get-ChildItem -LiteralPath 'C:\workspace\build\siltok-speed\p04' -Filter 'SiltokSpeed-Setup-*.exe' -File)
    if ($taskInstaller.Count -ne 1) { throw 'Require the single concrete p04 installer' }
    if ((Get-FileHash -LiteralPath $taskInstaller[0].FullName -Algorithm SHA256).Hash.ToLowerInvariant() -ne $InstallerSha256) { throw 'Installer identity mismatch' }
    New-Item -ItemType Directory -Path $taskEvidence | Out-Null
    Start-Transcript -LiteralPath (Join-Path $taskEvidence 'elevated.log') | Out-Null
    & C:\workspace\tools\bin\record-flash-install.ps1 -InstallId i07 -Phase before -RunRoot $taskRun
    $taskArguments = @('/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART',('/DIR='+$taskInstall),'/MODELSDIR=C:\workspace\cfg\siltok-speed\flash-perf\perf8-off-v1',('/LOG='+$taskEvidence+'\install.log'))
    $taskProcess = Start-Process -FilePath $taskInstaller[0].FullName -ArgumentList $taskArguments -WindowStyle Hidden -Wait -PassThru
    if ($taskProcess.ExitCode -ne 0) { throw "Installer failed: $($taskProcess.ExitCode)" }
    & C:\workspace\tools\bin\record-flash-install.ps1 -InstallId i07 -Phase installed -RunRoot $taskRun
    $taskOwned = $true
    $taskService = Get-CimInstance Win32_Service -Filter "Name='SiltokSpeed'"
    if ($taskService.PathName -notlike '*C:\workspace\build\siltok-speed\i07\SiltokSpeed.exe*') { throw 'Wrong service owner' }
    if ($taskService.State -ne 'Stopped') { Stop-Service SiltokSpeed -Force }
    $taskBundle = Join-Path $taskInstall 'python\Lib\site-packages\silitok_speed\__init__.pyd'
    if ((Get-FileHash -LiteralPath $taskBundle -Algorithm SHA256).Hash.ToLowerInvariant() -ne $BundleSha256) { throw 'Installed bundle differs from accepted p04' }
    [ordered]@{status='installed_service_stopped'; observed_utc=[DateTime]::UtcNow.ToString('o'); install_path=$taskInstall; launcher_admin=$true; session_id=(Get-Process -Id $PID).SessionId; installer_exitcode=$taskProcess.ExitCode; installer_sha256=$InstallerSha256; bundle_sha256=$BundleSha256} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskEvidence 'result.json') -Encoding utf8
    $taskDeadline = [DateTime]::UtcNow.AddMinutes(30)
    $taskProgress = Join-Path $taskEvidence 'suite-progress.json'
    while ($true) {
        if (Test-Path -LiteralPath $taskProgress) {
            $taskState = Get-Content -LiteralPath $taskProgress -Raw | ConvertFrom-Json
            if ($taskState.status -in @('passed','failed')) { break }
        }
        if ([DateTime]::UtcNow -gt $taskDeadline) { throw 'Bounded installed acceptance wait timed out' }
        Start-Sleep -Seconds 2
    }
    if ($taskState.status -ne 'passed' -or $taskState.scope -ne 'actual_installed_payload_not_clean_os') { throw 'Actual installed suites failed' }
    $taskExit = 0
} catch {
    [ordered]@{status='failed'; observed_utc=[DateTime]::UtcNow.ToString('o'); error=$_.ToString()} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRun 'install-i07-error.json') -Encoding utf8
    Write-Error -ErrorRecord $_ -ErrorAction Continue
} finally {
    if ($taskOwned) {
        try {
            # 通用清理器先证明绝对路径、安装前快照和服务归属，再卸载本次实例。
            & C:\workspace\tools\bin\cleanup-flash-speed.ps1 -InstallPath $taskInstall -RunRoot $taskRun -EvidenceDir (Join-Path $taskEvidence 'cleanup-preview')
            & C:\workspace\tools\bin\cleanup-flash-speed.ps1 -InstallPath $taskInstall -RunRoot $taskRun -Apply
            $taskAfterPath = Join-Path $taskEvidence 'cleanup\after.json'
            $taskAfter = Get-Content -LiteralPath $taskAfterPath -Raw | ConvertFrom-Json -AsHashtable
            if (-not $taskAfter.ok) { throw 'Owned installer cleanup failed' }
            $taskAfter.scope = 'actual_installed_payload_not_clean_os'
            $taskAfter.actual_install_validated = ($taskState.status -eq 'passed')
            if (Test-Path -LiteralPath (Join-Path $taskEvidence 'suite-progress.json')) {
                $taskAfter.suite_progress_sha256 = (Get-FileHash -LiteralPath (Join-Path $taskEvidence 'suite-progress.json') -Algorithm SHA256).Hash.ToLowerInvariant()
            }
            $taskAfter | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $taskAfterPath -Encoding utf8
        } catch {
            $taskExit = 1
            [ordered]@{status='failed'; observed_utc=[DateTime]::UtcNow.ToString('o'); error=$_.ToString()} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskEvidence 'cleanup-error.json') -Encoding utf8
            Write-Error -ErrorRecord $_ -ErrorAction Continue
        }
    }
    if (Test-Path -LiteralPath (Join-Path $taskEvidence 'elevated.log')) { Stop-Transcript | Out-Null }
}
exit $taskExit
