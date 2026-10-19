[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidatePattern('^i\d{2}$')][string]$InstallId,
    [Parameter(Mandatory)][ValidateSet('before','installed')][string]$Phase,
    [string]$RunRoot = 'C:\workspace\runs\siltok-speed\flash-accept-20261015'
)
$ErrorActionPreference = 'Stop'
$testInstall = [IO.Path]::GetFullPath("C:\workspace\build\siltok-speed\$InstallId")
$testEvidence = Join-Path ([IO.Path]::GetFullPath($RunRoot)) "install-$InstallId"
if (-not $testEvidence.StartsWith('C:\workspace\runs\siltok-speed\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Installer evidence must remain inside the Speed runs directory'
}
$testData = 'C:\ProgramData\Siltok\Speed'
$eventKey = 'HKLM:\SYSTEM\CurrentControlSet\Services\EventLog\Application\SiltokSpeed'
$service = Get-CimInstance Win32_Service -Filter "Name='SiltokSpeed'"
$uninstallKeys = @(
    'HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{A3F1C2E4-7B6D-4E9A-8C12-9D4E5F6A7B80}}_is1',
    'HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{A3F1C2E4-7B6D-4E9A-8C12-9D4E5F6A7B80}_is1',
    'HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\{A3F1C2E4-7B6D-4E9A-8C12-9D4E5F6A7B80}}_is1',
    'HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\{A3F1C2E4-7B6D-4E9A-8C12-9D4E5F6A7B80}_is1'
)
$presentKeys = @($uninstallKeys | Where-Object { Test-Path -LiteralPath $_ })
$output = Join-Path $testEvidence "$Phase.json"
if (Test-Path -LiteralPath $output) { throw "Preserve existing installer snapshot: $output" }
$snapshot = [ordered]@{
    format = 1
    observed_utc = [DateTime]::UtcNow.ToString('o')
    phase = $Phase
    install_path = $testInstall
    data_path = $testData
    service = if ($service) { @{ path=$service.PathName; state=$service.State; pid=$service.ProcessId } } else { $null }
    install_present = Test-Path -LiteralPath $testInstall
    data_present = Test-Path -LiteralPath $testData
    event_source_present = Test-Path -LiteralPath $eventKey
    uninstall_keys = $presentKeys
}
if ($Phase -eq 'before') {
    # 安装前必须证明应用、全局服务和共享数据根都不存在，不能把现有用户数据收编为测试数据。
    if ($snapshot.install_present -or $snapshot.data_present -or $service -or
        $snapshot.event_source_present -or $presentKeys.Count) {
        throw 'Existing installation/data/service must be investigated before installer acceptance'
    }
} else {
    # 结合本轮安装前快照、安装日志和精确服务路径，建立后续清理的归属凭据。
    $before = Get-Content -LiteralPath (Join-Path $testEvidence 'before.json') -Raw | ConvertFrom-Json
    if ($before.install_path -ne $testInstall -or $before.data_present -or $before.service) {
        throw 'Installer baseline does not prove a fresh test instance'
    }
    if (-not $service -or $service.PathName -notmatch ('^"?' + [regex]::Escape($testInstall + '\SiltokSpeed.exe') + '(?:"|\s|$)')) {
        throw 'Installed service belongs to a different application directory'
    }
    $installLog = Join-Path $testEvidence 'install.log'
    if (-not (Select-String -LiteralPath $installLog -SimpleMatch "Creating directory: $testData")) {
        throw 'Installer log does not prove ownership of the new ProgramData root'
    }
    $snapshot.data_creation_utc = (Get-Item -LiteralPath $testData).CreationTimeUtc.ToString('o')
    $bundle = Join-Path $testInstall 'python\Lib\site-packages\silitok_speed\__init__.pyd'
    $payload = [IO.File]::ReadAllBytes($bundle)
    if ([Text.Encoding]::ASCII.GetString($payload, $payload.Length-16, 16) -ne 'SPEED-BUNDLE-V2!') {
        throw 'Installed product is not a protected Speed bundle'
    }
    $length = [BitConverter]::ToUInt64($payload, $payload.Length-24)
    if ($length -lt 28 -or $length -gt $payload.Length-24) { throw 'Invalid Speed bundle payload length' }
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $snapshot.bundle_cache_hash = ([BitConverter]::ToString($sha.ComputeHash($payload,
            [int]($payload.Length-24-$length), [int]$length))).Replace('-','').ToLowerInvariant()
    } finally { $sha.Dispose() }
    $snapshot.bundle_sha256 = (Get-FileHash -LiteralPath $bundle).Hash.ToLowerInvariant()
    $snapshot.install_log_sha256 = (Get-FileHash -LiteralPath $installLog).Hash.ToLowerInvariant()
}
New-Item -ItemType Directory -Path $testEvidence -Force | Out-Null
$snapshot | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $output -Encoding utf8
$snapshot | ConvertTo-Json -Depth 6
