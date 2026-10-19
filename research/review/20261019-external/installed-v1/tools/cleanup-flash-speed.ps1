<# Dry-run by default. -Apply is only for the explicit, test-owned installer cleanup.
   Source, weights, toolchains, venvs, build artifacts and acceptance evidence stay intact.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$InstallPath,
    [string]$EvidenceDir = '',
    [string]$RunRoot = 'C:\workspace\runs\siltok-speed\flash-accept-20261015',
    [switch]$Apply
)
$ErrorActionPreference = 'Stop'
$InstallPath = [IO.Path]::GetFullPath($InstallPath).TrimEnd('\')
if ($InstallPath -notmatch '^C:\\workspace\\build\\siltok-speed\\i\d{2}$') {
    throw "Cleanup install path must be an explicit test instance C:\workspace\build\siltok-speed\iNN: $InstallPath"
}
$installId = Split-Path -Leaf $InstallPath
$installEvidence = Join-Path ([IO.Path]::GetFullPath($RunRoot)) ('install-'+$installId)
if (-not $installEvidence.StartsWith('C:\workspace\runs\siltok-speed\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Cleanup ownership evidence must remain inside the Speed runs directory'
}
$ownership = Get-Content -LiteralPath (Join-Path $installEvidence 'installed.json') -Raw | ConvertFrom-Json -DateKind String
$baseline = Get-Content -LiteralPath (Join-Path $installEvidence 'before.json') -Raw | ConvertFrom-Json -DateKind String
if ($ownership.install_path -ne $InstallPath -or $baseline.data_present -or $baseline.install_present -or $baseline.service) {
    throw 'Installation snapshots do not prove this test instance ownership'
}
if (-not $EvidenceDir) { $EvidenceDir = Join-Path $installEvidence 'cleanup' }
$EvidenceDir = [IO.Path]::GetFullPath($EvidenceDir).TrimEnd('\')
if (-not $EvidenceDir.StartsWith(($installEvidence+'\'), [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Evidence directory must stay under the workspace acceptance directory'
}
$DataPath = 'C:\ProgramData\Siltok\Speed'
$InitialLog = Join-Path $installEvidence 'install.log'
$PriorUninstallLog = Join-Path $installEvidence 'unused-prior-uninstall.log'
$EventSourceKey = 'HKLM:\SYSTEM\CurrentControlSet\Services\EventLog\Application\SiltokSpeed'
$ServiceKey = 'HKLM:\SYSTEM\CurrentControlSet\Services\SiltokSpeed'
$UninstallKeys = @(
    'HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{A3F1C2E4-7B6D-4E9A-8C12-9D4E5F6A7B80}}_is1',
    'HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{A3F1C2E4-7B6D-4E9A-8C12-9D4E5F6A7B80}_is1',
    'HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\{A3F1C2E4-7B6D-4E9A-8C12-9D4E5F6A7B80}}_is1',
    'HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\{A3F1C2E4-7B6D-4E9A-8C12-9D4E5F6A7B80}_is1'
)
$AllowedTrees = [Collections.Generic.List[string]]::new()
$AllowedTrees.Add($InstallPath)
$AllowedTrees.Add($DataPath)
$InstalledTestTrees = @(
    ("C:\workspace\cache\triton\installed-"+$installId),
    ("C:\workspace\cache\ext\installed-"+$installId),
    (Join-Path $installEvidence 'tmp')
)
foreach ($path in $InstalledTestTrees) { $AllowedTrees.Add($path) }
$UserTemp = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\')

function Test-InInstall([string]$Path) {
    # 只认本次明确指定的安装目录，不按进程名称猜归属。
    return $Path -and $Path.StartsWith($InstallPath + '\', [StringComparison]::OrdinalIgnoreCase)
}
function Get-OwnedProcesses {
    # 查询路径不终止进程；执行清理时只处理安装目录里的残留可执行文件。
    return @(Get-CimInstance Win32_Process | Where-Object { Test-InInstall $_.ExecutablePath } |
        Select-Object ProcessId, ParentProcessId, ExecutablePath, CommandLine)
}
function Assert-ServiceOwnership {
    # Inno 卸载会按服务名操作，先核对 SCM 的可执行文件路径属于本测试实例。
    $service = Get-CimInstance Win32_Service -Filter "Name='SiltokSpeed'"
    if ($service -and $service.PathName -notmatch ('^"?' + [regex]::Escape($InstallPath + '\SiltokSpeed.exe') + '(?:"|\s|$)')) {
        throw "SiltokSpeed belongs to another install; refusing cleanup: $($service.PathName)"
    }
    return $service
}
function Remove-OwnedTree([string]$Path) {
    # 每次递归删除前重新核对绝对路径白名单，禁止重解析点指到模型或源码目录。
    $full = [IO.Path]::GetFullPath($Path).TrimEnd('\')
    if (-not $AllowedTrees.Contains($full)) { throw "Recursive target is not in the verified allowlist: $full" }
    if (-not (Test-Path -LiteralPath $full)) { return }
    $item = Get-Item -LiteralPath $full -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Refusing recursive removal of reparse point: $full" }
    $links = @(Get-ChildItem -LiteralPath $full -Recurse -Force -Attributes ReparsePoint -ErrorAction Stop)
    if ($links.Count) { throw "Refusing tree containing reparse points: $full" }
    Remove-Item -LiteralPath $full -Recurse -Force
}
function Get-ShortcutInventory {
    # 仅清理由当前安装目录可执行文件作为目标的 Siltok 快捷方式。
    $folders = @('C:\Users\UserPC\Desktop', 'C:\Users\Public\Desktop',
        'C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Siltok',
        'C:\Users\UserPC\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Siltok')
    $shell = New-Object -ComObject WScript.Shell
    foreach ($folder in $folders) {
        if (Test-Path -LiteralPath $folder) {
            foreach ($file in Get-ChildItem -LiteralPath $folder -Filter '*Siltok*.lnk' -Recurse -File -ErrorAction Stop) {
                $target = $shell.CreateShortcut($file.FullName).TargetPath
                [pscustomobject]@{ Path=$file.FullName; Target=$target; Owned=[bool](Test-InInstall $target) }
            }
        }
    }
}
function Get-UninstallTempDirs([string[]]$Logs) {
    # 从本次卸载日志提取实际临时目录，限定在用户 TEMP 内的 Inno 卸载目录。
    $found = @()
    foreach ($log in $Logs) {
        if (-not (Test-Path -LiteralPath $log)) { continue }
        foreach ($line in Get-Content -LiteralPath $log -Encoding UTF8) {
            if ($line -match 'Current Uninstall EXE: (.+)\\_unins\.tmp$') {
                $path = [IO.Path]::GetFullPath($Matches[1]).TrimEnd('\')
                if ((Split-Path -Parent $path) -ne $UserTemp -or (Split-Path -Leaf $path) -notmatch '^is-[A-Za-z0-9]+-uninstall\.tmp$') {
                    throw "Uninstall temporary path is outside the verified scope: $path"
                }
                $found += $path
            }
        }
    }
    return @($found | Sort-Object -Unique)
}

# ProgramData 归属由首次安装日志和 NTFS 创建时间双重绑定；未执行删除。
if (Test-Path -LiteralPath $DataPath) {
    $created = (Get-Item -LiteralPath $DataPath -Force).CreationTimeUtc.ToString('o')
    if ($created -ne $ownership.data_creation_utc) { throw "Unexpected ProgramData ownership timestamp: $created" }
    $proof = Select-String -LiteralPath $InitialLog -SimpleMatch 'Creating directory: C:\ProgramData\Siltok\Speed' | Select-Object -First 1
    if (-not $proof) { throw 'Initial installer log does not prove ownership of ProgramData data root' }
}
$service = Assert-ServiceOwnership
$beforeProcesses = Get-OwnedProcesses
$shortcuts = @(Get-ShortcutInventory)
foreach ($key in $UninstallKeys) {
    if (Test-Path -LiteralPath $key) {
        $entry = Get-ItemProperty -LiteralPath $key
        if ($entry.InstallLocation -and [IO.Path]::GetFullPath($entry.InstallLocation).TrimEnd('\') -ne $InstallPath) {
            throw "Uninstall registration belongs to another install: $($entry.InstallLocation)"
        }
    }
}
# 不导入产品、不初始化 GPU；从安装包的受保护 bundle 尾部提取实际内容寻址缓存名。
$BundleHash = $null
$bundle = Join-Path $InstallPath 'python\Lib\site-packages\silitok_speed\__init__.pyd'
if (Test-Path -LiteralPath $bundle) {
    $bytes = [IO.File]::ReadAllBytes($bundle)
    if ([Text.Encoding]::ASCII.GetString($bytes, $bytes.Length-16, 16) -ne 'SPEED-BUNDLE-V2!') { throw 'Invalid installed bundle trailer' }
    $length = [BitConverter]::ToUInt64($bytes, $bytes.Length-24)
    if ($length -lt 28 -or $length -gt ($bytes.Length-24)) { throw 'Invalid installed bundle payload length' }
    $sha = [Security.Cryptography.SHA256]::Create()
    $BundleHash = ([BitConverter]::ToString($sha.ComputeHash($bytes, [int]($bytes.Length-24-$length), [int]$length))).Replace('-','').ToLowerInvariant()
    $sha.Dispose()
    $bytes = $null
} else {
    # 使用本轮安装快照记录的缓存身份，允许卸载后复核；不使用历史固定哈希。
    $BundleHash = $ownership.bundle_cache_hash
}
if ($BundleHash -ne $ownership.bundle_cache_hash -or $BundleHash -notmatch '^[a-f0-9]{64}$') {
    throw 'Installed bundle cache identity differs from the ownership snapshot'
}
$CachePaths = @((Join-Path $UserTemp ('siltok-speed-assets\'+$BundleHash)),
    ('C:\Windows\Temp\siltok-speed-assets\'+$BundleHash)) | Select-Object -Unique
foreach ($path in $CachePaths) { $AllowedTrees.Add([IO.Path]::GetFullPath($path).TrimEnd('\')) }
$uninstallLog = Join-Path $EvidenceDir 'uninstall.log'
$TempDirs = @(Get-UninstallTempDirs @($PriorUninstallLog, $uninstallLog))
foreach ($path in $TempDirs) { $AllowedTrees.Add($path) }
$DataBefore = @(if (Test-Path -LiteralPath $DataPath) {
    Get-ChildItem -LiteralPath $DataPath -Recurse -Force | Select-Object FullName,CreationTime,LastWriteTime,Length
})
$report = [ordered]@{
    apply=[bool]$Apply; install_path=$InstallPath; bundle_cache_hash=$BundleHash;
    service_before=if($service){@{path=$service.PathName;state=$service.State}}else{$null};
    processes_before=$beforeProcesses; data_before=$DataBefore; shortcuts=$shortcuts;
    cache_paths=$CachePaths; uninstall_temp_dirs=$TempDirs; installed_test_trees=$InstalledTestTrees;
    registry_paths=@($ServiceKey,$EventSourceKey)+$UninstallKeys;
    protected_workspace_paths=@('C:\workspace\codebase\siltok-speed','C:\workspace\models\siltok-speed','C:\workspace\envs\siltok-speed','C:\workspace\tools');
    plan=@('Uninstall the explicitly owned test instance using its own Inno uninstaller',
           'Stop only remaining processes whose executable path is inside that exact test install',
           'Copy ProgramData evidence into the workspace, then delete only proven test-owned data',
           'Remove this exact application EventLog source, service/AppID leftovers and owned shortcuts',
           'Remove only this bundle hash from user/SYSTEM temp caches and logged Inno uninstall temp directories',
           'Verify application/service/registrations/cache absence and preserve source/models/venvs/tools')
}
New-Item -ItemType Directory -Force -Path $EvidenceDir | Out-Null
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $EvidenceDir 'before.json') -Encoding UTF8
if (-not $Apply) {
    $report | ConvertTo-Json -Depth 8
    Write-Host 'Dry run only. No application, service, registry, data, shortcut or cache was removed.'
    return
}

$uninstaller = Join-Path $InstallPath 'unins000.exe'
if (Test-Path -LiteralPath $uninstaller) {
    $arguments = @('/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART',('/LOG="'+$uninstallLog+'"'))
    $process = Start-Process -FilePath $uninstaller -ArgumentList $arguments -WindowStyle Hidden -Wait -PassThru
    $report['uninstaller_exit'] = $process.ExitCode
    if ($process.ExitCode -ne 0) { throw "Inno uninstall failed: $($process.ExitCode)" }
}
$remainingService = Assert-ServiceOwnership
if ($remainingService) {
    if ($remainingService.State -ne 'Stopped') { Stop-Service -Name SiltokSpeed -Force }
    & sc.exe delete SiltokSpeed | Out-Null
    if ($LASTEXITCODE -notin @(0,1060)) { throw "Owned service deletion failed: $LASTEXITCODE" }
}
foreach ($owned in Get-OwnedProcesses) {
    # 重新读取 PID 的路径，避免旧快照 PID 被系统重用后误杀别的任务。
    $current = Get-CimInstance Win32_Process -Filter "ProcessId=$($owned.ProcessId)"
    if ($current -and (Test-InInstall $current.ExecutablePath)) { Stop-Process -Id $owned.ProcessId -Force }
}
if ((Get-OwnedProcesses).Count) { throw 'Owned test application processes have not exited' }
if (Test-Path -LiteralPath $DataPath) {
    Copy-Item -LiteralPath $DataPath -Destination (Join-Path $EvidenceDir 'programdata') -Recurse -Force
    Remove-OwnedTree $DataPath
}
Remove-OwnedTree $InstallPath
foreach ($path in $CachePaths) { Remove-OwnedTree $path }
foreach ($path in $InstalledTestTrees) { Remove-OwnedTree $path }
$TempDirs = @(Get-UninstallTempDirs @($PriorUninstallLog, $uninstallLog))
foreach ($path in $TempDirs) {
    if (-not $AllowedTrees.Contains($path)) { $AllowedTrees.Add($path) }
    Remove-OwnedTree $path
}
foreach ($shortcut in $shortcuts | Where-Object Owned) { Remove-Item -LiteralPath $shortcut.Path -Force }
foreach ($key in $UninstallKeys + @($EventSourceKey)) {
    if (Test-Path -LiteralPath $key) { Remove-Item -LiteralPath $key -Recurse -Force }
}
$applicationLogKey = 'HKLM:\SYSTEM\CurrentControlSet\Services\EventLog\Application'
$properties = Get-ItemProperty -LiteralPath $applicationLogKey
if ($properties.PSObject.Properties.Name -contains 'Sources' -and $properties.Sources -contains 'SiltokSpeed') {
    $remainingSources = [string[]]@($properties.Sources | Where-Object { $_ -ne 'SiltokSpeed' })
    Set-ItemProperty -LiteralPath $applicationLogKey -Name Sources -Value $remainingSources
}
# 父目录只在为空时非递归删除；不触碰其他产品、缓存或模型内容。
foreach ($parent in @('C:\ProgramData\Siltok', (Join-Path $UserTemp 'siltok-speed-assets'), 'C:\Windows\Temp\siltok-speed-assets',
    'C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Siltok',
    'C:\Users\UserPC\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Siltok')) {
    if ((Test-Path -LiteralPath $parent) -and @([IO.Directory]::GetFileSystemEntries($parent)).Count -eq 0) {
        Remove-Item -LiteralPath $parent -Force
    }
}
$emptyTestModels = Join-Path $installEvidence 'empty-models'
if ((Test-Path -LiteralPath $emptyTestModels) -and @([IO.Directory]::GetFileSystemEntries($emptyTestModels)).Count -eq 0) {
    Remove-Item -LiteralPath $emptyTestModels -Force
}
$checks = [ordered]@{
    install_directory_absent= -not (Test-Path -LiteralPath $InstallPath);
    programdata_absent= -not (Test-Path -LiteralPath $DataPath);
    service_absent= -not [bool](Get-CimInstance Win32_Service -Filter "Name='SiltokSpeed'");
    owned_processes_absent= (Get-OwnedProcesses).Count -eq 0;
    service_registry_absent= -not (Test-Path -LiteralPath $ServiceKey);
    event_source_absent= -not (Test-Path -LiteralPath $EventSourceKey);
    uninstall_keys_absent= @($UninstallKeys | Where-Object { Test-Path -LiteralPath $_ }).Count -eq 0;
    bundle_caches_absent= @($CachePaths | Where-Object { Test-Path -LiteralPath $_ }).Count -eq 0;
    installed_test_trees_absent= @($InstalledTestTrees | Where-Object { Test-Path -LiteralPath $_ }).Count -eq 0;
    logged_uninstaller_temps_absent= @($TempDirs | Where-Object { Test-Path -LiteralPath $_ }).Count -eq 0;
    owned_shortcuts_absent= @(Get-ShortcutInventory | Where-Object Owned).Count -eq 0;
    source_preserved= Test-Path -LiteralPath 'C:\workspace\codebase\siltok-speed';
    models_preserved= Test-Path -LiteralPath 'C:\workspace\models\siltok-speed';
    venvs_preserved= Test-Path -LiteralPath 'C:\workspace\envs\siltok-speed';
    tools_preserved= Test-Path -LiteralPath 'C:\workspace\tools'
}
$report['after_checks'] = $checks
$report['ok'] = @($checks.Values | Where-Object { -not $_ }).Count -eq 0
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $EvidenceDir 'after.json') -Encoding UTF8
$report | ConvertTo-Json -Depth 8
if (-not $report.ok) { throw 'Cleanup absence checks failed; see after.json' }

