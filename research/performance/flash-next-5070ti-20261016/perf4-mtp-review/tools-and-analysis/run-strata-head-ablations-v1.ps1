[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$taskRoot = 'C:\workspace\runs\strata\flash-perf-20261016'
$supervisor = 'C:\workspace\runs\siltok-speed\flash-perf-20261016\consumer-tools\supervise.py'
$controllerPython = 'C:\workspace\envs\siltok-speed\flash-perf-cand3\Scripts\python.exe'
$strataPython = 'C:\workspace\envs\strata\flash-perf\Scripts\python.exe'
$cases = @(
    @{ id='reference-mtp-cjk-fixed-3x-v1'; extra=@(); sync=$false },
    @{ id='reference-mtp-en-fixed-3x-v1'; extra=@('--draft-vocab','C:\workspace\wt\strata\flash-perf\data\draft_vocab_en.bin'); sync=$false },
    @{ id='reference-mtp-commitsync-fixed-3x-v1'; extra=@(); sync=$true }
)
foreach($case in $cases) {
    $env:STRATA_MTP_FULL_HEAD='0'
    if($case.sync) {$env:STRATA_COMMIT_SYNC='1'} else {Remove-Item Env:STRATA_COMMIT_SYNC -ErrorAction SilentlyContinue}
    $prefix=Join-Path $taskRoot $case.id
    $arguments=@('-B',$supervisor,'--prefix',$prefix,'--cwd',$taskRoot,'--timeout','900','--',
        $strataPython,'-B','probe-ablation-v1.py','--output',"$($case.id).json",'--mtp','--suffix','0','--slots','7019','--repetitions','3')
    $arguments += $case.extra
    & $controllerPython @arguments
    if($LASTEXITCODE -ne 0){throw "Strata ablation failed: $($case.id)"}
}
