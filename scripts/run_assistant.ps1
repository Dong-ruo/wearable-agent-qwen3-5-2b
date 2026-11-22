param(
    [ValidateSet('plan','check','model','chat','serve','smoke','evaluate','compare')]
    [string]$Task = 'plan',
    [string]$Config = 'configs/assistant.json',
    [string]$Output = '',
    [string]$Python = '',
    [switch]$Resume,
    [switch]$Execute,
    [switch]$Pascal
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Join-Path $PSScriptRoot '..')
if (-not $Python) { $Python = if ($Task -eq 'model') { '.venv-train/Scripts/python.exe' } else { '.venv-data/Scripts/python.exe' } }
$Python = (Resolve-Path -LiteralPath $Python).ProviderPath
if ($Task -eq 'check') {
    & $Python -B -X utf8 -m pytest tests/test_wearable_assistant.py -q -p no:cacheprovider
} elseif ($Task -eq 'model') {
    $Arguments = @('-B','-X','utf8','-m','wearable_assistant.model_server')
    if ($Pascal) { $Arguments += @('--dtype','float32','--nf4','--attention','eager','--max-context','4096') }
    if ($Execute) { $Arguments += '--execute' }
    & $Python @Arguments
} else {
    $Arguments = @('-B','-X','utf8','-m','wearable_assistant','--config',$Config)
    if ($Task -in @('smoke','evaluate','compare')) {
        if (-not $Output) { $Output = 'outputs/assistant/' + $Task + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss') }
        $Arguments += $(if ($Task -eq 'compare') { 'compare' } else { 'evaluate' })
        $Arguments += @('--output',$Output)
        if ($Task -eq 'smoke') { $Arguments += @('--splits','val','--limit','5') }
        if ($Resume) { $Arguments += '--resume' }
    } else { $Arguments += $Task }
    if ($Execute -and $Task -ne 'plan') { $Arguments += '--execute' }
    & $Python @Arguments
}
if ($LASTEXITCODE -ne 0) { throw "Assistant command failed with exit code $LASTEXITCODE" }
