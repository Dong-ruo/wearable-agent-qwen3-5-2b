param([switch]$Recycle)

$ErrorActionPreference = 'Stop'
$WorkspaceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath.TrimEnd('\')
$WorkspacePrefix = $WorkspaceRoot + [IO.Path]::DirectorySeparatorChar
$ReportPath = Join-Path $WorkspaceRoot $(if ($Recycle) { 'reports/workspace-cleanup.json' } else { 'reports/workspace-cleanup-preview.json' })

function Assert-WorkspaceTarget([string]$Candidate) {
    $ResolvedTarget = (Resolve-Path -LiteralPath $Candidate).ProviderPath.TrimEnd('\')
    if (-not $ResolvedTarget.StartsWith($WorkspacePrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Target outside workspace: $ResolvedTarget"
    }
    $Ancestor = Get-Item -LiteralPath $ResolvedTarget -Force
    while ($null -ne $Ancestor -and $Ancestor.FullName -ne $WorkspaceRoot) {
        if ($Ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Reparse point refused: $($Ancestor.FullName)"
        }
        $Ancestor = if ($Ancestor.PSIsContainer) { $Ancestor.Parent } else { $Ancestor.Directory }
    }
    return $ResolvedTarget
}

function Save-CleanupReport {
    $script:CleanupReport | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $ReportPath -Encoding UTF8
}

$RelativeTargets = @(
    'models/Qwen3.5-2B/.cache',
    'data/sft_v1_inventory.invalid-lengths-20261116',
    '.pytest_cache',
    'src/lightagent/__pycache__',
    'src/lightagent/integrations/__pycache__',
    'tests/__pycache__',
    'wearable_tasks/__pycache__',
    'wearable_tasks/tests/__pycache__'
)
$Targets = @()
foreach ($RelativeTarget in $RelativeTargets) {
    $Candidate = Join-Path $WorkspaceRoot $RelativeTarget
    if (-not (Test-Path -LiteralPath $Candidate)) { continue }
    $ResolvedTarget = Assert-WorkspaceTarget $Candidate
    $Descendants = @(Get-ChildItem -LiteralPath $ResolvedTarget -Recurse -Force)
    if (@($Descendants | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count -gt 0) {
        throw "Reparse point below target: $ResolvedTarget"
    }
    $Files = @($Descendants | Where-Object { -not $_.PSIsContainer })
    $Targets += [ordered]@{
        path = $ResolvedTarget
        bytes = [long](($Files | Measure-Object -Property Length -Sum).Sum)
        file_count = $Files.Count
        status = 'planned'
        files = @($Files | ForEach-Object { [ordered]@{ path = $_.FullName; bytes = $_.Length } })
    }
}
$script:CleanupReport = [ordered]@{
    status = 'planned'
    mode = 'windows_recycle_bin_only_no_permanent_delete'
    started_at = (Get-Date -Format o)
    workspace = $WorkspaceRoot
    recycled_bytes = 0L
    recycled_file_count = 0
    disk_space_reclaimed = $false
    targets = $Targets
    archived = @()
    protected = @()
    preserved = @('source/configuration/tests/docs', 'all virtual environments',
        'valid inventory and formal SFT data', 'raw datasets and licenses',
        'formal and smoke executable task datasets', 'model weights and tokenizer',
        'all existing adapters, experiment evidence and reports')
}
Save-CleanupReport
if (-not $Recycle) {
    $Targets | ForEach-Object { [pscustomobject]$_ } | Select-Object path, bytes, file_count | Format-Table -AutoSize
    Write-Output 'Preview only. -Recycle sends these exact directories to the Windows Recycle Bin.'
    return
}
if (-not $IsWindows) { throw 'Recycle mode requires Windows PowerShell 7+' }
Add-Type -AssemblyName Microsoft.VisualBasic.Core

try {
    $ModelPath = Join-Path $WorkspaceRoot 'models/Qwen3.5-2B/model.safetensors-00001-of-00001.safetensors'
    $ExpectedModelHash = 'aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1'
    Write-Output 'Verifying completed model weights before recycling interrupted downloads...'
    $ActualModelHash = (Get-FileHash -LiteralPath $ModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualModelHash -ne $ExpectedModelHash) { throw 'Model hash mismatch; cleanup stopped' }
    $ModelBefore = Get-Item -LiteralPath $ModelPath
    $CleanupReport.model_integrity = [ordered]@{ path = $ModelPath; sha256 = $ActualModelHash; bytes = $ModelBefore.Length }

    foreach ($Dataset in @('data/sft_v1', 'data/sft_v1_inventory')) {
        $DatasetRoot = Join-Path $WorkspaceRoot $Dataset
        $Preparation = Get-Content -LiteralPath (Join-Path $DatasetRoot 'preparation-report.json') -Raw -Encoding UTF8 | ConvertFrom-Json
        foreach ($Asset in $Preparation.asset_sha256.PSObject.Properties) {
            $AssetPath = Join-Path $DatasetRoot $Asset.Name
            if ((Get-FileHash -LiteralPath $AssetPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Asset.Value) {
                throw "Dataset hash mismatch: $AssetPath"
            }
        }
    }

    $InvalidDirectory = Join-Path $WorkspaceRoot 'data/sft_v1_inventory.invalid-lengths-20261116'
    if (Test-Path -LiteralPath $InvalidDirectory) {
        $InvalidReport = Get-Content -LiteralPath (Join-Path $InvalidDirectory 'preparation-report.json') -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($InvalidReport.status -ne 'invalid_do_not_use') { throw 'Old inventory is not explicitly invalid' }
        $ArchiveRoot = Join-Path $WorkspaceRoot 'reports/archive/invalid-lengths-20261116'
        New-Item -ItemType Directory -Path $ArchiveRoot -Force | Out-Null
        $null = Assert-WorkspaceTarget $ArchiveRoot
        foreach ($RelativeFile in @('preparation-report.json', 'metadata/tokenizer.json', 'metadata/review-schema.example.json')) {
            $OriginalFile = Join-Path $InvalidDirectory $RelativeFile
            $ArchivedFile = Join-Path $ArchiveRoot $RelativeFile
            New-Item -ItemType Directory -Path (Split-Path -Parent $ArchivedFile) -Force | Out-Null
            $OriginalHash = (Get-FileHash -LiteralPath $OriginalFile -Algorithm SHA256).Hash.ToLowerInvariant()
            if (-not (Test-Path -LiteralPath $ArchivedFile)) { Copy-Item -LiteralPath $OriginalFile -Destination $ArchivedFile }
            if ((Get-FileHash -LiteralPath $ArchivedFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne $OriginalHash) {
                throw 'Archive verification failed'
            }
            $CleanupReport.archived += [ordered]@{ original = $OriginalFile; archived = $ArchivedFile; sha256 = $OriginalHash }
        }
    }

    $ProtectedPaths = @('data/sft_v1/preparation-report.json', 'data/sft_v1/train.jsonl',
        'data/sft_v1/validation.jsonl', 'data/sft_v1/metadata/train.jsonl', 'data/sft_v1/metadata/validation.jsonl',
        'data/sft_v1/metadata/mask-audit.json', 'data/sft_v1_inventory/preparation-report.json',
        'models/Qwen3.5-2B/tokenizer.json', 'models/Qwen3.5-2B/tokenizer_config.json', 'models/Qwen3.5-2B/config.json',
        'artifacts/sft-v1-p1000-smoke/final-adapter/adapter_model.safetensors',
        'artifacts/sft-v1-p1000-smoke/run-manifest.json',
        '.venv/Scripts/python.exe', '.venv-data/Scripts/python.exe', '.venv-train/Scripts/python.exe')
    $CleanupReport.protected = @($ProtectedPaths | ForEach-Object {
        $ProtectedFile = Join-Path $WorkspaceRoot $_
        [ordered]@{ path = $ProtectedFile; sha256 = (Get-FileHash -LiteralPath $ProtectedFile -Algorithm SHA256).Hash.ToLowerInvariant() }
    })
    $CleanupReport.status = 'recycling'
    Save-CleanupReport
    foreach ($Target in $Targets) {
        $FinalTarget = Assert-WorkspaceTarget $Target.path
        # Recheck descendants immediately before the reversible operation.
        if (@(Get-ChildItem -LiteralPath $FinalTarget -Recurse -Force | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count -gt 0) {
            throw "Reparse point below target: $FinalTarget"
        }
        [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory($FinalTarget,
            [Microsoft.VisualBasic.FileIO.UIOption]::OnlyErrorDialogs,
            [Microsoft.VisualBasic.FileIO.RecycleOption]::SendToRecycleBin,
            [Microsoft.VisualBasic.FileIO.UICancelOption]::ThrowException)
        if (Test-Path -LiteralPath $FinalTarget) { throw "Target still present: $FinalTarget" }
        $Target.status = 'recycled'
        $CleanupReport.recycled_bytes += $Target.bytes
        $CleanupReport.recycled_file_count += $Target.file_count
        Save-CleanupReport
    }
    foreach ($ProtectedFile in $CleanupReport.protected) {
        if ((Get-FileHash -LiteralPath $ProtectedFile.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ProtectedFile.sha256) {
            throw "Protected file changed: $($ProtectedFile.path)"
        }
    }
    $ModelAfter = Get-Item -LiteralPath $ModelPath
    if ($ModelAfter.Length -ne $ModelBefore.Length -or $ModelAfter.LastWriteTimeUtc -ne $ModelBefore.LastWriteTimeUtc) {
        throw 'Model weight metadata changed'
    }
    $CleanupReport.status = 'completed'
    $CleanupReport.finished_at = Get-Date -Format o
    $CleanupReport.integrity_after_cleanup = 'Protected hashes unchanged; model weight size and timestamp unchanged'
    Save-CleanupReport
    [pscustomobject]@{ status = $CleanupReport.status; recycled_bytes = $CleanupReport.recycled_bytes;
        recycled_files = $CleanupReport.recycled_file_count; protected_hashes = $CleanupReport.protected.Count } | ConvertTo-Json
} catch {
    $CleanupReport.status = 'failed'
    $CleanupReport.error = $_.Exception.Message
    Save-CleanupReport
    throw
}
