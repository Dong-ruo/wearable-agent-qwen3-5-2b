param(
    [Parameter(Mandatory = $true)][string]$PythonExe,
    [ValidateSet('inventory', 'mix', 'audit')][string]$Stage = 'inventory',
    [string]$TokenizerDirectory = 'models/Qwen3.5-2B',
    [string]$Manifest = 'configs/sft-data-v1.json',
    [string]$Policy = 'configs/sft-mixture-v1.example.json',
    [string]$InventoryDirectory = 'data/sft_v1_inventory',
    [string]$DatasetDirectory = 'data/sft_v1'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$sftProjectRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
# Resolve an explicit interpreter before changing working directory.
if (Test-Path -LiteralPath $PythonExe -PathType Leaf) {
    $sftPython = (Resolve-Path -LiteralPath $PythonExe).ProviderPath
}
else {
    $sftPython = (Get-Command -Name $PythonExe -CommandType Application -ErrorAction Stop).Source
}

Push-Location -LiteralPath $sftProjectRoot
try {
    $sftArguments = @('-X', 'utf8', '-m', 'lightagent.sft_data')
    switch ($Stage) {
        'inventory' {
            $sftArguments += @('prepare', '--manifest', $Manifest, '--tokenizer', $TokenizerDirectory,
                              '--output', $InventoryDirectory)
        }
        'mix' {
            $sftArguments += @('mix', '--dataset', $InventoryDirectory, '--policy', $Policy,
                              '--output', $DatasetDirectory)
        }
        'audit' {
            $sftArguments += @('audit-mask', '--dataset', $DatasetDirectory, '--tokenizer', $TokenizerDirectory)
        }
    }
    & $sftPython @sftArguments
    if ($LASTEXITCODE -ne 0) {
        throw ('SFT data preparation failed with exit code {0}; no training was requested.' -f $LASTEXITCODE)
    }
}
finally {
    Pop-Location
}
