param(
    [ValidateSet('smoke', 'formal', 'all')][string]$Stage = 'smoke',
    [int]$Seed = 42,
    [switch]$Overwrite
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$taskProjectRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$taskLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($null -ne $taskLauncher) {
    $taskPython = $taskLauncher.Source
    $taskPythonPrefix = @('-3.10', '-E', '-S', '-X', 'utf8')
}
else {
    $taskPython = Join-Path $taskProjectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) {
        throw 'Python 3.10 launcher and project interpreter were not found.'
    }
    $taskPythonPrefix = @('-E', '-S', '-X', 'utf8')
}

function Invoke-TaskPython {
    param([string[]]$TaskArguments)
    & $taskPython @taskPythonPrefix @TaskArguments
    $taskExitCode = $LASTEXITCODE
    if ($taskExitCode -ne 0) {
        throw ('Command failed (exit code {0}): {1} {2} {3}' -f $taskExitCode, $taskPython, ($taskPythonPrefix -join ' '), ($TaskArguments -join ' '))
    }
}

Push-Location -LiteralPath $taskProjectRoot
try {
    $taskChecks = 'outputs/wearable_tasks/unit-tests.json'
    $taskSmoke = 'data/wearable_tasks/smoke'
    $taskFormal = 'data/wearable_tasks/formal'
    Invoke-TaskPython -TaskArguments @('-m', 'wearable_tasks', 'check', '--output', $taskChecks)
    if ($Stage -eq 'smoke' -or $Stage -eq 'all') {
        $taskArgs = @('-m', 'wearable_tasks', 'build', '--config', 'wearable_tasks/configs/smoke.json',
            '--output', $taskSmoke, '--checks-report', $taskChecks, '--seed', "$Seed")
        if ($Overwrite) { $taskArgs += '--overwrite' }
        Invoke-TaskPython -TaskArguments $taskArgs
    }
    if ($Stage -eq 'formal' -or $Stage -eq 'all') {
        $taskArgs = @('-m', 'wearable_tasks', 'build', '--config', 'wearable_tasks/configs/formal.json',
            '--output', $taskFormal, '--checks-report', $taskChecks,
            '--smoke-report', "$taskSmoke/validation-report.json", '--seed', "$Seed")
        if ($Overwrite) { $taskArgs += '--overwrite' }
        Invoke-TaskPython -TaskArguments $taskArgs
    }
}
finally {
    Pop-Location
}
