param(
    [string]$Python = "python",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunArgs
)
$ErrorActionPreference = 'Stop'
& $Python (Join-Path $PSScriptRoot 'run_5cv.py') @RunArgs
exit $LASTEXITCODE
