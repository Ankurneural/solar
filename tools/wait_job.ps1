<#
Block until runs/jobs/<Name>.status reads DONE (used to queue detached jobs one after another):

  powershell -NoProfile -ExecutionPolicy Bypass -File tools/wait_job.ps1 -Name c2_seed2

Exits 0 when the predecessor finished with exit code 0, else 1 (the queued command can then stop).
#>
param([Parameter(Mandatory = $true)][string]$Name, [int]$PollSeconds = 60)
$status = Join-Path (Split-Path -Parent $PSScriptRoot) "runs\jobs\$Name.status"
while ($true) {
    $s = if (Test-Path $status) { (Get-Content $status -Raw).Trim() } else { "" }
    if ($s -like "DONE*") { "waited for $Name -> $s"; if ($s -eq "DONE 0") { exit 0 } else { exit 1 } }
    Start-Sleep $PollSeconds
}
