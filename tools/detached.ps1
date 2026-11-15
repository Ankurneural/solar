<#
Run a job detached from the calling process tree, so it survives Claude / VS Code
restarts (a restart kills child processes of the shell that started them).

  powershell -NoProfile -File tools/detached.ps1 -Name refiner_w71_b1 -Command "<cmd.exe command line>"

The job runs under cmd.exe in the project directory via WMI (Win32_Process.Create),
which parents it to the WMI service instead of the caller. Output goes to
runs/jobs/<Name>.log; runs/jobs/<Name>.status holds RUNNING, then DONE <exit code>.
Jobs still apply src/resources.py limits (below-normal priority, CPU/GPU caps).
#>
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$Command
)
$root = Split-Path -Parent $PSScriptRoot
$jobs = Join-Path $root "runs\jobs"
New-Item -ItemType Directory -Force $jobs | Out-Null
$log = Join-Path $jobs "$Name.log"
$status = Join-Path $jobs "$Name.status"
$bat = Join-Path $jobs "$Name.cmd"
Set-Content -Encoding ascii $status "RUNNING"
# Paths in the .cmd are relative to its own location, so the repo can be moved/copied.
Set-Content -Encoding ascii $bat @"
@echo off
cd /d "%~dp0..\.."
($Command) > "runs\jobs\$Name.log" 2>&1
rem redirect first: "echo DONE 0> f" would redirect stream 0 and write nothing
>"runs\jobs\$Name.status" echo DONE %errorlevel%
"@
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "cmd.exe /c `"$bat`""; CurrentDirectory = $root }
if ($r.ReturnValue -ne 0) { Set-Content -Encoding ascii $status "FAILED to start ($($r.ReturnValue))"; exit 1 }
"started $Name (pid $($r.ProcessId)); log: $log"
