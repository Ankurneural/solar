<#
After a laptop restart: relaunch every job in runs/jobs that is still marked RUNNING
but has no live process. The jobs' scripts resume from their own checkpoints
(train.py: last.pt, infer.py: per-frame parts, predict_oof.py: existing maps,
train_refiner.py: finished folds are kept).

  powershell -NoProfile -File tools/resume.ps1           # list + relaunch
  powershell -NoProfile -File tools/resume.ps1 -DryRun   # list only
#>
param([switch]$DryRun)
$root = Split-Path -Parent $PSScriptRoot
$jobs = Join-Path $root "runs\jobs"
$live = (Get-CimInstance Win32_Process -Filter "Name='cmd.exe'").CommandLine -join "`n"
foreach ($st in Get-ChildItem $jobs -Filter *.status) {
    $name = $st.BaseName
    $state = (Get-Content $st.FullName -Raw).Trim()
    $bat = Join-Path $jobs "$name.cmd"
    if ($state -ne "RUNNING") { "{0,-28} {1}" -f $name, $state; continue }
    if ($live -match [regex]::Escape("$name.cmd")) { "{0,-28} RUNNING (alive)" -f $name; continue }
    if ($DryRun) { "{0,-28} RUNNING but dead -> would relaunch" -f $name; continue }
    # Older .cmd files hold absolute paths of the machine that started them: make them repo-relative.
    $txt = Get-Content $bat -Raw
    $fixed = $txt -replace 'cd /d "[^"]*"', 'cd /d "%~dp0..\.."' -replace '"[A-Za-z]:\\[^"]*\\runs\\jobs\\', '"runs\jobs\'
    if ($fixed -ne $txt) { Set-Content -Path $bat -Value $fixed -Encoding ascii -NoNewline }
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = "cmd.exe /c `"$bat`""; CurrentDirectory = $root }
    "{0,-28} RUNNING but dead -> relaunched (pid {1})" -f $name, $r.ProcessId
}
