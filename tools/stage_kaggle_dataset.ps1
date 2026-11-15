<#
Rebuild the Kaggle dataset folder (ankurparmar/solarfragments-weights) for the reproduction notebook.

  powershell -NoProfile -ExecutionPolicy Bypass -File tools/stage_kaggle_dataset.ps1            # -> runs/kaggle_dataset
  python tools/kaggle_cli.py datasets version -p runs/kaggle_dataset -m "<msg>" -r tar -t -q     # upload a new version

Contents: inference code, both final run dirs (ensemble/postprocess/scorer/stack), best.pt + config of every member,
the 5 rotated-crop refiners (once, in refiners/), and Linux wheels of requirements-infer.txt for cp310-cp313
(the Kaggle image used Python 3.13 on 11-09). Checkpoints are hardlinked, so the folder costs no extra disk space.
#>
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$D = "runs\kaggle_dataset"
$finals = "w11_all_768b", "w_s2_44h"
New-Item -ItemType Directory -Force $D | Out-Null

function Link($src, $dst) {
    New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
    if (-not (Test-Path $dst)) { New-Item -ItemType HardLink -Path $dst -Target (Resolve-Path $src) | Out-Null }
}

$members = foreach ($r in $finals) {
    Select-String -Path "runs\$r\ensemble.yaml" -Pattern '^- run: (.+)$' | ForEach-Object { $_.Matches[0].Groups[1].Value.Trim() }
}
foreach ($m in ($members | Sort-Object -Unique)) {
    foreach ($k in 0..4) {
        Link "$m\fold$k\best.pt" "$D\$m\fold$k\best.pt"
        Copy-Item "$m\fold$k\config_used.yaml" "$D\$m\fold$k\" -Force
    }
}
foreach ($r in $finals) {
    New-Item -ItemType Directory -Force "$D\runs\$r" | Out-Null
    foreach ($f in "ensemble.yaml", "postprocess_tuned.yaml", "scorer.joblib", "stack.yaml") { Copy-Item "runs\$r\$f" "$D\runs\$r\" -Force }
}
foreach ($k in 0..4) { Link "runs\w11_all_768b\refiner\fold$k.pt" "$D\refiners\fold$k.pt" }

New-Item -ItemType Directory -Force "$D\src", "$D\tools" | Out-Null
Copy-Item src\*.py "$D\src\" -Force
Copy-Item infer.py, prepare_data.py, evaluate.py, requirements.txt, requirements-infer.txt, LICENSE "$D\" -Force
Copy-Item tools\submit.py "$D\tools\" -Force
Copy-Item tools\kaggle_dataset-metadata.json "$D\dataset-metadata.json" -Force

New-Item -ItemType Directory -Force "$D\wheels" | Out-Null
foreach ($py in "310", "311", "312", "313") {
    & .venv\Scripts\python.exe -m pip download -q -r requirements-infer.txt --no-deps --only-binary=:all: --implementation cp `
        --python-version $py --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 -d "$D\wheels"
}
Get-ChildItem $D -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
"staged {0} files, {1:N1} GB -> {2}" -f (Get-ChildItem $D -Recurse -File).Count, ((Get-ChildItem $D -Recurse -File | Measure-Object Length -Sum).Sum / 1GB), $D
