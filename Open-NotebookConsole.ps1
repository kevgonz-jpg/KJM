param(
    [string]$RuntimeDir,
    [switch]$DryRun
)

$python = "c:/python313/python.exe"

if (-not $RuntimeDir) {
    $RuntimeDir = & $python -m jupyter --runtime-dir
}

if (-not (Test-Path $RuntimeDir)) {
    throw "No existe el directorio runtime de Jupyter: $RuntimeDir"
}

$kernelFiles = Get-ChildItem -Path $RuntimeDir -Filter 'kernel-*.json' | Sort-Object LastWriteTime -Descending

if (-not $kernelFiles) {
    throw "No se encontro ningun archivo kernel-*.json en $RuntimeDir. Abre primero el notebook para crear el kernel."
}

function Get-KernelName {
    param([string]$Path)

    try {
        return (Get-Content -Path $Path -Raw | ConvertFrom-Json).kernel_name
    }
    catch {
        return $null
    }
}

$vscodeKernel = $kernelFiles | Where-Object {
    (Get-KernelName $_.FullName) -like 'python3112jvsc*'
} | Select-Object -First 1

$kernelFile = if ($vscodeKernel) {
    $vscodeKernel.FullName
} else {
    $kernelFiles[0].FullName
}

if ($DryRun) {
    Write-Host "Kernel detectado: $kernelFile"
    exit 0
}

& $python -m jupyter_console --existing $kernelFile