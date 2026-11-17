# Waits for the "everything finished" flag and then puts the PC to sleep.
#
# Hibernation is enabled on this machine, so the usual rundll32 powrprof trick
# would hibernate instead of sleeping. SetSuspendState(Suspend, ...) sleeps (S3)
# and needs no administrator rights.
#
#   -Probar     dry run: reports what it would do, never sleeps
#   -MaxHoras   give up (without sleeping) after this many hours; default 8
param(
    [switch]$Probar,
    [double]$MaxHoras = 8
)

$base   = "F:\Software Task generation Model"
$bandera = Join-Path $base "logs\LISTO.flag"
$limite = (Get-Date).AddHours($MaxHoras)

Write-Host "Esperando la senal: $bandera"
Write-Host "Se rinde a las $($limite.ToString('HH:mm')) si no aparece."

while (-not (Test-Path $bandera)) {
    if ((Get-Date) -gt $limite) {
        Write-Host "No aparecio la senal en $MaxHoras h. No duermo la PC."
        exit 1
    }
    Start-Sleep -Seconds 30
}

Write-Host "Senal encontrada:"
Get-Content $bandera | ForEach-Object { Write-Host "  $_" }

# safety net: never sleep while something of ours is still computing
for ($i = 0; $i -lt 20; $i++) {
    $py = @(Get-Process python -ErrorAction SilentlyContinue)
    if ($py.Count -eq 0) { break }
    Write-Host "  hay $($py.Count) proceso(s) python vivos; espero 30 s mas"
    Start-Sleep -Seconds 30
}

if ($Probar) {
    Write-Host "PRUEBA: aqui dormiria la PC (no lo hago por -Probar)."
    exit 0
}

Write-Host "Durmiendo la PC..."
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::SetSuspendState('Suspend', $false, $false) | Out-Null
