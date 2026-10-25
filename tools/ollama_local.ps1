param([ValidateSet('Start', 'Stop')][string]$Action = 'Start')
$ErrorActionPreference = 'Stop'
$armRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$armExe = Join-Path $armRoot '.cache\ollama\ollama.exe'
$armPidFile = Join-Path $armRoot '.cache\ollama\server.pid'
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_MODELS = Join-Path $armRoot '.cache\ollama-models'
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_CONTEXT_LENGTH = '2048'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_MAX_LOADED_MODELS = '1'
$env:OLLAMA_VULKAN = 'false'
$env:CUDA_VISIBLE_DEVICES = '-1'

if ($Action -eq 'Stop') {
    if (-not (Test-Path -LiteralPath $armPidFile)) { throw 'No project-owned Ollama PID record.' }
    $armProcessId = [int](Get-Content -LiteralPath $armPidFile)
    $armProcess = Get-Process -Id $armProcessId -ErrorAction SilentlyContinue
    if ($null -eq $armProcess) { Write-Output 'Project Ollama already stopped.'; return }
    if ([IO.Path]::GetFullPath($armProcess.Path) -ne $armExe) { throw 'PID does not match the project Ollama executable.' }
    & $armExe stop qwen2.5:1.5b
    Stop-Process -Id $armProcessId
    Write-Output 'Stopped project Ollama.'
    return
}

if (-not (Test-Path -LiteralPath $armExe)) { throw 'Run python tools/prepare_ollama.py first.' }
try {
    $armVersion = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/version' -TimeoutSec 2
    Write-Output ('Ollama is already listening, version: ' + $armVersion.version)
    return
} catch { }
$armServer = Start-Process -FilePath $armExe -ArgumentList 'serve' -WorkingDirectory $armRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $armRoot '.cache\ollama\server.stdout.log') -RedirectStandardError (Join-Path $armRoot '.cache\ollama\server.stderr.log') -PassThru
$armServer.Id | Set-Content -LiteralPath $armPidFile
Write-Output ('Started project Ollama PID: ' + $armServer.Id)
