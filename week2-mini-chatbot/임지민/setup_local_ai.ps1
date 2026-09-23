# Downloads public local models; never calls a paid API.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
New-Item -ItemType Directory -Force '.local-ai' | Out-Null
if (-not (Test-Path '.local-ai/ollama/ollama.exe')) {
    Invoke-WebRequest 'https://github.com/ollama/ollama/releases/download/v0.34.3/ollama-windows-amd64.zip' -OutFile '.local-ai/ollama.zip'
    Expand-Archive -LiteralPath '.local-ai/ollama.zip' -DestinationPath '.local-ai/ollama' -Force
}
$env:OLLAMA_MODELS = Join-Path $PSScriptRoot '.local-ai/models'
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_MAX_LOADED_MODELS = '1'
try { Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' | Out-Null }
catch {
    Start-Process -FilePath (Join-Path $PSScriptRoot '.local-ai/ollama/ollama.exe') -ArgumentList 'serve' -WindowStyle Hidden
    $serverReady = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Seconds 1
        try {
            Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2 | Out-Null
            $serverReady = $true
            break
        } catch { }
    }
    if (-not $serverReady) { throw 'Local server did not start. Check port 11434 and run_local_ai.bat.' }
}
& '.local-ai/ollama/ollama.exe' pull qwen3.5:2b
if ($LASTEXITCODE -ne 0) { throw 'Chat model download failed' }
& '.local-ai/ollama/ollama.exe' pull embeddinggemma
if ($LASTEXITCODE -ne 0) { throw 'Embedding model download failed' }
Write-Output 'Local models ready. Run run_chatbot.bat.'
