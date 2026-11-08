$ErrorActionPreference = 'Stop'

$labRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $labRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment not found: $python"
}

$env:HF_HOME = Join-Path $labRoot '.cache\huggingface'
$env:HF_HUB_DISABLE_XET = '1'
$env:TEMP = Join-Path $labRoot '.tmp'
$env:TMP = Join-Path $labRoot '.tmp'
$env:PIP_CACHE_DIR = Join-Path $labRoot '.pip-cache'
$env:TOKENIZERS_PARALLELISM = 'false'

New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:TEMP, $env:PIP_CACHE_DIR | Out-Null

$modelCache = Join-Path $env:HF_HOME 'hub\models--sentence-transformers--all-MiniLM-L6-v2'
$datasetCache = Join-Path $env:HF_HOME 'datasets\fancyzhx___ag_news'
if ((Test-Path -LiteralPath $modelCache) -and (Test-Path -LiteralPath $datasetCache)) {
    $env:HF_HUB_OFFLINE = '1'
    $env:HF_DATASETS_OFFLINE = '1'
}

& $python (Join-Path $labRoot 'src\train.py') @args
