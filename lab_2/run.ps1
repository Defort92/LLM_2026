$ErrorActionPreference = 'Stop'

$labRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ownPython = Join-Path $labRoot '.venv\Scripts\python.exe'
$sharedPython = Join-Path (Split-Path -Parent $labRoot) 'lab_1\.venv\Scripts\python.exe'

if (Test-Path -LiteralPath $ownPython) {
    $python = $ownPython
}
elseif (Test-Path -LiteralPath $sharedPython) {
    $python = $sharedPython
}
else {
    throw "Virtual environment not found. Create lab_2\.venv or keep lab_1\.venv."
}

$env:HF_HOME = Join-Path $labRoot '.cache\huggingface'
$env:HF_HUB_DISABLE_XET = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:NLTK_DATA = Join-Path $labRoot '.cache\nltk_data'
$env:TEMP = Join-Path $labRoot '.tmp'
$env:TMP = Join-Path $labRoot '.tmp'
$env:PIP_CACHE_DIR = Join-Path $labRoot '.pip-cache'
$env:TOKENIZERS_PARALLELISM = 'false'

New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:NLTK_DATA, $env:TEMP, $env:PIP_CACHE_DIR | Out-Null

$decoderCache = Join-Path $env:HF_HOME 'hub\models--distilbert--distilgpt2'
$seq2seqCache = Join-Path $env:HF_HOME 'hub\models--google--flan-t5-small'
$datasetCache = Join-Path $env:HF_HOME 'datasets\OpenRL___daily_dialog'
if ((Test-Path -LiteralPath $decoderCache) -and (Test-Path -LiteralPath $seq2seqCache) -and (Test-Path -LiteralPath $datasetCache)) {
    $env:HF_HUB_OFFLINE = '1'
    $env:HF_DATASETS_OFFLINE = '1'
}

& $python (Join-Path $labRoot 'src\train.py') @args
