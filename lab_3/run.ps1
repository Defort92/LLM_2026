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
    throw "Virtual environment not found. Create lab_3\.venv or keep lab_1\.venv."
}

$env:HF_HOME = Join-Path $labRoot '.cache\huggingface'
$env:HF_HUB_DISABLE_XET = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TEMP = Join-Path $labRoot '.tmp'
$env:TMP = Join-Path $labRoot '.tmp'
$env:PIP_CACHE_DIR = Join-Path $labRoot '.pip-cache'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:PYTHONIOENCODING = 'utf-8'

New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:TEMP, $env:PIP_CACHE_DIR | Out-Null

$modelCache = Join-Path $env:HF_HOME 'hub\models--intfloat--multilingual-e5-small'
if (Test-Path -LiteralPath $modelCache) {
    $env:HF_HUB_OFFLINE = '1'
}

& $python (Join-Path $labRoot 'src\train.py') @args

