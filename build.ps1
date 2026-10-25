# 编译 BGStateExporter 并安装到 HDT 插件目录
# 用法：powershell -ExecutionPolicy Bypass -File build.ps1 [-HDTPath "HDT所在目录"]
# 不传 -HDTPath 时依次查找：与本项目同级的 ..\HDT、%LocalAppData%\HearthstoneDeckTracker\app-最新版本
# 需要：.NET SDK 6 或更高版本（https://dotnet.microsoft.com/download）

param([string]$HDTPath = "")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Test-HdtDir([string]$dir) {
    return $dir -and (Test-Path (Join-Path $dir "HearthstoneDeckTracker.exe")) -and (Test-Path (Join-Path $dir "BobsBuddy.dll"))
}

# 1. 找到 HDT 目录（里面要有 HearthstoneDeckTracker.exe 和 BobsBuddy.dll）
$candidates = @()
if ($HDTPath) { $candidates += $HDTPath }
$candidates += (Join-Path $PSScriptRoot "..\HDT")
$installed = Get-ChildItem (Join-Path $env:LOCALAPPDATA "HearthstoneDeckTracker") -Directory -Filter "app-*" -ErrorAction SilentlyContinue |
    Sort-Object { [version]($_.Name -replace '^app-', '') } -Descending |
    Select-Object -First 1
if ($installed) { $candidates += $installed.FullName }

$hdt = $null
foreach ($c in $candidates) {
    if (Test-HdtDir $c) { $hdt = (Resolve-Path $c).Path; break }
}
if (-not $hdt) {
    Write-Host "没找到 HDT 目录。试过：" -ForegroundColor Red
    $candidates | ForEach-Object { Write-Host "  $_" }
    Write-Host "请这样指定：powershell -ExecutionPolicy Bypass -File build.ps1 -HDTPath D:\你的\HDT"
    exit 1
}
Write-Host "HDT 目录：$hdt"

# 2. 编译
dotnet build -c Release "-p:HDTPath=$hdt"
if ($LASTEXITCODE -ne 0) { Write-Host "编译失败" -ForegroundColor Red; exit 1 }

# 3. 安装到 %AppData%\HearthstoneDeckTracker\Plugins
#    注意：HDT 目录下的 Plugins 只是镜像，HDT 启动时会用 AppData 里的内容覆盖它，装在那里会被删掉
$pluginDir = Join-Path $env:APPDATA "HearthstoneDeckTracker\Plugins\BGStateExporter"
New-Item -ItemType Directory -Force -Path $pluginDir | Out-Null
Copy-Item "bin\Release\BGStateExporter.dll" $pluginDir -Force
Write-Host "已安装到 $pluginDir" -ForegroundColor Green
Write-Host "HDT 正在运行的话需要重启。然后：选项 -> 跟踪器 -> 插件 -> 勾选 BG State Exporter -> 点「打开网页面板」"
