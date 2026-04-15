[CmdletBinding()]
param(
    [ValidateSet('cpu', 'gpu')]
    [string]$Profile = 'cpu',

    [switch]$SkipCudaProbe,

    [switch]$DryRun,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AppArgs
)

$ErrorActionPreference = 'Stop'

function Get-RepoRoot {
    return (Get-Item (Join-Path $PSScriptRoot '..\..')).FullName
}

function Resolve-WslRepoLocation {
    param(
        [Parameter(Mandatory = $true)]
        [string]$WindowsPath
    )

    $normalized = $WindowsPath.Trim()
    $uncPattern = '^(?:\\\\wsl(?:\.localhost|\$)\\)(?<distro>[^\\]+)(?<suffix>\\.*)$'
    if ($normalized -match $uncPattern) {
        $distro = $Matches['distro']
        $suffix = $Matches['suffix'] -replace '\\', '/'
        if (-not $suffix.StartsWith('/')) {
            $suffix = "/$suffix"
        }
        return [pscustomobject]@{
            Distro = $distro
            LinuxPath = $suffix
        }
    }

    $linuxPath = (& wsl.exe wslpath -a -u $normalized).Trim()
    if (-not $linuxPath) {
        throw "无法将 Windows 路径转换为 WSL 路径: $WindowsPath"
    }
    return [pscustomobject]@{
        Distro = $null
        LinuxPath = $linuxPath
    }
}

function Invoke-WslGuiLauncher {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRootWindows,

        [Parameter(Mandatory = $true)]
        [ValidateSet('cpu', 'gpu')]
        [string]$LaunchProfile,

        [switch]$PassSkipCudaProbe,

        [switch]$PassDryRun,

        [string[]]$ForwardArgs = @()
    )

    $resolved = Resolve-WslRepoLocation -WindowsPath $RepoRootWindows
    $wslArgs = @()
    if ($resolved.Distro) {
        $wslArgs += @('-d', $resolved.Distro)
    }
    $wslArgs += @('--cd', $resolved.LinuxPath, 'bash', 'scripts/gui/run_eit_app.sh', "--$LaunchProfile")
    if ($PassSkipCudaProbe) {
        $wslArgs += '--skip-cuda-probe'
    }
    if ($PassDryRun) {
        Write-Host "[eit-gui] WSL command:" -ForegroundColor Cyan
        Write-Host ("wsl.exe " + ($wslArgs -join ' ')) -ForegroundColor DarkCyan
        return
    }
    if ($ForwardArgs.Count -gt 0) {
        $wslArgs += '--'
        $wslArgs += $ForwardArgs
    }
    & wsl.exe @wslArgs
    if ($LASTEXITCODE -ne 0) {
        throw "GUI 启动失败，WSL 退出码: $LASTEXITCODE"
    }
}

$repoRoot = Get-RepoRoot
Invoke-WslGuiLauncher `
    -RepoRootWindows $repoRoot `
    -LaunchProfile $Profile `
    -PassSkipCudaProbe:$SkipCudaProbe `
    -PassDryRun:$DryRun `
    -ForwardArgs $AppArgs
