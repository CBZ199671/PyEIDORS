[CmdletBinding()]
param(
    [ValidateSet('both', 'pyeidors_to_eidors', 'eidors_to_pyeidors')]
    [string]$Direction = 'both',

    [ValidateSet('coarse', 'medium', 'fine')]
    [string]$MeshLevel = 'medium',

    [ValidateSet('low_z', 'high_z')]
    [string]$Scenario = 'low_z',

    [double]$ElectrodeCoverage = 0.5,

    [switch]$IncludeSelfSource,

    [switch]$RenderPlots,

    [string]$OutDir = 'docs/benchmarks/interop',

    [string]$DockerContainer = 'pyeidors',

    [string]$PythonExe = '/opt/final_venv/bin/python',

    [string]$MatlabExe = 'D:\Program Files\MATLAB\R2023b\bin\matlab.exe'
)

$ErrorActionPreference = 'Stop'

function Get-RepoRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
}

function Get-RepoRelativePosixPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $relative = [System.IO.Path]::GetRelativePath($RepoRoot, $Path)
    return ($relative -replace '\\', '/')
}

function Invoke-WslRepoCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,
        [Parameter(Mandatory = $true)]
        [string]$Command
    )

    $drive = $RepoRoot.Substring(0, 1).ToLowerInvariant()
    $rest = $RepoRoot.Substring(2) -replace '\\', '/'
    $repoWsl = "/mnt/$drive$rest"
    return (& wsl.exe bash -lc "cd '$repoWsl' && $Command")
}

function Invoke-DockerPython {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command
    )

    & wsl.exe bash -lc "docker exec $DockerContainer bash -lc 'cd /root/shared && $Command'"
}

function Invoke-MatlabBatch {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BatchCommand
    )

    & $MatlabExe -batch $BatchCommand
}

$repoRoot = Get-RepoRoot
if (-not [System.IO.Path]::IsPathRooted($OutDir)) {
    $OutDir = Join-Path $repoRoot $OutDir
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$compareDir = (Join-Path $repoRoot 'compare_with_Eidors') -replace '\\', '/'
$commit = (Invoke-WslRepoCommand -RepoRoot $repoRoot -Command 'git rev-parse HEAD').Trim()

$resultPaths = @()

if ($Direction -in @('both', 'pyeidors_to_eidors')) {
    $prefix = "pyeidors_${MeshLevel}_${Scenario}"
    $meshMat = Join-Path $OutDir "${prefix}_geometry.mat"
    $forwardCsv = Join-Path $OutDir "${prefix}_forward.csv"
    $configJson = Join-Path $OutDir "${prefix}_import_config.json"
    $resultJson = Join-Path $OutDir "${prefix}_eidors_result.json"
    $detailsMat = Join-Path $OutDir "${prefix}_eidors_details.mat"
    $selfResultJson = Join-Path $OutDir "${prefix}_pyeidors_result.json"
    $selfDetailsMat = Join-Path $OutDir "${prefix}_pyeidors_details.mat"

    $meshMatRel = Get-RepoRelativePosixPath -RepoRoot $repoRoot -Path $meshMat
    $forwardCsvRel = Get-RepoRelativePosixPath -RepoRoot $repoRoot -Path $forwardCsv

    Invoke-DockerPython " $PythonExe scripts/interop/export_geometry_from_pyeidors.py --output-mat /root/shared/$meshMatRel --forward-export-csv /root/shared/$forwardCsvRel --mesh-level $MeshLevel --scenario $Scenario --electrode-coverage $ElectrodeCoverage"

    @{
        mesh_mat = ($meshMat -replace '\\', '/')
        input_csv = ($forwardCsv -replace '\\', '/')
        output_json = ($resultJson -replace '\\', '/')
        details_mat = ($detailsMat -replace '\\', '/')
        commit = $commit
    } | ConvertTo-Json | Set-Content -Path $configJson -Encoding UTF8

    $configJsonPosix = $configJson -replace '\\', '/'
    Invoke-MatlabBatch "cd('$compareDir'); import_geometry_from_pyeidors('$configJsonPosix');"
    $resultPaths += $resultJson
    if ($RenderPlots) {
        $detailsMatRel = Get-RepoRelativePosixPath -RepoRoot $repoRoot -Path $detailsMat
        $plotPrefixRel = Get-RepoRelativePosixPath -RepoRoot $repoRoot -Path (Join-Path $OutDir "${prefix}_eidors")
        Invoke-DockerPython " $PythonExe scripts/interop/render_bridge_report.py --geometry-mat /root/shared/$meshMatRel --details-mat /root/shared/$detailsMatRel --output-prefix /root/shared/$plotPrefixRel --title-prefix PyEIDORS_to_EIDORS"
    }

    if ($IncludeSelfSource) {
        $selfResultJsonRel = Get-RepoRelativePosixPath -RepoRoot $repoRoot -Path $selfResultJson
        $selfDetailsMatRel = Get-RepoRelativePosixPath -RepoRoot $repoRoot -Path $selfDetailsMat
        Invoke-DockerPython " $PythonExe scripts/interop/import_geometry_from_eidors.py --mesh-mat /root/shared/$meshMatRel --input-csv /root/shared/$forwardCsvRel --output-json /root/shared/$selfResultJsonRel --details-mat /root/shared/$selfDetailsMatRel"
        $resultPaths += $selfResultJson
        if ($RenderPlots) {
            $selfPlotPrefixRel = Get-RepoRelativePosixPath -RepoRoot $repoRoot -Path (Join-Path $OutDir "${prefix}_pyeidors")
            Invoke-DockerPython " $PythonExe scripts/interop/render_bridge_report.py --geometry-mat /root/shared/$meshMatRel --details-mat /root/shared/$selfDetailsMatRel --output-prefix /root/shared/$selfPlotPrefixRel --title-prefix PyEIDORS_to_PyEIDORS"
        }
    }
}

if ($Direction -in @('both', 'eidors_to_pyeidors')) {
    $prefix = "eidors_${MeshLevel}_${Scenario}"
    $meshMat = Join-Path $OutDir "${prefix}_geometry.mat"
    $forwardCsv = Join-Path $OutDir "${prefix}_forward.csv"
    $configJson = Join-Path $OutDir "${prefix}_export_config.json"
    $resultJson = Join-Path $OutDir "${prefix}_pyeidors_result.json"
    $detailsMat = Join-Path $OutDir "${prefix}_pyeidors_details.mat"
    $selfConfigJson = Join-Path $OutDir "${prefix}_self_import_config.json"
    $selfResultJson = Join-Path $OutDir "${prefix}_eidors_result.json"
    $selfDetailsMat = Join-Path $OutDir "${prefix}_eidors_details.mat"

    @{
        output_mat = ($meshMat -replace '\\', '/')
        forward_export_csv = ($forwardCsv -replace '\\', '/')
        mesh_level = $MeshLevel
        scenario = $Scenario
        n_elec = 16
    } | ConvertTo-Json | Set-Content -Path $configJson -Encoding UTF8

    $configJsonPosix = $configJson -replace '\\', '/'
    Invoke-MatlabBatch "cd('$compareDir'); export_geometry_from_eidors('$configJsonPosix');"

    $meshMatRel = Get-RepoRelativePosixPath -RepoRoot $repoRoot -Path $meshMat
    $forwardCsvRel = Get-RepoRelativePosixPath -RepoRoot $repoRoot -Path $forwardCsv
    $resultJsonRel = Get-RepoRelativePosixPath -RepoRoot $repoRoot -Path $resultJson
    $detailsMatRel = Get-RepoRelativePosixPath -RepoRoot $repoRoot -Path $detailsMat
    Invoke-DockerPython " $PythonExe scripts/interop/import_geometry_from_eidors.py --mesh-mat /root/shared/$meshMatRel --input-csv /root/shared/$forwardCsvRel --output-json /root/shared/$resultJsonRel --details-mat /root/shared/$detailsMatRel"
    $resultPaths += $resultJson
    if ($RenderPlots) {
        $plotPrefixRel = Get-RepoRelativePosixPath -RepoRoot $repoRoot -Path (Join-Path $OutDir "${prefix}_pyeidors")
        Invoke-DockerPython " $PythonExe scripts/interop/render_bridge_report.py --geometry-mat /root/shared/$meshMatRel --details-mat /root/shared/$detailsMatRel --output-prefix /root/shared/$plotPrefixRel --title-prefix EIDORS_to_PyEIDORS"
    }

    if ($IncludeSelfSource) {
        @{
            mesh_mat = ($meshMat -replace '\\', '/')
            input_csv = ($forwardCsv -replace '\\', '/')
            output_json = ($selfResultJson -replace '\\', '/')
            details_mat = ($selfDetailsMat -replace '\\', '/')
            commit = $commit
        } | ConvertTo-Json | Set-Content -Path $selfConfigJson -Encoding UTF8
        $selfConfigJsonPosix = $selfConfigJson -replace '\\', '/'
        Invoke-MatlabBatch "cd('$compareDir'); import_geometry_from_pyeidors('$selfConfigJsonPosix');"
        $resultPaths += $selfResultJson
        if ($RenderPlots) {
            $selfDetailsMatRel = Get-RepoRelativePosixPath -RepoRoot $repoRoot -Path $selfDetailsMat
            $selfPlotPrefixRel = Get-RepoRelativePosixPath -RepoRoot $repoRoot -Path (Join-Path $OutDir "${prefix}_eidors")
            Invoke-DockerPython " $PythonExe scripts/interop/render_bridge_report.py --geometry-mat /root/shared/$meshMatRel --details-mat /root/shared/$selfDetailsMatRel --output-prefix /root/shared/$selfPlotPrefixRel --title-prefix EIDORS_to_EIDORS"
        }
    }
}

if ($resultPaths.Count -gt 0) {
    $summaryCsv = Join-Path $OutDir "same_geometry_interop_summary_${MeshLevel}_${Scenario}.csv"
    $rows = foreach ($resultPath in $resultPaths) {
        $row = Get-Content $resultPath | ConvertFrom-Json
        [pscustomobject]@{
            source_framework = $row.source_framework
            framework = $row.framework
            exchange_format = $row.exchange_format
            study = $row.study
            imported_same_geometry = if ($null -eq $row.imported_same_geometry) { $false } else { [bool]$row.imported_same_geometry }
            mesh_name = $row.mesh_name
            n_nodes = $row.n_nodes
            n_elements = $row.n_elements
            voltage_rmse = $row.voltage_rmse
            conductivity_rmse = $row.conductivity_rmse
        }
    }
    $rows | Export-Csv -NoTypeInformation -Encoding UTF8 $summaryCsv
    Write-Host "Wrote $summaryCsv"
}
