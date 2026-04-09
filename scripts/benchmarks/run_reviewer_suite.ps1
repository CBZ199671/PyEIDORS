param(
    [ValidateSet("full", "smoke")]
    [string]$Mode = "full",
    [ValidateSet("all", "core", "heavy", "fairness", "aggregate")]
    [string]$Phase = "all",
    [int]$Warmups = 3,
    [int]$Repeats = 10,
    [bool]$SkipExisting = $true,
    [switch]$Force,
    [bool]$ContinueOnCaseError = $true
)

$ErrorActionPreference = "Stop"

if ($Force) {
    $SkipExisting = $false
}

$repoRoot = "D:\workspace\PyEIDORS"
$docsRoot = Join-Path $repoRoot "docs\benchmarks\reviewer_suite"
$rawRoot = Join-Path $docsRoot "raw"
$fairnessRoot = Join-Path $docsRoot "fairness"
$crossRoot = Join-Path $fairnessRoot "raw_cross"
$aggregateRoot = Join-Path $docsRoot "aggregated"
$stateRoot = Join-Path $docsRoot "state"
$staleRoot = Join-Path $stateRoot "stale"
New-Item -ItemType Directory -Force -Path $rawRoot | Out-Null
New-Item -ItemType Directory -Force -Path $crossRoot | Out-Null
New-Item -ItemType Directory -Force -Path $aggregateRoot | Out-Null
New-Item -ItemType Directory -Force -Path $stateRoot | Out-Null
New-Item -ItemType Directory -Force -Path $staleRoot | Out-Null

$matlabExe = "D:\Program Files\MATLAB\R2023b\bin\matlab.exe"
$matlabScriptDir = "D:/workspace/PyEIDORS/compare_with_Eidors"
$pythonCase = "scripts/benchmarks/benchmark_reviewer_case.py"
$pythonCrossCase = "scripts/benchmarks/cross_generation_case.py"
$pythonMeshControl = "scripts/benchmarks/mesh_matched_control.py"
$pythonAggregate = "scripts/benchmarks/aggregate_reviewer_suite.py"
$llmDemo = "scripts/reviewer_demos/run_llm_agent_case.py"
$commit = (git -C $repoRoot rev-parse HEAD).Trim()
$runId = "{0}-{1}" -f (Get-Date -Format "yyyyMMdd-HHmmss"), $PID
$manifestPath = Join-Path $stateRoot "run_manifest.json"
$currentCasePath = Join-Path $stateRoot "current_case.json"
$runStatusPath = Join-Path $stateRoot "run_status.jsonl"
$heavyCommandsPath = Join-Path $stateRoot "heavy_commands.txt"
$containerStateRel = "docs/benchmarks/reviewer_suite/state/current_case.json"
$containerRawRoot = "docs/benchmarks/reviewer_suite/raw"

function Write-Utf8NoBom {
    param(
        [string]$Path,
        [string]$Content
    )
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $utf8NoBom)
}

function Append-Utf8NoBom {
    param(
        [string]$Path,
        [string]$Content
    )
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::AppendAllText($Path, $Content, $utf8NoBom)
}

function Write-JsonFile {
    param(
        [string]$Path,
        [object]$Data
    )
    $json = $Data | ConvertTo-Json -Depth 8
    Write-Utf8NoBom -Path $Path -Content $json
}

function Get-NowIso {
    return (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
}

function Invoke-ExternalCommand {
    param(
        [string]$Exe,
        [string[]]$Arguments
    )
    & $Exe @Arguments 2>&1 | ForEach-Object {
        if ($null -ne $_) {
            $_ | Out-Host
        }
    }
    if ($null -eq $LASTEXITCODE) {
        return 0
    }
    return [int]$LASTEXITCODE
}

function ConvertTo-IntegerExitCode {
    param([object]$Value)

    if ($null -eq $Value) {
        return 0
    }
    if ($Value -is [int]) {
        return [int]$Value
    }
    if ($Value -is [long] -or $Value -is [short] -or $Value -is [byte]) {
        return [int]$Value
    }
    if ($Value -is [double] -or $Value -is [float] -or $Value -is [decimal]) {
        $numericValue = [double]$Value
        if ($numericValue -eq [Math]::Truncate($numericValue)) {
            return [int]$numericValue
        }
    }
    if ($Value -is [string]) {
        $parsed = 0
        if ([int]::TryParse($Value.Trim(), [ref]$parsed)) {
            return $parsed
        }
    }
    $typeName = $Value.GetType().FullName
    throw "Internal bookkeeping error: non-integer exit code returned ($typeName)"
}

function Get-ExpectedDevice {
    param([pscustomobject]$Case)
    if ($Case.Kind -ne "benchmark") {
        if ($Case.Device) {
            return $Case.Device
        }
        return "na"
    }
    if ($Case.Runner -eq "matlab") {
        return "cpu"
    }
    if ($Case.Framework -eq "pyeidors" -and @("jacobian", "absolute_gn") -contains $Case.Task) {
        return $Case.Device
    }
    return "cpu"
}

function New-CaseSpec {
    param(
        [string]$Kind,
        [string]$Runner,
        [string]$Framework = "",
        [string]$Task = "",
        [string]$MeshLevel = "",
        [string]$Scenario = "low_z",
        [string]$Device = "cpu",
        [int]$NFrames = 1,
        [string]$PhaseName = "core",
        [string]$OutputRel = "",
        [string[]]$OutputRels = @(),
        [string]$Label = ""
    )

    $allOutputRels = @()
    if ($OutputRel) {
        $allOutputRels += $OutputRel
    }
    if ($OutputRels) {
        $allOutputRels += $OutputRels
    }
    $outputPaths = @($allOutputRels | ForEach-Object { Join-Path $repoRoot ($_ -replace '/', '\') })
    $frameworkKey = if ($Framework) { $Framework } elseif ($Runner -eq "matlab") { "eidors" } else { $Label }
    $deviceKey = Get-ExpectedDevice ([pscustomobject]@{
        Kind = $Kind
        Runner = $Runner
        Framework = $Framework
        Task = $Task
        Device = $Device
    })
    $caseIdParts = @($frameworkKey, $Task, $MeshLevel, $Scenario, $deviceKey, "f$NFrames", "w$Warmups", "r$Repeats")
    if ($Label) {
        $caseIdParts = @($Label) + $caseIdParts
    }
    $caseId = ($caseIdParts | Where-Object { $_ -and $_ -ne "" }) -join "-"
    [pscustomobject]@{
        Kind = $Kind
        Runner = $Runner
        Framework = $frameworkKey
        FrameworkArg = $Framework
        Task = $Task
        MeshLevel = $MeshLevel
        Scenario = $Scenario
        Device = $deviceKey
        DeviceArg = $Device
        NFrames = $NFrames
        PhaseName = $PhaseName
        OutputRel = $OutputRel
        OutputRels = $allOutputRels
        OutputPaths = $outputPaths
        Label = $Label
        CaseId = $caseId
        Commit = $commit
    }
}

function New-BenchmarkCase {
    param(
        [string]$Framework,
        [string]$Task,
        [string]$MeshLevel,
        [string]$Scenario = "low_z",
        [string]$Device = "cpu",
        [int]$NFrames = 1,
        [string]$Runner = "docker"
    )
    $frameworkKey = if ($Runner -eq "matlab") { "eidors" } else { $Framework }
    $deviceKey = if ($Runner -eq "matlab") { "cpu" } else { $Device }
    $name = "${frameworkKey}_${Task}_${MeshLevel}_${Scenario}_${deviceKey}_f${NFrames}.json"
    $phaseName = if ($Task -eq "absolute_gn" -and $MeshLevel -eq "fine") { "heavy" } else { "core" }
    $case = New-CaseSpec `
        -Kind "benchmark" `
        -Runner $Runner `
        -Framework $Framework `
        -Task $Task `
        -MeshLevel $MeshLevel `
        -Scenario $Scenario `
        -Device $deviceKey `
        -NFrames $NFrames `
        -PhaseName $phaseName `
        -OutputRel ("docs/benchmarks/reviewer_suite/raw/{0}" -f $name)
    if ($Runner -eq "matlab") {
        $cfgPath = Join-Path $rawRoot ($name -replace '\.json$', '_config.json')
        $case | Add-Member -NotePropertyName ConfigPath -NotePropertyValue $cfgPath
    }
    return $case
}

function Set-CurrentCaseState {
    param(
        [string]$Status,
        [pscustomobject]$Case = $null,
        [hashtable]$Extra = @{}
    )
    $payload = [ordered]@{
        run_id = $runId
        status = $Status
        updated_at = Get-NowIso
        phase = if ($Case) { $Case.PhaseName } else { $Phase }
    }
    if ($Case) {
        $payload.case_id = $Case.CaseId
        $payload.kind = $Case.Kind
        $payload.runner = $Case.Runner
        $payload.framework = $Case.Framework
        $payload.task = $Case.Task
        $payload.mesh_level = $Case.MeshLevel
        $payload.scenario = $Case.Scenario
        $payload.device = $Case.Device
        $payload.n_frames = $Case.NFrames
        $payload.output_json = if ($Case.OutputRel) { $Case.OutputRel } else { "" }
        $payload.label = $Case.Label
    }
    foreach ($key in $Extra.Keys) {
        $payload[$key] = $Extra[$key]
    }
    Write-JsonFile -Path $currentCasePath -Data $payload
}

function Append-RunStatusEvent {
    param(
        [pscustomobject]$Case,
        [string]$Status,
        [string]$Message = ""
    )
    $event = [ordered]@{
        run_id = $runId
        case_id = if ($Case) { $Case.CaseId } else { "" }
        framework = if ($Case) { $Case.Framework } else { "" }
        task = if ($Case) { $Case.Task } else { "" }
        mesh_level = if ($Case) { $Case.MeshLevel } else { "" }
        scenario = if ($Case) { $Case.Scenario } else { "" }
        device = if ($Case) { $Case.Device } else { "" }
        n_frames = if ($Case) { [int]$Case.NFrames } else { 0 }
        phase = if ($Case) { $Case.PhaseName } else { $Phase }
        kind = if ($Case) { $Case.Kind } else { "" }
        runner = if ($Case) { $Case.Runner } else { "" }
        output_json = if ($Case -and $Case.OutputRel) { $Case.OutputRel } else { "" }
        status = $Status
        message = $Message
        updated_at = Get-NowIso
    }
    Append-Utf8NoBom -Path $runStatusPath -Content (($event | ConvertTo-Json -Compress) + [Environment]::NewLine)
}

function Move-StaleResult {
    param([pscustomobject]$Case)
    foreach ($path in $Case.OutputPaths) {
        if (-not (Test-Path $path)) {
            continue
        }
        $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $dest = Join-Path $staleRoot ("{0}_{1}{2}" -f [System.IO.Path]::GetFileNameWithoutExtension($path), $timestamp, [System.IO.Path]::GetExtension($path))
        Move-Item -Path $path -Destination $dest -Force
    }
}

function Test-BenchmarkResult {
    param([pscustomobject]$Case)
    $path = $Case.OutputPaths[0]
    if (-not (Test-Path $path)) {
        return [pscustomobject]@{ Status = "missing"; Reason = "file_missing" }
    }
    try {
        $payload = Get-Content -Path $path -Raw | ConvertFrom-Json
    } catch {
        return [pscustomobject]@{ Status = "stale"; Reason = "invalid_json" }
    }

    $required = @("framework", "task", "mesh_level", "scenario", "device", "warmups", "repeats", "commit", "mean", "median", "iqr")
    foreach ($field in $required) {
        if (-not ($payload.PSObject.Properties.Name -contains $field)) {
            return [pscustomobject]@{ Status = "stale"; Reason = "missing_$field" }
        }
    }

    $expectedFrames = if ($Case.Task -eq "multi_frame_difference") { [int]$Case.NFrames } else { 1 }
    if (
        [string]$payload.framework -ne [string]$Case.Framework -or
        [string]$payload.task -ne [string]$Case.Task -or
        [string]$payload.mesh_level -ne [string]$Case.MeshLevel -or
        [string]$payload.scenario -ne [string]$Case.Scenario -or
        [string]$payload.device -ne [string](Get-ExpectedDevice $Case) -or
        [int]$payload.warmups -ne [int]$Warmups -or
        [int]$payload.repeats -ne [int]$Repeats -or
        [int]$payload.n_frames -ne [int]$expectedFrames -or
        [string]$payload.commit -ne [string]$commit
    ) {
        return [pscustomobject]@{ Status = "stale"; Reason = "identity_mismatch" }
    }
    return [pscustomobject]@{ Status = "valid"; Reason = "" }
}

function Test-MetaOutputs {
    param([pscustomobject]$Case)
    if (-not $Case.OutputPaths -or $Case.OutputPaths.Count -eq 0) {
        return $false
    }
    foreach ($path in $Case.OutputPaths) {
        if (-not (Test-Path $path)) {
            return $false
        }
    }
    return $true
}

function Should-SkipCase {
    param([pscustomobject]$Case)
    if (-not $SkipExisting) {
        return [pscustomobject]@{ Skip = $false; Status = "missing"; Reason = "skip_disabled" }
    }
    if ($Case.Kind -eq "benchmark") {
        $result = Test-BenchmarkResult -Case $Case
        if ($result.Status -eq "valid") {
            return [pscustomobject]@{ Skip = $true; Status = "valid"; Reason = "" }
        }
        if ($result.Status -eq "stale") {
            Move-StaleResult -Case $Case
        }
        return [pscustomobject]@{ Skip = $false; Status = $result.Status; Reason = $result.Reason }
    }
    if ($Case.Kind -eq "aggregate") {
        return [pscustomobject]@{ Skip = $false; Status = "missing"; Reason = "always_run" }
    }
    if (Test-MetaOutputs -Case $Case) {
        return [pscustomobject]@{ Skip = $true; Status = "valid"; Reason = "" }
    }
    return [pscustomobject]@{ Skip = $false; Status = "missing"; Reason = "file_missing" }
}

function Invoke-DockerBenchmarkCase {
    param([pscustomobject]$Case)
    $commandText = @(
        "cd /root/shared",
        "python $pythonCase --framework $($Case.FrameworkArg) --task $($Case.Task) --mesh-level $($Case.MeshLevel) --scenario $($Case.Scenario) --device $($Case.DeviceArg) --warmups $Warmups --repeats $Repeats --n-frames $($Case.NFrames) --output-json $($Case.OutputRel.Replace('\', '/')) --run-id $runId --case-id $($Case.CaseId) --phase $($Case.PhaseName) --state-json $containerStateRel"
    ) -join " && "
    Write-Host "[RUN] docker benchmark $($Case.CaseId)"
    return Invoke-ExternalCommand -Exe "docker" -Arguments @("exec", "pyeidors", "bash", "-lc", $commandText)
}

function Invoke-MatlabBenchmarkCase {
    param([pscustomobject]$Case)
    $cfg = [ordered]@{
        mesh_level = $Case.MeshLevel
        scenario = $Case.Scenario
        task = $Case.Task
        warmups = $Warmups
        repeats = $Repeats
        n_frames = $Case.NFrames
        n_elec = 16
        absolute_max_iter = 15
        commit = $commit
        output_json = $Case.OutputPaths[0]
        run_id = $runId
        case_id = $Case.CaseId
    }
    Write-JsonFile -Path $Case.ConfigPath -Data $cfg
    Set-CurrentCaseState -Status "running" -Case $Case
    Write-Host "[RUN] MATLAB benchmark $($Case.CaseId)"
    $batch = "cd('$matlabScriptDir'); benchmark_reviewer_case('$($Case.ConfigPath -replace '\\','/')');"
    return Invoke-ExternalCommand -Exe $matlabExe -Arguments @("-batch", $batch)
}

function Invoke-LlmDemoCase {
    Write-Host "[RUN] LLM demo"
    return Invoke-ExternalCommand -Exe "docker" -Arguments @("exec", "pyeidors", "bash", "-lc", "cd /root/shared && python $llmDemo")
}

function Invoke-CrossGenerationStudies {
    $pyeidorsForward = "results/simulation_parity/softwarex_cross_pyeidors_source/synthetic_forward_data.csv"
    $pyeidorsMetrics = "results/simulation_parity/softwarex_cross_pyeidors_source/metrics.json"
    $cmdPyForward = "cd /root/shared && python scripts/run_synthetic_parity.py --output-root results/simulation_parity/softwarex_cross_pyeidors_source --mode both --save-forward-csv --difference-solver single-step --step-size-calibration --gn-max-iterations 15 --background 1.0 --contact-impedance 1e-6 --phantom-center 0.3 0.2 --phantom-radius 0.2 --phantom-contrast 2.0"
    $exitCode = Invoke-ExternalCommand -Exe "docker" -Arguments @("exec", "pyeidors", "bash", "-lc", $cmdPyForward)
    if ($exitCode -ne 0) { return $exitCode }

    $eidorsSelfConfig = Join-Path $crossRoot "eidors_source_config.json"
    $eidorsSelfOutput = Join-Path $crossRoot "eidors_from_eidors.json"
    $eidorsForwardCsv = Join-Path $fairnessRoot "eidors_source_forward.csv"
    $eidorsSelfCfg = [ordered]@{
        mesh_level = "medium"
        scenario = "low_z"
        source_framework = "eidors"
        n_elec = 16
        commit = $commit
        output_json = $eidorsSelfOutput
        forward_export_csv = $eidorsForwardCsv
    }
    Write-JsonFile -Path $eidorsSelfConfig -Data $eidorsSelfCfg
    $exitCode = Invoke-ExternalCommand -Exe $matlabExe -Arguments @("-batch", "cd('$matlabScriptDir'); cross_generation_case('$($eidorsSelfConfig -replace '\\','/')');")
    if ($exitCode -ne 0) { return $exitCode }

    $eidorsOnPyConfig = Join-Path $crossRoot "eidors_on_pyeidors_config.json"
    $eidorsOnPyOutput = Join-Path $crossRoot "eidors_from_pyeidors.json"
    $eidorsOnPyCfg = [ordered]@{
        mesh_level = "medium"
        scenario = "low_z"
        source_framework = "pyeidors"
        n_elec = 16
        commit = $commit
        input_csv = "D:\workspace\PyEIDORS\results\simulation_parity\softwarex_cross_pyeidors_source\synthetic_forward_data.csv"
        output_json = $eidorsOnPyOutput
    }
    Write-JsonFile -Path $eidorsOnPyConfig -Data $eidorsOnPyCfg
    $exitCode = Invoke-ExternalCommand -Exe $matlabExe -Arguments @("-batch", "cd('$matlabScriptDir'); cross_generation_case('$($eidorsOnPyConfig -replace '\\','/')');")
    if ($exitCode -ne 0) { return $exitCode }

    foreach ($framework in @("pyeidors", "pyeit")) {
        $pyOut = "docs/benchmarks/reviewer_suite/fairness/raw_cross/${framework}_from_pyeidors.json"
        $cmd = "cd /root/shared && python $pythonCrossCase --framework $framework --source-framework pyeidors --input-csv $pyeidorsForward --source-metrics-json $pyeidorsMetrics --output-json $pyOut --mesh-level medium --scenario low_z"
        $exitCode = Invoke-ExternalCommand -Exe "docker" -Arguments @("exec", "pyeidors", "bash", "-lc", $cmd)
        if ($exitCode -ne 0) { return $exitCode }
    }

    foreach ($framework in @("pyeidors", "pyeit")) {
        $pyOut = "docs/benchmarks/reviewer_suite/fairness/raw_cross/${framework}_from_eidors.json"
        $cmd = "cd /root/shared && python $pythonCrossCase --framework $framework --source-framework eidors --input-csv docs/benchmarks/reviewer_suite/fairness/eidors_source_forward.csv --output-json $pyOut --mesh-level medium --scenario low_z"
        $exitCode = Invoke-ExternalCommand -Exe "docker" -Arguments @("exec", "pyeidors", "bash", "-lc", $cmd)
        if ($exitCode -ne 0) { return $exitCode }
    }
    return 0
}

function Invoke-MeshMatchedControlCase {
    $cmd = "cd /root/shared && python $pythonMeshControl --output-json docs/benchmarks/reviewer_suite/fairness/mesh_matched_control.json --output-csv docs/benchmarks/reviewer_suite/fairness/mesh_matched_control.csv --scenario low_z --matched-refinement 5 --finer-refinement 10 --reference-eidors-elements 2130"
    return Invoke-ExternalCommand -Exe "docker" -Arguments @("exec", "pyeidors", "bash", "-lc", $cmd)
}

function Invoke-AggregationCase {
    $cmd = "cd /root/shared && python $pythonAggregate --mode $Mode --state-dir docs/benchmarks/reviewer_suite/state"
    return Invoke-ExternalCommand -Exe "docker" -Arguments @("exec", "pyeidors", "bash", "-lc", $cmd)
}

function Invoke-Case {
    param([pscustomobject]$Case)

    $skip = Should-SkipCase -Case $Case
    if ($skip.Skip) {
        Append-RunStatusEvent -Case $Case -Status "skipped_existing" -Message "Reused existing result"
        return
    }

    Append-RunStatusEvent -Case $Case -Status "running" -Message "Case started"
    if ($Case.Kind -ne "benchmark" -or $Case.Runner -eq "matlab") {
        Set-CurrentCaseState -Status "running" -Case $Case
    }

    $rawExitCode = 0
    switch ($Case.Kind) {
        "benchmark" {
            if ($Case.Runner -eq "docker") {
                $rawExitCode = Invoke-DockerBenchmarkCase -Case $Case
            } else {
                $rawExitCode = Invoke-MatlabBenchmarkCase -Case $Case
            }
        }
        "meta" {
            switch ($Case.Label) {
                "llm_poc" { $rawExitCode = Invoke-LlmDemoCase }
                "cross_generation" { $rawExitCode = Invoke-CrossGenerationStudies }
                "mesh_matched_control" { $rawExitCode = Invoke-MeshMatchedControlCase }
                default { throw "Unknown meta case: $($Case.Label)" }
            }
        }
        "aggregate" {
            $rawExitCode = Invoke-AggregationCase
        }
        default {
            throw "Unknown case kind: $($Case.Kind)"
        }
    }

    try {
        $exitCode = ConvertTo-IntegerExitCode -Value $rawExitCode
    } catch {
        $message = $_.Exception.Message
        Append-RunStatusEvent -Case $Case -Status "failed" -Message $message
        Set-CurrentCaseState -Status "failed" -Case $Case -Extra @{
            exit_code = ""
            raw_exit_code = [string]$rawExitCode
        }
        if (-not $ContinueOnCaseError) {
            throw "Case failed: $($Case.CaseId) ($message)"
        }
        return
    }

    if ($exitCode -eq 0) {
        Append-RunStatusEvent -Case $Case -Status "completed" -Message "Case finished"
        if ($Case.Kind -ne "benchmark" -or $Case.Runner -eq "matlab") {
            Set-CurrentCaseState -Status "completed" -Case $Case
        }
        if ($Case.Kind -eq "aggregate") {
            [void](Invoke-AggregationCase)
        }
        return
    }

    Append-RunStatusEvent -Case $Case -Status "failed" -Message ("Exit code {0}" -f $exitCode)
    Set-CurrentCaseState -Status "failed" -Case $Case -Extra @{ exit_code = $exitCode }
    if (-not $ContinueOnCaseError) {
        throw "Case failed: $($Case.CaseId) (exit code $exitCode)"
    }
}

function Get-CaseRecord {
    param([pscustomobject]$Case)
    return [ordered]@{
        case_id = $Case.CaseId
        kind = $Case.Kind
        runner = $Case.Runner
        framework = $Case.Framework
        task = $Case.Task
        mesh_level = $Case.MeshLevel
        scenario = $Case.Scenario
        device = $Case.Device
        n_frames = [int]$Case.NFrames
        phase = $Case.PhaseName
        outputs = $Case.OutputRels
        label = $Case.Label
    }
}

function Write-HeavyCommands {
    $lines = @(
        "powershell -ExecutionPolicy Bypass -File D:\workspace\PyEIDORS\scripts\benchmarks\run_reviewer_suite.ps1 -Mode $Mode -Phase heavy -Warmups $Warmups -Repeats $Repeats",
        "Get-Content D:\workspace\PyEIDORS\docs\benchmarks\reviewer_suite\state\current_case.json"
    )
    Write-Utf8NoBom -Path $heavyCommandsPath -Content (($lines -join [Environment]::NewLine) + [Environment]::NewLine)
}

$coreCases = @()
$heavyCases = @()
$fairnessCases = @(
    (New-CaseSpec -Kind "meta" -Runner "docker" -Task "llm_poc" -PhaseName "fairness" -OutputRel "docs/benchmarks/reviewer_r1_q3/metrics.json" -Label "llm_poc"),
    (New-CaseSpec -Kind "meta" -Runner "mixed" -Task "cross_generation" -PhaseName "fairness" -OutputRels @(
        "docs/benchmarks/reviewer_suite/fairness/raw_cross/eidors_from_eidors.json",
        "docs/benchmarks/reviewer_suite/fairness/raw_cross/eidors_from_pyeidors.json",
        "docs/benchmarks/reviewer_suite/fairness/raw_cross/pyeidors_from_eidors.json",
        "docs/benchmarks/reviewer_suite/fairness/raw_cross/pyeidors_from_pyeidors.json",
        "docs/benchmarks/reviewer_suite/fairness/raw_cross/pyeit_from_eidors.json",
        "docs/benchmarks/reviewer_suite/fairness/raw_cross/pyeit_from_pyeidors.json"
    ) -Label "cross_generation"),
    (New-CaseSpec -Kind "meta" -Runner "docker" -Task "mesh_matched_control" -PhaseName "fairness" -OutputRels @(
        "docs/benchmarks/reviewer_suite/fairness/mesh_matched_control.json",
        "docs/benchmarks/reviewer_suite/fairness/mesh_matched_control.csv"
    ) -Label "mesh_matched_control")
)
$aggregateCase = New-CaseSpec -Kind "aggregate" -Runner "docker" -Task "aggregate" -PhaseName "aggregate" -OutputRel "docs/benchmarks/reviewer_suite/aggregated/aggregate_all_cases.json" -Label "aggregate"

if ($Mode -eq "smoke") {
    $coreCases += New-BenchmarkCase -Framework "pyeidors" -Task "forward" -MeshLevel "coarse"
    $coreCases += New-BenchmarkCase -Framework "pyeit" -Task "difference" -MeshLevel "coarse"
    $coreCases += New-BenchmarkCase -Framework "" -Task "difference" -MeshLevel "coarse" -Runner "matlab"
} else {
    $meshLevels = @("coarse", "medium", "fine")
    $frameCounts = @(1, 10, 100, 1000)

    foreach ($mesh in $meshLevels) {
        foreach ($case in @(
            (New-BenchmarkCase -Framework "pyeidors" -Task "forward" -MeshLevel $mesh),
            (New-BenchmarkCase -Framework "pyeidors" -Task "jacobian" -MeshLevel $mesh -Device "cpu"),
            (New-BenchmarkCase -Framework "pyeidors" -Task "jacobian" -MeshLevel $mesh -Device "gpu"),
            (New-BenchmarkCase -Framework "pyeidors" -Task "difference" -MeshLevel $mesh),
            (New-BenchmarkCase -Framework "pyeidors" -Task "absolute_gn" -MeshLevel $mesh -Device "cpu"),
            (New-BenchmarkCase -Framework "pyeidors" -Task "absolute_gn" -MeshLevel $mesh -Device "gpu"),
            (New-BenchmarkCase -Framework "pyeit" -Task "forward" -MeshLevel $mesh),
            (New-BenchmarkCase -Framework "pyeit" -Task "jacobian" -MeshLevel $mesh),
            (New-BenchmarkCase -Framework "pyeit" -Task "difference" -MeshLevel $mesh -Scenario "low_z"),
            (New-BenchmarkCase -Framework "pyeit" -Task "difference" -MeshLevel $mesh -Scenario "high_z"),
            (New-BenchmarkCase -Framework "pyeit" -Task "absolute_gn" -MeshLevel $mesh),
            (New-BenchmarkCase -Framework "" -Task "forward" -MeshLevel $mesh -Runner "matlab"),
            (New-BenchmarkCase -Framework "" -Task "jacobian" -MeshLevel $mesh -Runner "matlab"),
            (New-BenchmarkCase -Framework "" -Task "difference" -MeshLevel $mesh -Scenario "low_z" -Runner "matlab"),
            (New-BenchmarkCase -Framework "" -Task "difference" -MeshLevel $mesh -Scenario "high_z" -Runner "matlab"),
            (New-BenchmarkCase -Framework "" -Task "absolute_gn" -MeshLevel $mesh -Runner "matlab")
        )) {
            if ($case.PhaseName -eq "heavy") {
                $heavyCases += $case
            } else {
                $coreCases += $case
            }
        }
    }

    foreach ($frames in $frameCounts) {
        $coreCases += New-BenchmarkCase -Framework "pyeidors" -Task "multi_frame_difference" -MeshLevel "medium" -NFrames $frames
        $coreCases += New-BenchmarkCase -Framework "pyeit" -Task "multi_frame_difference" -MeshLevel "medium" -Scenario "low_z" -NFrames $frames
        $coreCases += New-BenchmarkCase -Framework "pyeit" -Task "multi_frame_difference" -MeshLevel "medium" -Scenario "high_z" -NFrames $frames
        $coreCases += New-BenchmarkCase -Framework "" -Task "multi_frame_difference" -MeshLevel "medium" -Scenario "low_z" -NFrames $frames -Runner "matlab"
        $coreCases += New-BenchmarkCase -Framework "" -Task "multi_frame_difference" -MeshLevel "medium" -Scenario "high_z" -NFrames $frames -Runner "matlab"
    }
}

$allCases = @($coreCases + $heavyCases + $fairnessCases + $aggregateCase)
$manifest = [ordered]@{
    run_id = $runId
    created_at = Get-NowIso
    mode = $Mode
    phase = $Phase
    warmups = $Warmups
    repeats = $Repeats
    skip_existing = $SkipExisting
    force = [bool]$Force
    continue_on_case_error = $ContinueOnCaseError
    commit = $commit
    cases = @($allCases | ForEach-Object { Get-CaseRecord $_ })
}
Write-JsonFile -Path $manifestPath -Data $manifest
Write-HeavyCommands
Set-CurrentCaseState -Status "idle" -Extra @{ mode = $Mode; selected_phase = $Phase }

switch ($Phase) {
    "all" {
        foreach ($case in $coreCases) {
            Invoke-Case -Case $case
        }
        foreach ($case in $heavyCases) {
            Append-RunStatusEvent -Case $case -Status "deferred" -Message "Deferred to heavy phase"
        }
        foreach ($case in $fairnessCases) {
            Invoke-Case -Case $case
        }
        Invoke-Case -Case $aggregateCase
    }
    "core" {
        foreach ($case in $coreCases) {
            Invoke-Case -Case $case
        }
    }
    "heavy" {
        foreach ($case in $heavyCases) {
            Invoke-Case -Case $case
        }
        Invoke-Case -Case $aggregateCase
    }
    "fairness" {
        foreach ($case in $fairnessCases) {
            Invoke-Case -Case $case
        }
    }
    "aggregate" {
        Invoke-Case -Case $aggregateCase
    }
}

Set-CurrentCaseState -Status "idle" -Extra @{ mode = $Mode; selected_phase = $Phase; finished_at = Get-NowIso }
Write-Host "[DONE] reviewer benchmark suite ($Phase)"
