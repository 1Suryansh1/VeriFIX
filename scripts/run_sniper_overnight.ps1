$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $repoRoot

$python = Resolve-Path ".venv/Scripts/python.exe"
$checkpoint = ".v3_checkpoints/cycle_2026_04_09_dirfix_e20_train02/v3_multitask_gat.pt"
$splitArtifact = ".analysis/quixbugs_split_20_20.json"

if (-not (Test-Path $python)) {
    throw "Python interpreter not found: $python"
}
if (-not (Test-Path $checkpoint)) {
    throw "Checkpoint not found: $checkpoint"
}
if (-not (Test-Path $splitArtifact)) {
    throw "Split artifact not found: $splitArtifact"
}

$timestamp = Get-Date -Format "yyyy_MM_dd_HHmmss"
$suiteDir = ".analysis/sniper_suite_$timestamp"
$runsDir = Join-Path $suiteDir "runs"
$logsDir = Join-Path $suiteDir "logs"
New-Item -ItemType Directory -Force -Path $suiteDir | Out-Null
New-Item -ItemType Directory -Force -Path $runsDir | Out-Null
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

# Cohesive no-training ablation plan from UPDATE_BRO findings:
# 1) Keep topology and weights fixed (depth=1, branch=1, node/action=0.6/0.4)
# 2) Primary sweep: validation budget (dominant failure bucket)
# 3) Secondary sweeps: max patch candidates, time budget, mild critic calibration
$nodeWeight = 0.6
$actionWeight = 0.4

$runs = @(
    # Validation ladder (core signal)
    @{ id = "V1_control_val025"; set = "validation"; iter = 40; val = 25;  maxPatch = 500; maxPerNode = 10; fl = 5; time = 20; ct = 0.00; hypothesis = "baseline control (expected ~3/20 ceiling)" },
    @{ id = "V2_val050";         set = "validation"; iter = 40; val = 50;  maxPatch = 500; maxPerNode = 10; fl = 5; time = 20; ct = 0.00; hypothesis = "test moderate validation expansion" },
    @{ id = "V3_val100";         set = "validation"; iter = 40; val = 100; maxPatch = 500; maxPerNode = 10; fl = 5; time = 20; ct = 0.00; hypothesis = "test strong validation expansion" },
    @{ id = "V4_val150";         set = "validation"; iter = 40; val = 150; maxPatch = 500; maxPerNode = 10; fl = 5; time = 25; ct = 0.00; hypothesis = "test near-max validation expansion" },
    @{ id = "V5_val200";         set = "validation"; iter = 40; val = 200; maxPatch = 500; maxPerNode = 10; fl = 5; time = 30; ct = 0.00; hypothesis = "test maximum validation pressure" },

    # Max patch candidate pressure (ranking/truncation axis)
    @{ id = "P1_patch300";       set = "max_patch";  iter = 40; val = 100; maxPatch = 300; maxPerNode = 10; fl = 5; time = 20; ct = 0.00; hypothesis = "check if smaller candidate pool improves precision" },
    @{ id = "P2_patch800";       set = "max_patch";  iter = 40; val = 100; maxPatch = 800; maxPerNode = 10; fl = 5; time = 20; ct = 0.00; hypothesis = "check if larger candidate pool improves recall" },

    # Time budget pressure (search compute axis)
    @{ id = "T1_time10";         set = "time_budget";iter = 40; val = 100; maxPatch = 500; maxPerNode = 10; fl = 5; time = 10; ct = 0.00; hypothesis = "test aggressive early cutoff" },
    @{ id = "T2_time40";         set = "time_budget";iter = 40; val = 100; maxPatch = 500; maxPerNode = 10; fl = 5; time = 40; ct = 0.00; hypothesis = "test extra search time for harder programs" },

    # Mild critic calibration (avoid known overfiltering at high thresholds)
    @{ id = "C1_ct010";          set = "critic";     iter = 40; val = 100; maxPatch = 500; maxPerNode = 10; fl = 5; time = 20; ct = 0.10; hypothesis = "test very soft critic gating" },
    @{ id = "C2_ct030";          set = "critic";     iter = 40; val = 100; maxPatch = 500; maxPerNode = 10; fl = 5; time = 20; ct = 0.30; hypothesis = "test soft critic gating" },
    @{ id = "C3_ct040";          set = "critic";     iter = 40; val = 100; maxPatch = 500; maxPerNode = 10; fl = 5; time = 20; ct = 0.45; hypothesis = "test upper soft-gating before overfilter risk" },
     @{ id = "C4_ct060";          set = "critic";     iter = 40; val = 100; maxPatch = 500; maxPerNode = 10; fl = 5; time = 20; ct = 0.60; hypothesis = "test upper soft-gating before overfilter risk" }
)

$manifest = [PSCustomObject]@{
    generated_at = (Get-Date -Format o)
    suite_dir = $suiteDir
    run_count = $runs.Count
    rationale = @(
        "Validation budget exhaustion dominates failures in prior traces",
        "Depth/branch ablations were low-value on this checkpoint",
        "High critic thresholds harmed hybrid; only mild critic is tested"
    )
    swept_knobs = @("max_validations", "max_patch_candidates", "time_budget", "v3_critic_threshold")
    fixed_knobs = [PSCustomObject]@{
        device = "cpu"
        mcts_max_depth = 1
        v3_min_rollout_depth = 1
        v3_branch_per_state = 1
        v3_candidate_node_weight = $nodeWeight
        v3_candidate_action_weight = $actionWeight
        run_v1 = $false
        run_v2 = $false
        run_v3_hybrid = $true
        run_v3_latent = $true
    }
    runs = $runs
}
$manifestPath = Join-Path $suiteDir "manifest.json"
$manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $manifestPath -Encoding UTF8

$successfulRuns = @()
$failedRuns = @()

foreach ($run in $runs) {
    $runId = "v4_v3sniper_$($run.id)_$timestamp"
    $outJson = Join-Path $runsDir "$runId.json"
    $logPath = Join-Path $logsDir "$runId.log"
    $workDir = ".work_holdout_ablation_v4_$runId"

    Write-Host ""
    Write-Host "=== START $runId ==="

    $benchArgs = @(
        "scripts/benchmark_holdout_ablation_v4.py",
        "--checkpoint", $checkpoint,
        "--split-artifact", $splitArtifact,
        "--split-side", "test",
        "--output-json", $outJson,
        "--device", "cpu",
        "--mcts-iterations", [string]$run.iter,
        "--mcts-max-depth", "1",
        "--max-validations", [string]$run.val,
        "--max-patch-candidates", [string]$run.maxPatch,
        "--max-candidates-per-node", [string]$run.maxPerNode,
        "--fl-top-n-lines", [string]$run.fl,
        "--time-budget", [string]$run.time,
        "--v3-min-rollout-depth", "1",
        "--v3-branch-per-state", "1",
        "--v3-critic-threshold", [string]$run.ct,
        "--v3-candidate-node-weight", [string]$nodeWeight,
        "--v3-candidate-action-weight", [string]$actionWeight,
        "--working-dir", $workDir,
        "--search-profile-name", $run.id,
        "--no-run-v1",
        "--no-run-v2",
        "--run-v3-hybrid",
        "--run-v3-latent",
        "--show-progress"
    )

    & $python @benchArgs 2>&1 | Tee-Object -FilePath $logPath

    if ($LASTEXITCODE -eq 0 -and (Test-Path $outJson)) {
        $successfulRuns += [PSCustomObject]@{
            RunId = $runId
            RunSet = $run.set
            OutputJson = $outJson
            Log = $logPath
            Hypothesis = $run.hypothesis
            Iterations = [int]$run.iter
            ValidationBudget = [int]$run.val
            MaxPatchCandidates = [int]$run.maxPatch
            TimeBudgetSeconds = [double]$run.time
            CriticThreshold = [double]$run.ct
        }
        Write-Host "=== DONE $runId (success) ==="
    }
    else {
        $failedRuns += [PSCustomObject]@{
            RunId = $runId
            Log = $logPath
            Hypothesis = $run.hypothesis
        }
        Write-Host "=== DONE $runId (failed, continuing) ==="
    }
}

if ($successfulRuns.Count -eq 0) {
    throw "No successful run JSON produced. Check logs in $logsDir"
}

$goldenJson = Join-Path $suiteDir "v3_golden_trace_sniper_$timestamp.json"
$goldenMd = Join-Path $suiteDir "v3_golden_trace_sniper_$timestamp.md"
$goldenLog = Join-Path $logsDir "golden_trace_$timestamp.log"

$goldenArgs = @("scripts/extract_v3_golden_trace.py")
foreach ($entry in $successfulRuns) {
    $goldenArgs += "--run-json"
    $goldenArgs += $entry.OutputJson
}
$goldenArgs += @(
    "--modes", "v3_hybrid,v3_latent",
    "--matrix-json", ".quixbugs_operator_matrix_after_phase2_batch4c.json",
    "--ideal-actions-json", ".analysis/_tmp_program_40_latest_with_action_guess.json",
    "--output-json", $goldenJson,
    "--output-md", $goldenMd,
    "--top-k", "25"
)

Write-Host ""
Write-Host "=== START GOLDEN TRACE ==="
& $python @goldenArgs 2>&1 | Tee-Object -FilePath $goldenLog
if ($LASTEXITCODE -ne 0) {
    throw "Golden trace extraction failed. Check $goldenLog"
}
Write-Host "=== DONE GOLDEN TRACE ==="

$summaryRows = @()
foreach ($entry in $successfulRuns) {
    $payload = Get-Content -Path $entry.OutputJson -Raw | ConvertFrom-Json
    $h = $payload.summary.v3_hybrid
    $l = $payload.summary.v3_latent

    $hRep = [int]$h.repaired
    $hAtt = [int]$h.attempted
    $lRep = [int]$l.repaired
    $lAtt = [int]$l.attempted
    $hRate = if ($hAtt -gt 0) { [double]$hRep / [double]$hAtt } else { 0.0 }
    $lRate = if ($lAtt -gt 0) { [double]$lRep / [double]$lAtt } else { 0.0 }
    $barrier = if (($hRep -gt 3) -or ($lRep -gt 3)) { "YES" } else { "NO" }

    $summaryRows += [PSCustomObject]@{
        RunId = $entry.RunId
        RunSet = $entry.RunSet
        Hypothesis = $entry.Hypothesis
        Iterations = $entry.Iterations
        ValidationBudget = $entry.ValidationBudget
        MaxPatchCandidates = $entry.MaxPatchCandidates
        TimeBudgetSeconds = $entry.TimeBudgetSeconds
        CriticThreshold = $entry.CriticThreshold
        HybridRepaired = $hRep
        HybridAttempted = $hAtt
        HybridRate = [Math]::Round($hRate, 3)
        LatentRepaired = $lRep
        LatentAttempted = $lAtt
        LatentRate = [Math]::Round($lRate, 3)
        BarrierBroken = $barrier
        OutputJson = $entry.OutputJson
        Log = $entry.Log
    }
}

$control = $summaryRows | Where-Object { $_.RunId -like "*V1_control_val025*" } | Select-Object -First 1
if ($null -ne $control) {
    foreach ($row in $summaryRows) {
        $row | Add-Member -NotePropertyName HybridDeltaVsControl -NotePropertyValue ($row.HybridRepaired - $control.HybridRepaired)
        $row | Add-Member -NotePropertyName LatentDeltaVsControl -NotePropertyValue ($row.LatentRepaired - $control.LatentRepaired)
    }
}

$tableMdPath = Join-Path $suiteDir "suite_table.md"
$tableCsvPath = Join-Path $suiteDir "suite_table.csv"
$summaryJsonPath = Join-Path $suiteDir "suite_summary.json"

$md = @(
    "# V3 Sniper Cohesive Suite",
    "",
    "Generated: $(Get-Date -Format o)",
    "",
    "| Run | Set | Hypothesis | Iter | Val | Patch | Time | Ct | Hybrid | Latent | Break3/20 | dHybrid | dLatent |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"
)

foreach ($row in $summaryRows) {
    $dh = if ($row.PSObject.Properties.Name -contains "HybridDeltaVsControl") { [int]$row.HybridDeltaVsControl } else { 0 }
    $dl = if ($row.PSObject.Properties.Name -contains "LatentDeltaVsControl") { [int]$row.LatentDeltaVsControl } else { 0 }
    $md += "| $($row.RunId) | $($row.RunSet) | $($row.Hypothesis) | $($row.Iterations) | $($row.ValidationBudget) | $($row.MaxPatchCandidates) | $($row.TimeBudgetSeconds) | $($row.CriticThreshold) | $($row.HybridRepaired)/$($row.HybridAttempted) | $($row.LatentRepaired)/$($row.LatentAttempted) | $($row.BarrierBroken) | $dh | $dl |"
}

$bestHybrid = $summaryRows | Sort-Object HybridRepaired -Descending | Select-Object -First 1
$bestLatent = $summaryRows | Sort-Object LatentRepaired -Descending | Select-Object -First 1
$barrierRuns = $summaryRows | Where-Object { $_.BarrierBroken -eq "YES" }

$md += ""
$md += "## Best Runs"
$md += "- Hybrid best: $($bestHybrid.HybridRepaired)/$($bestHybrid.HybridAttempted) in $($bestHybrid.RunId)"
$md += "- Latent best: $($bestLatent.LatentRepaired)/$($bestLatent.LatentAttempted) in $($bestLatent.RunId)"
$md += ""
$md += "## Barrier Break Status"
if ($barrierRuns.Count -gt 0) {
    foreach ($row in $barrierRuns) {
        $md += "- $($row.RunId)"
    }
}
else {
    $md += "- No run exceeded 3/20"
}

if ($failedRuns.Count -gt 0) {
    $md += ""
    $md += "## Failed Runs"
    foreach ($row in $failedRuns) {
        $md += "- $($row.RunId)"
    }
}

$md += ""
$md += "## Artifacts"
$md += "- Manifest: $manifestPath"
$md += "- Golden JSON: $goldenJson"
$md += "- Golden Markdown: $goldenMd"
$md += "- Logs dir: $logsDir"

Set-Content -Path $tableMdPath -Value ($md -join "`r`n") -Encoding UTF8
$summaryRows | Export-Csv -Path $tableCsvPath -NoTypeInformation -Encoding UTF8

$summaryPayload = [PSCustomObject]@{
    generated_at = (Get-Date -Format o)
    suite_dir = $suiteDir
    manifest = $manifestPath
    successful_runs = $successfulRuns.Count
    failed_runs = $failedRuns.Count
    rows = $summaryRows
    best_hybrid = $bestHybrid
    best_latent = $bestLatent
    barrier_runs = $barrierRuns
    golden_json = $goldenJson
    golden_markdown = $goldenMd
    table_markdown = $tableMdPath
    table_csv = $tableCsvPath
}
$summaryPayload | ConvertTo-Json -Depth 8 | Set-Content -Path $summaryJsonPath -Encoding UTF8

Write-Host ""
Write-Host "=== FINAL TABLE ==="
Get-Content -Path $tableMdPath
Write-Host ""
Write-Host "Cohesive suite done. Artifacts in: $suiteDir"
