%% Fair EIDORS P1 float64 classic/Robin CEM benchmark on the common mesh
clc; clear; close all;

eidorsStartup = 'D:\Program Files\MATLAB\R2023b\toolbox\eidors-v3.12-ng\eidors\startup.m';
if exist('eidors_default', 'file') ~= 2
    if exist(eidorsStartup, 'file') == 2
        run(eidorsStartup);
    else
        error('EIDORS startup script not found: %s', eidorsStartup);
    end
end

out_dir = getenv('CEM_BENCHMARK_OUTPUT_DIR');
if isempty(out_dir)
    script_dir = fileparts(mfilename('fullpath'));
    out_dir = fullfile(fileparts(script_dir), 'output', 'cem_formulation_comparison');
end
if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end
mesh_mat = getenv('CEM_COMMON_MESH_MAT');
if isempty(mesh_mat)
    mesh_mat = fullfile(out_dir, 'common_mesh', 'cem_common_p1.mat');
end
if exist(mesh_mat, 'file') ~= 2
    error('Common mesh MAT file not found: %s', mesh_mat);
end
payload = load(mesh_mat);

config.n_electrodes = double(payload.n_elec);
config.radius_m = 4.0;
config.conductivity_s_per_m = double(payload.background);
config.conductivity = config.conductivity_s_per_m;
config.contact_impedance = double(payload.contact_impedance);
config.electrode_coverage = double(payload.electrode_coverage);
config.potential_order = 1;
config.drive_skip = NaN;
if isfield(payload, 'drive_skip')
    config.drive_skip = double(payload.drive_skip);
end
config.timing_repeats = 11;
config.timing_operations_per_sample = 16;
repeat_override = str2double(getenv('CEM_TIMING_REPEATS'));
if isfinite(repeat_override) && repeat_override >= 3
    config.timing_repeats = floor(repeat_override);
end

nodes = double(payload.nodes);
elems = double(payload.elems);
boundary_edges = double(payload.boundary_edges);
electrode_nodes = double(payload.electrode_nodes);
electrode_counts = double(payload.electrode_node_counts(:));
currents = double(payload.current_patterns);
L = config.n_electrodes;
if size(currents, 1) ~= L
    error('Current matrix must have %d electrode rows.', L);
end

fmdl = eidors_obj('fwd_model', 'shared_pyeidors_cem_p1');
fmdl.nodes = nodes;
fmdl.elems = elems;
fmdl.boundary = boundary_edges;
fmdl.gnd_node = choose_ground_node(nodes, electrode_nodes, electrode_counts);
fmdl.solve = @fwd_solve_1st_order;
fmdl.system_mat = @system_mat_1st_order;
fmdl.jacobian = @jacobian_adjoint;
fmdl.normalize_measurements = 0;
for electrode = 1:L
    active_nodes = electrode_nodes(electrode, 1:electrode_counts(electrode));
    fmdl.electrode(electrode).nodes = active_nodes(active_nodes > 0);
    fmdl.electrode(electrode).z_contact = config.contact_impedance;
end
for pattern = 1:size(currents, 2)
    stimulation(pattern).stim_pattern = sparse(currents(:, pattern)); %#ok<SAGROW>
    stimulation(pattern).meas_pattern = speye(L); %#ok<SAGROW>
end
fmdl.stimulation = stimulation;
mesh_import_verified = isequal(fmdl.nodes, nodes) && ...
    isequal(fmdl.elems, elems) && isequal(fmdl.boundary, boundary_edges) && ...
    size(fmdl.electrode, 2) == L;
if ~mesh_import_verified
    error('EIDORS did not preserve the imported common mesh exactly.');
end

img = mk_image(fmdl, config.conductivity_s_per_m);
if isfield(payload, 'truth_elem_data')
    source_conductivity = double(payload.truth_elem_data(:));
    if numel(source_conductivity) ~= size(elems, 1)
        error('truth_elem_data must contain one conductivity per element.');
    end
    img.elem_data = source_conductivity;
end
img.fwd_solve.get_all_nodes = 1;
assembly_started = tic;
system_matrix = calc_system_mat(img);
assembly_seconds = toc(assembly_started);
E = system_matrix.E;
n_nodes = size(nodes, 1);
expected_size = n_nodes + L;
if size(E, 1) ~= expected_size || size(E, 2) ~= expected_size
    error('Expected a %d-by-%d EIDORS CEM matrix.', expected_size, expected_size);
end
A_R = E(1:n_nodes, 1:n_nodes);
C = E(1:n_nodes, (n_nodes + 1):(n_nodes + L));
D = E((n_nodes + 1):(n_nodes + L), (n_nodes + 1):(n_nodes + L));

[timing, classic_potential, classic_voltage, robin_potential, robin_voltage] = ...
    benchmark_preassembled_blocks(A_R, C, D, currents, config.timing_repeats, ...
    config.timing_operations_per_sample);
timing.assembly_seconds = assembly_seconds;
timing.mesh_import_seconds = 0;

voltage_relative_l2 = relative_l2(robin_voltage, classic_voltage);
potential_relative_l2 = relative_l2(robin_potential, classic_potential);
official_data = fwd_solve(img);
official_voltage = reshape(official_data.meas, L, []);
official_voltage = official_voltage - mean(official_voltage, 1);
official_classic_relative_l2 = relative_l2(official_voltage, classic_voltage);

k = 1:(L / 2);
solver_column = strings(2 * size(currents, 2), 1);
formulation_column = strings(2 * size(currents, 2), 1);
mode_column = strings(2 * size(currents, 2), 1);
frequency_column = zeros(2 * size(currents, 2), 1);
current_norm_column = zeros(2 * size(currents, 2), 1);
voltage_norm_column = zeros(2 * size(currents, 2), 1);
resistance_column = zeros(2 * size(currents, 2), 1);
row = 0;
for formulation_index = 1:2
    if formulation_index == 1
        formulation = "classic";
        voltage = classic_voltage;
    else
        formulation = "robin_transconductance";
        voltage = robin_voltage;
    end
    for pattern = 1:size(currents, 2)
        row = row + 1;
        solver_column(row) = "EIDORS";
        formulation_column(row) = formulation;
        if pattern <= length(k)
            mode_column(row) = "cosine";
            frequency_column(row) = k(pattern);
        else
            mode_column(row) = "sine";
            frequency_column(row) = k(pattern - length(k));
        end
        current_norm_column(row) = norm(currents(:, pattern));
        voltage_norm_column(row) = norm(voltage(:, pattern));
        resistance_column(row) = voltage_norm_column(row) / current_norm_column(row);
    end
end
results = table( ...
    solver_column, formulation_column, mode_column, frequency_column, ...
    current_norm_column, voltage_norm_column, resistance_column, ...
    'VariableNames', { ...
        'solver', 'formulation', 'mode', 'spatial_frequency', ...
        'current_norm_a', 'voltage_norm_v', 'characteristic_resistance_ohm'});
writetable(results, fullfile(out_dir, 'eidors_characteristic_resistance.csv'));
save(fullfile(out_dir, 'eidors_raw_voltages.mat'), ...
    'currents', 'classic_voltage', 'robin_voltage', ...
    'classic_potential', 'robin_potential', '-v7');
assembled_blocks = fullfile(out_dir, 'eidors_assembled_blocks.mat');
save(assembled_blocks, 'A_R', 'C', 'D', 'currents', '-v7');

report.solver = 'EIDORS';
report.suite_schema = '';
if isfield(payload, 'suite_schema')
    report.suite_schema = char(payload.suite_schema);
end
report.case_id = '';
if isfield(payload, 'case_id')
    report.case_id = char(payload.case_id);
end
report.eidors_version = eidors_obj('eidors_version');
report.interpreter_version = eidors_obj('interpreter_version');
report.physical_config = config;
report.physical_config.conductivity_pattern = 'uniform';
if isfield(payload, 'conductivity_pattern')
    report.physical_config.conductivity_pattern = char(payload.conductivity_pattern);
end
report.physical_config.conductivity_digest = '';
if isfield(payload, 'conductivity_digest')
    report.physical_config.conductivity_digest = char(payload.conductivity_digest);
end
report.discretization.vertices = size(nodes, 1);
report.discretization.cells = size(elems, 1);
report.discretization.boundary_edges = size(boundary_edges, 1);
report.discretization.degrees_of_freedom = n_nodes;
report.discretization.element_family = 'EIDORS P1 triangle';
report.discretization.potential_order = 1;
report.discretization.electrode_integration = 'EIDORS system_mat_fields CEM';
report.discretization.mesh_fingerprint_schema = char(payload.mesh_fingerprint_schema);
report.discretization.mesh_fingerprint = char(payload.mesh_fingerprint);
report.discretization.mesh_import_verified = mesh_import_verified;
report.discretization.common_mesh_role = 'imported MAT connectivity';
report.linear_solver.classic = 'MATLAB sparse LU on augmented CEM matrix';
report.linear_solver.robin = 'MATLAB sparse A_R LU plus dense reduced LU';
report.linear_solver.scalar_dtype = 'float64';
report.timing = timing;
report.assembled_blocks = assembled_blocks;
report.within_solver.electrode_voltage_relative_l2 = voltage_relative_l2;
report.within_solver.body_potential_relative_l2 = potential_relative_l2;
report.within_solver.classic_voltage_balance_max_abs = ...
    max(abs(sum(classic_voltage, 1)));
report.within_solver.robin_voltage_balance_max_abs = ...
    max(abs(sum(robin_voltage, 1)));
report.within_solver.official_fwd_solve_vs_block_classic_relative_l2 = ...
    official_classic_relative_l2;
report.raw_electrode_voltages.classic = classic_voltage;
report.raw_electrode_voltages.robin_transconductance = robin_voltage;
report.implementation_note = [ ...
    'EIDORS directly imports the canonical node/element/electrode arrays. ', ...
    'Official fwd_solve is validation-only; timing uses independent classic ', ...
    'and Robin factor states on identical preassembled A_R/C/D blocks.' ...
];

fid = fopen(fullfile(out_dir, 'eidors_report.json'), 'w');
if fid < 0
    error('Could not create EIDORS JSON report in %s.', out_dir);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid, jsonencode(report, 'PrettyPrint', true), 'char');
fprintf('\nEIDORS classic/Robin relative L2: %.6e\n', voltage_relative_l2);
fprintf('EIDORS CEM benchmark artifacts: %s\n', out_dir);


function [timing, classic_potential, classic_voltage, robin_potential, robin_voltage] = ...
    benchmark_preassembled_blocks(A_R, C, D, currents, repeats, operations_per_sample)
classic_cold = zeros(repeats, 1);
robin_cold = zeros(repeats, 1);
classic_setup = zeros(repeats, 1);
robin_setup = zeros(repeats, 1);
classic_cold_solve = zeros(repeats, 1);
robin_cold_solve = zeros(repeats, 1);
% Prime MATLAB dispatch/allocator paths, then discard both fresh states.
cold_classic(A_R, C, D, currents);
cold_robin(A_R, C, D, currents);
for repetition = 1:repeats
    if mod(repetition, 2) == 1
        [classic_cold(repetition), classic_setup(repetition), ...
            classic_cold_solve(repetition), classic_potential, classic_voltage] = ...
            sample_cold_classic(A_R, C, D, currents, operations_per_sample);
        [robin_cold(repetition), robin_setup(repetition), ...
            robin_cold_solve(repetition), robin_potential, robin_voltage] = ...
            sample_cold_robin(A_R, C, D, currents, operations_per_sample);
    else
        [robin_cold(repetition), robin_setup(repetition), ...
            robin_cold_solve(repetition), robin_potential, robin_voltage] = ...
            sample_cold_robin(A_R, C, D, currents, operations_per_sample);
        [classic_cold(repetition), classic_setup(repetition), ...
            classic_cold_solve(repetition), classic_potential, classic_voltage] = ...
            sample_cold_classic(A_R, C, D, currents, operations_per_sample);
    end
end

classic_state = build_classic_state(A_R, C, D);
robin_state = build_robin_state(A_R, C, D);
classic_warm = zeros(repeats, 1);
robin_warm = zeros(repeats, 1);
for repetition = 1:repeats
    if mod(repetition, 2) == 1
        [robin_warm(repetition), robin_potential, robin_voltage] = ...
            sample_warm_robin(robin_state, currents, operations_per_sample);
        [classic_warm(repetition), classic_potential, classic_voltage] = ...
            sample_warm_classic(classic_state, currents, operations_per_sample);
    else
        [classic_warm(repetition), classic_potential, classic_voltage] = ...
            sample_warm_classic(classic_state, currents, operations_per_sample);
        [robin_warm(repetition), robin_potential, robin_voltage] = ...
            sample_warm_robin(robin_state, currents, operations_per_sample);
    end
end

timing.schema = 'cem-fair-timing-v2';
timing.scope = 'preassembled_A_R_C_D_blocks';
timing.repeats = repeats;
timing.operations_per_sample = operations_per_sample;
timing.rhs_count = size(currents, 2);
timing.alternating_order = true;
timing.untimed_runtime_priming = true;
timing.cross_formulation_cache_reuse = false;
timing.paired_cold_decomposition = true;
timing.classic.cold_seconds = summarize_samples(classic_cold);
timing.classic.setup_seconds = summarize_samples(classic_setup);
timing.classic.cold_solve_seconds = summarize_samples(classic_cold_solve);
timing.classic.warm_reuse_seconds = summarize_samples(classic_warm);
timing.classic.cold_over_warm_reuse_speedup = ...
    timing.classic.cold_seconds.median / timing.classic.warm_reuse_seconds.median;
timing.classic.cold_sparse_factorizations = repeats * operations_per_sample;
timing.classic.cold_dense_factorizations = 0;
timing.classic.warm_sparse_factorizations = 1;
timing.classic.warm_dense_factorizations = 0;
timing.classic.warm_cache_hits = repeats * operations_per_sample;
timing.classic.rhs_solves_per_sample = size(currents, 2);
timing.robin_transconductance.cold_seconds = summarize_samples(robin_cold);
timing.robin_transconductance.setup_seconds = summarize_samples(robin_setup);
timing.robin_transconductance.cold_solve_seconds = summarize_samples(robin_cold_solve);
timing.robin_transconductance.warm_reuse_seconds = summarize_samples(robin_warm);
timing.robin_transconductance.cold_over_warm_reuse_speedup = ...
    timing.robin_transconductance.cold_seconds.median / ...
    timing.robin_transconductance.warm_reuse_seconds.median;
timing.robin_transconductance.cold_sparse_factorizations = ...
    repeats * operations_per_sample;
timing.robin_transconductance.cold_dense_factorizations = ...
    repeats * operations_per_sample;
timing.robin_transconductance.warm_sparse_factorizations = 1;
timing.robin_transconductance.warm_dense_factorizations = 1;
timing.robin_transconductance.warm_cache_hits = repeats * operations_per_sample;
timing.robin_transconductance.rhs_solves_per_sample = size(currents, 2);
timing.robin_transconductance.response_basis_rhs_count = size(D, 1) - 1;
end


function [total_mean, setup_mean, solve_mean, potential, voltage] = ...
    sample_cold_classic(A_R, C, D, currents, operations)
totals = zeros(operations, 1);
setups = zeros(operations, 1);
solves = zeros(operations, 1);
for operation = 1:operations
    total_started = tic;
    setup_started = tic;
    state = build_classic_state(A_R, C, D);
    setups(operation) = toc(setup_started);
    solve_started = tic;
    [potential, voltage] = solve_classic(state, currents);
    solves(operation) = toc(solve_started);
    totals(operation) = toc(total_started);
end
total_mean = mean(totals);
setup_mean = mean(setups);
solve_mean = mean(solves);
end


function [total_mean, setup_mean, solve_mean, potential, voltage] = ...
    sample_cold_robin(A_R, C, D, currents, operations)
totals = zeros(operations, 1);
setups = zeros(operations, 1);
solves = zeros(operations, 1);
for operation = 1:operations
    total_started = tic;
    setup_started = tic;
    state = build_robin_state(A_R, C, D);
    setups(operation) = toc(setup_started);
    solve_started = tic;
    [potential, voltage] = solve_robin(state, currents);
    solves(operation) = toc(solve_started);
    totals(operation) = toc(total_started);
end
total_mean = mean(totals);
setup_mean = mean(setups);
solve_mean = mean(solves);
end


function [elapsed_mean, potential, voltage] = ...
    sample_warm_classic(state, currents, operations)
samples = zeros(operations, 1);
for operation = 1:operations
    started = tic;
    [potential, voltage] = solve_classic(state, currents);
    samples(operation) = toc(started);
end
elapsed_mean = mean(samples);
end


function [elapsed_mean, potential, voltage] = ...
    sample_warm_robin(state, currents, operations)
samples = zeros(operations, 1);
for operation = 1:operations
    started = tic;
    [potential, voltage] = solve_robin(state, currents);
    samples(operation) = toc(started);
end
elapsed_mean = mean(samples);
end


function state = build_classic_state(A_R, C, D)
n_nodes = size(A_R, 1);
L = size(D, 1);
constraint = sparse(ones(L, 1));
matrix = [A_R, C, sparse(n_nodes, 1); ...
    C.', D, constraint; sparse(1, n_nodes), constraint.', sparse(1, 1)];
state.factor = decomposition(matrix, 'lu');
state.n_nodes = n_nodes;
state.n_electrodes = L;
end


function [potential, voltage] = solve_classic(state, currents)
rhs = zeros(state.n_nodes + state.n_electrodes + 1, size(currents, 2));
rhs((state.n_nodes + 1):(state.n_nodes + state.n_electrodes), :) = currents;
solution = state.factor \ rhs;
potential = solution(1:state.n_nodes, :);
voltage = solution((state.n_nodes + 1):(state.n_nodes + state.n_electrodes), :);
end


function [potential, voltage] = cold_classic(A_R, C, D, currents)
state = build_classic_state(A_R, C, D);
[potential, voltage] = solve_classic(state, currents);
end


function state = build_robin_state(A_R, C, D)
L = size(D, 1);
Q = helmert_basis(L);
state.body_factor = decomposition(A_R, 'lu');
state.response_basis = state.body_factor \ (C * Q);
reduced_map = Q.' * (D * Q - C.' * state.response_basis);
state.reduced_factor = decomposition(full(reduced_map), 'lu');
state.electrode_basis = Q;
end


function [potential, voltage] = solve_robin(state, currents)
coefficients = state.reduced_factor \ (state.electrode_basis.' * currents);
potential = -(state.response_basis * coefficients);
voltage = state.electrode_basis * coefficients;
end


function [potential, voltage] = cold_robin(A_R, C, D, currents)
state = build_robin_state(A_R, C, D);
[potential, voltage] = solve_robin(state, currents);
end


function Q = helmert_basis(L)
Q = zeros(L, L - 1);
for column = 1:(L - 1)
    scale = sqrt(column * (column + 1));
    Q(1:column, column) = 1 / scale;
    Q(column + 1, column) = -column / scale;
end
end


function summary = summarize_samples(samples)
raw_values = double(samples(:));
values = sort(raw_values);
summary.samples = raw_values.';
summary.median = median(values);
summary.iqr = percentile_linear(values, 0.75) - percentile_linear(values, 0.25);
summary.minimum = min(values);
summary.maximum = max(values);
end


function value = percentile_linear(sorted_values, fraction)
position = 1 + (numel(sorted_values) - 1) * fraction;
lower = floor(position);
upper = ceil(position);
if lower == upper
    value = sorted_values(lower);
else
    weight = position - lower;
    value = sorted_values(lower) * (1 - weight) + sorted_values(upper) * weight;
end
end


function value = relative_l2(candidate, reference)
value = norm(candidate - reference, 'fro') / max(norm(reference, 'fro'), eps);
end


function gnd_node = choose_ground_node(nodes, electrode_nodes, electrode_counts)
mask = false(size(nodes, 1), 1);
for index = 1:numel(electrode_counts)
    active = electrode_nodes(index, 1:electrode_counts(index));
    active = active(active > 0);
    mask(active) = true;
end
free_nodes = find(~mask);
if isempty(free_nodes)
    gnd_node = 1;
    return;
end
[~, nearest] = min(sum(nodes(free_nodes, :).^2, 2));
gnd_node = free_nodes(nearest);
end
