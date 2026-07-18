%% Compare EIDORS classic CEM with its Robin-transconductance Schur form
clc; clear; close all;

eidorsStartup = 'D:\Program Files\MATLAB\R2023b\toolbox\eidors-v3.12-ng\eidors\startup.m';
if exist('eidors_default', 'file') ~= 2
    if exist(eidorsStartup, 'file') == 2
        run(eidorsStartup);
    else
        error('EIDORS startup script not found: %s', eidorsStartup);
    end
end

config.n_electrodes = 16;
config.radius_m = 4.0;
config.conductivity_s_per_m = 0.25;
config.contact_impedance = 1.0;
config.electrode_coverage = 0.7;
config.maxh_m = 0.10;
config.potential_order = 1;

L = config.n_electrodes;
k = 1:(L / 2);
electrode_index = (0:(L - 1))';
mid_theta = 2 * pi * (electrode_index + config.electrode_coverage / 2) / L;
I_cos = cos(mid_theta * k);
I_sin = sin(mid_theta * k);
currents = [I_cos, I_sin];
currents = currents - mean(currents, 1);

electrode_size = 2 * pi * config.radius_m / L * config.electrode_coverage;
electrode_positions = [rad2deg(mid_theta), zeros(L, 1)];
fmdl = ng_mk_cyl_models( ...
    [0, config.radius_m, config.maxh_m], ...
    electrode_positions, ...
    [electrode_size, 0, 1]);
for electrode = 1:L
    fmdl.electrode(electrode).z_contact = config.contact_impedance;
end
fmdl.normalize_measurements = 0;
for pattern = 1:size(currents, 2)
    stimulation(pattern).stim_pattern = sparse(currents(:, pattern)); %#ok<SAGROW>
    stimulation(pattern).meas_pattern = speye(L); %#ok<SAGROW>
end
fmdl.stimulation = stimulation;
img = mk_image(fmdl, config.conductivity_s_per_m);
img.fwd_solve.get_all_nodes = 1;

classic_started = tic;
classic_data = fwd_solve(img);
classic_seconds = toc(classic_started);
classic_voltage = reshape(classic_data.meas, L, []);
classic_voltage = classic_voltage - mean(classic_voltage, 1);

system_matrix = calc_system_mat(img);
E = system_matrix.E;
n_nodes = size(fmdl.nodes, 1);
expected_size = n_nodes + L;
if size(E, 1) ~= expected_size
    error('Expected %d node/electrode unknowns, got %d.', expected_size, size(E, 1));
end
A_R = E(1:n_nodes, 1:n_nodes);
C = E(1:n_nodes, (n_nodes + 1):(n_nodes + L));
D = E((n_nodes + 1):(n_nodes + L), (n_nodes + 1):(n_nodes + L));

Q = zeros(L, L - 1);
for column = 1:(L - 1)
    scale = sqrt(column * (column + 1));
    Q(1:column, column) = 1 / scale;
    Q(column + 1, column) = -column / scale;
end

robin_started = tic;
response_basis = A_R \ (C * Q);
reduced_map = Q.' * (D * Q - C.' * response_basis);
reduced_coefficients = reduced_map \ (Q.' * currents);
robin_voltage = Q * reduced_coefficients;
robin_seconds = toc(robin_started);

voltage_relative_l2 = norm(robin_voltage - classic_voltage, 'fro') / ...
    max(norm(classic_voltage, 'fro'), eps);
response_relative_residual = norm(A_R * response_basis - C * Q, 'fro') / ...
    max(norm(C * Q, 'fro'), eps);

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
    'VariableNames', {
        'solver', 'formulation', 'mode', 'spatial_frequency', ...
        'current_norm_a', 'voltage_norm_v', 'characteristic_resistance_ohm'});

out_dir = getenv('CEM_BENCHMARK_OUTPUT_DIR');
if isempty(out_dir)
    script_dir = fileparts(mfilename('fullpath'));
    out_dir = fullfile(fileparts(script_dir), 'output', 'cem_formulation_comparison');
end
if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end
writetable(results, fullfile(out_dir, 'eidors_characteristic_resistance.csv'));
save(fullfile(out_dir, 'eidors_raw_voltages.mat'), ...
    'currents', 'classic_voltage', 'robin_voltage', '-v7');

report.solver = 'EIDORS';
report.eidors_version = eidors_obj('eidors_version');
report.interpreter_version = eidors_obj('interpreter_version');
report.physical_config = config;
report.discretization.nodes = size(fmdl.nodes, 1);
report.discretization.elements = size(fmdl.elems, 1);
report.discretization.element_family = 'EIDORS first-order triangle';
report.discretization.potential_order = config.potential_order;
report.discretization.electrode_integration = 'EIDORS system_mat_fields CEM';
report.linear_solver.classic = 'EIDORS fwd_solve_1st_order';
report.linear_solver.robin = 'MATLAB sparse backslash A_R plus reduced map';
report.linear_solver.scalar_dtype = 'float64';
report.within_solver.electrode_voltage_relative_l2 = voltage_relative_l2;
report.within_solver.classic_voltage_balance_max_abs = ...
    max(abs(sum(classic_voltage, 1)));
report.within_solver.robin_voltage_balance_max_abs = ...
    max(abs(sum(robin_voltage, 1)));
report.within_solver.classic_seconds = classic_seconds;
report.within_solver.robin_seconds = robin_seconds;
report.robin_diagnostics.rank = rank(full(reduced_map));
report.robin_diagnostics.condition_number = cond(full(reduced_map));
report.robin_diagnostics.response_relative_residual = response_relative_residual;
report.implementation_note = [
    'Robin form is the exact Schur complement of the EIDORS classic CEM ', ...
    'matrix; reciprocal products use non-conjugate transpose.' ...
];

fid = fopen(fullfile(out_dir, 'eidors_report.json'), 'w');
if fid < 0
    error('Could not create EIDORS JSON report in %s.', out_dir);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid, jsonencode(report, 'PrettyPrint', true), 'char');
fprintf('\nEIDORS classic/Robin relative L2: %.6e\n', voltage_relative_l2);
fprintf('EIDORS CEM benchmark artifacts: %s\n', out_dir);
