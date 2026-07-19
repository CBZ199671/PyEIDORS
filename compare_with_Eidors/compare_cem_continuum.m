function report = compare_cem_continuum(out_dir, mesh_mat)
%% EIDORS P1 float64 solve for one true-circle continuum-suite fixture

eidorsStartup = 'D:\Program Files\MATLAB\R2023b\toolbox\eidors-v3.12-ng\eidors\startup.m';
if exist('eidors_default', 'file') ~= 2
    if exist(eidorsStartup, 'file') == 2
        run(eidorsStartup);
    else
        error('EIDORS startup script not found: %s', eidorsStartup);
    end
end

if nargin < 1 || isempty(out_dir)
    out_dir = getenv('CEM_CONTINUUM_OUTPUT_DIR');
end
if nargin < 2 || isempty(mesh_mat)
    mesh_mat = getenv('CEM_CONTINUUM_MESH_MAT');
end
if isempty(out_dir) || isempty(mesh_mat)
    error('CEM_CONTINUUM_OUTPUT_DIR and CEM_CONTINUUM_MESH_MAT are required.');
end
if exist(mesh_mat, 'file') ~= 2
    error('Continuum common mesh MAT file not found: %s', mesh_mat);
end
if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end
payload = load(mesh_mat);

nodes = double(payload.nodes);
elems = double(payload.elems);
boundary_edges = double(payload.boundary_edges);
electrode_nodes = double(payload.electrode_nodes);
electrode_counts = double(payload.electrode_node_counts(:));
currents = double(payload.current_patterns);
L = double(payload.n_elec);
conductivity = double(payload.background);
contact_impedance = double(payload.contact_impedance);

fmdl = eidors_obj('fwd_model', 'true_circle_continuum_common_p1');
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
    fmdl.electrode(electrode).z_contact = contact_impedance;
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
    error('EIDORS did not preserve the true-circle common mesh exactly.');
end

img = mk_image(fmdl, conductivity);
img.fwd_solve.get_all_nodes = 1;
system_matrix = calc_system_mat(img);
E = system_matrix.E;
n_nodes = size(nodes, 1);
if size(E, 1) ~= n_nodes + L || size(E, 2) ~= n_nodes + L
    error('Unexpected EIDORS CEM matrix size.');
end
A_R = E(1:n_nodes, 1:n_nodes);
C = E(1:n_nodes, (n_nodes + 1):(n_nodes + L));
D = E((n_nodes + 1):(n_nodes + L), (n_nodes + 1):(n_nodes + L));

classic_state = build_classic_state(A_R, C, D);
[classic_potential, classic_voltage] = solve_classic(classic_state, currents);
robin_state = build_robin_state(A_R, C, D);
[robin_potential, robin_voltage] = solve_robin(robin_state, currents);

report.solver = 'EIDORS';
report.suite_schema = char(payload.suite_schema);
report.case_id = char(payload.case_id);
report.mesh_level_id = char(payload.mesh_level);
report.eidors_version = eidors_obj('eidors_version');
report.interpreter_version = eidors_obj('interpreter_version');
report.physical_config.radius = 1.0;
report.physical_config.n_electrodes = L;
report.physical_config.electrode_coverage = double(payload.electrode_coverage);
report.physical_config.conductivity = conductivity;
report.physical_config.contact_impedance = contact_impedance;
report.physical_config.drive_skip = double(payload.drive_skip);
report.discretization.vertices = size(nodes, 1);
report.discretization.cells = size(elems, 1);
report.discretization.boundary_edges = size(boundary_edges, 1);
report.discretization.degrees_of_freedom = n_nodes;
report.discretization.element_family = 'EIDORS P1 triangle';
report.discretization.potential_order = 1;
report.discretization.scalar_dtype = 'float64';
report.discretization.mesh_fingerprint_schema = char(payload.mesh_fingerprint_schema);
report.discretization.mesh_fingerprint = char(payload.mesh_fingerprint);
report.discretization.mesh_import_verified = mesh_import_verified;
report.discretization.target_h = double(payload.target_h);
report.discretization.h_max = double(payload.h_max);
report.discretization.boundary_chord_max = double(payload.boundary_chord_max);
report.discretization.boundary_sagitta_max = double(payload.boundary_sagitta_max);
report.linear_solver.classic = 'MATLAB sparse LU on augmented CEM matrix';
report.linear_solver.robin = 'MATLAB sparse A_R LU plus dense reduced LU';
report.linear_solver.scalar_dtype = 'float64';
report.within_solver.electrode_voltage_relative_l2 = ...
    relative_l2(robin_voltage, classic_voltage);
report.within_solver.body_potential_relative_l2 = ...
    relative_l2(robin_potential, classic_potential);
report.raw_electrode_voltages.classic = classic_voltage;
report.raw_electrode_voltages.robin_transconductance = robin_voltage;
report.implementation_note = [ ...
    'EIDORS imports the canonical MAT P1 true-circle chord mesh unchanged, ', ...
    'assembles official CEM blocks, and solves independent Classic and Robin ', ...
    'float64 systems.' ...
];

fid = fopen(fullfile(out_dir, 'eidors_report.json'), 'w');
if fid < 0
    error('Could not create EIDORS continuum report in %s.', out_dir);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid, jsonencode(report, 'PrettyPrint', true), 'char');
fprintf('EIDORS continuum report: %s\n', out_dir);
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


function Q = helmert_basis(L)
Q = zeros(L, L - 1);
for column = 1:(L - 1)
    scale = sqrt(column * (column + 1));
    Q(1:column, column) = 1 / scale;
    Q(column + 1, column) = -column / scale;
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
