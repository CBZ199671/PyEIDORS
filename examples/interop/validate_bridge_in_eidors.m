function report = validate_bridge_in_eidors(run_script, report_path)
%VALIDATE_BRIDGE_IN_EIDORS Rebuild and forward-solve a Bridge Package in EIDORS.
%
% report = validate_bridge_in_eidors(run_script)
% report = validate_bridge_in_eidors(run_script, report_path)
%
% `run_script` is the generated `run_in_eidors.m`. The optional JSON report
% makes a real MATLAB/EIDORS acceptance run reproducible and machine-readable.

if nargin < 1 || isempty(run_script)
    error('A generated run_in_eidors.m path is required.');
end
run_script = char(run_script);
if exist(run_script, 'file') ~= 2
    error('Bridge import script was not found: %s', run_script);
end
if nargin < 2 || isempty(report_path)
    report_path = fullfile(fileparts(run_script), 'eidors_import_report.json');
end
report_path = char(report_path);

run(run_script);
if ~exist('fmdl', 'var') || ~exist('payload', 'var')
    error('The Bridge import script did not create fmdl and payload.');
end

n_stimulations = numel(fmdl.stimulation);
n_measurements = 0;
for i = 1:n_stimulations
    n_measurements = n_measurements + size(fmdl.stimulation(i).meas_pattern, 1);
end

boundary_exact = isequal(double(fmdl.boundary), double(payload.boundary_facets));
electrodes_exact = numel(fmdl.electrode) == double(payload.n_elec);
if electrodes_exact
    counts = double(payload.electrode_node_counts(:));
    for i = 1:numel(fmdl.electrode)
        expected = double(payload.electrode_nodes(i, 1:counts(i)));
        actual = double(fmdl.electrode(i).nodes(:)');
        electrodes_exact = electrodes_exact && isequal(actual, expected);
    end
end

protocol_exact = isfield(payload, 'stim_matrix') && ...
    isfield(payload, 'meas_matrices') && ...
    isfield(payload, 'measurement_counts') && ...
    n_stimulations == size(payload.stim_matrix, 1);
if protocol_exact
    measurement_counts = double(payload.measurement_counts(:));
    for i = 1:n_stimulations
        expected_stim = double(payload.stim_matrix(i, :)');
        actual_stim = full(double(fmdl.stimulation(i).stim_pattern(:)));
        count = measurement_counts(i);
        expected_meas = reshape( ...
            double(payload.meas_matrices(i, 1:count, :)), ...
            count, ...
            numel(fmdl.electrode));
        actual_meas = full(double(fmdl.stimulation(i).meas_pattern));
        protocol_exact = protocol_exact && ...
            isequal(size(actual_meas), size(expected_meas)) && ...
            max(abs(actual_stim - expected_stim), [], 'all') <= 1e-12 && ...
            max(abs(actual_meas - expected_meas), [], 'all') <= 1e-12;
    end
end

img_check = mk_image(fmdl, double(payload.background));
data_check = fwd_solve(img_check);
forward_finite = ~isempty(data_check.meas) && ...
    all(isfinite(real(data_check.meas(:)))) && ...
    all(isfinite(imag(data_check.meas(:))));
forward_count_exact = numel(data_check.meas) == n_measurements;

report = struct();
report.schema = 'eidors_bridge_import_acceptance_v1';
report.status = 'passed';
report.geometry_format = char(payload.exchange_format);
report.dimension = size(fmdl.nodes, 2);
report.n_nodes = size(fmdl.nodes, 1);
report.n_elements = size(fmdl.elems, 1);
report.n_boundary_facets = size(fmdl.boundary, 1);
report.n_electrodes = numel(fmdl.electrode);
report.n_stimulations = n_stimulations;
report.n_measurements = n_measurements;
report.boundary_exact = logical(boundary_exact);
report.electrodes_exact = logical(electrodes_exact);
report.protocol_exact = logical(protocol_exact);
report.forward_finite = logical(forward_finite);
report.forward_count_exact = logical(forward_count_exact);
report.eidors_version = eidors_obj('eidors_version');

passed = boundary_exact && electrodes_exact && protocol_exact && ...
    forward_finite && forward_count_exact;
if ~passed
    report.status = 'failed';
end

json_text = jsonencode(report, PrettyPrint=true);
fid = fopen(report_path, 'w');
if fid < 0
    error('Unable to write EIDORS acceptance report: %s', report_path);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s', json_text);
clear cleanup;

fprintf('EIDORS Bridge acceptance: %s (%d nodes, %d elements, %d measurements)\n', ...
    report.status, report.n_nodes, report.n_elements, report.n_measurements);
if ~passed
    error('EIDORS Bridge acceptance checks failed. See %s', report_path);
end
end
