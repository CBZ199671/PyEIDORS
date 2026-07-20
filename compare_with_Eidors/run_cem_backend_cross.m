function count = run_cem_backend_cross(manifest_path)
%% Solve every bit-fixed CEM block payload with MATLAB sparse LU.
if nargin < 1 || isempty(manifest_path)
    manifest_path = getenv('CEM_BACKEND_CROSS_MANIFEST');
end
if isempty(manifest_path) || exist(manifest_path, 'file') ~= 2
    error('Backend-cross manifest not found: %s', manifest_path);
end
manifest = jsondecode(fileread(manifest_path));
entries = manifest.records;
empty_voltages = struct('classic', [], 'robin_transconductance', []);
empty_record = struct( ...
    'case_id', '', ...
    'assembly', '', ...
    'backend', '', ...
    'block_sha256', '', ...
    'raw_electrode_voltages', empty_voltages);
records = repmat(empty_record, numel(entries), 1);
for index = 1:numel(entries)
    entry = entries(index);
    payload = load(wsl_path_to_unc(entry.block_path));
    A_R = double(payload.A_R);
    C = double(payload.C);
    D = double(payload.D);
    currents = double(payload.currents);
    [classic_voltage, robin_voltage] = solve_both(A_R, C, D, currents);
    record.case_id = char(entry.case_id);
    record.assembly = char(entry.assembly);
    record.backend = 'matlab_sparse_lu';
    record.block_sha256 = char(entry.block_sha256);
    record.raw_electrode_voltages.classic = classic_voltage;
    record.raw_electrode_voltages.robin_transconductance = robin_voltage;
    records(index) = record;
end
output.schema = char(manifest.schema);
output.backend = 'matlab_sparse_lu';
output.matlab_version = version;
output.records = records;
output_path = wsl_path_to_unc(manifest.matlab_output);
fid = fopen(output_path, 'w');
if fid < 0
    error('Could not create MATLAB backend-cross output: %s', output_path);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid, jsonencode(output, 'PrettyPrint', true), 'char');
count = numel(records);
fprintf('MATLAB backend-cross records: %d\n', count);
end


function [classic_voltage, robin_voltage] = solve_both(A_R, C, D, currents)
n_nodes = size(A_R, 1);
L = size(D, 1);
constraint = sparse(ones(L, 1));
classic_matrix = [A_R, C, sparse(n_nodes, 1); ...
    C.', D, constraint; sparse(1, n_nodes), constraint.', sparse(1, 1)];
classic_factor = decomposition(classic_matrix, 'lu');
rhs = zeros(n_nodes + L + 1, size(currents, 2));
rhs((n_nodes + 1):(n_nodes + L), :) = currents;
classic_solution = classic_factor \ rhs;
classic_voltage = classic_solution((n_nodes + 1):(n_nodes + L), :);

Q = helmert_basis(L);
body_factor = decomposition(A_R, 'lu');
response_basis = body_factor \ (C * Q);
reduced_map = Q.' * (D * Q - C.' * response_basis);
reduced_factor = decomposition(full(reduced_map), 'lu');
coefficients = reduced_factor \ (Q.' * currents);
robin_voltage = Q * coefficients;
end


function Q = helmert_basis(L)
Q = zeros(L, L - 1);
for column = 1:(L - 1)
    scale = sqrt(column * (column + 1));
    Q(1:column, column) = 1 / scale;
    Q(column + 1, column) = -column / scale;
end
end


function converted = wsl_path_to_unc(path_value)
converted = char(path_value);
prefix = '/home/';
if startsWith(converted, prefix)
    converted = ['\\wsl.localhost\Ubuntu-22.04', strrep(converted, '/', '\')];
end
end
