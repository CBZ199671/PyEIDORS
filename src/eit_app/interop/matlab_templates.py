"""MATLAB bridge templates used by the Interop Hub."""

from __future__ import annotations


CAPTURE_SCRIPT_TEMPLATE = r"""function run_capture_from_eidors(config_path)
if nargin < 1
    error('run_capture_from_eidors requires a JSON config path.');
end

raw_config = fileread(config_path);
if ~isempty(raw_config) && double(raw_config(1)) == 65279
    raw_config = raw_config(2:end);
end
cfg = jsondecode(raw_config);

if exist('eidors_default', 'file') ~= 2
    if isfield(cfg, 'eidors_startup') && exist(cfg.eidors_startup, 'file') == 2
        run(cfg.eidors_startup);
    else
        error('EIDORS startup script not found.');
    end
end

target_script = char(cfg.target_script);
if exist(target_script, 'file') ~= 2
    error('Target EIDORS script not found: %s', target_script);
end

script_dir = fileparts(target_script);
if isempty(script_dir)
    script_dir = pwd;
end

orig_dir = pwd;
cleanup = onCleanup(@() cd(orig_dir));
cd(script_dir);
run(target_script);

fmdl = local_pick_fmdl();
imdl = local_pick_imdl();
img = local_pick_image();
vh = local_pick_data({'vh', 'vhom', 'data_homogeneous'});
vi = local_pick_data({'vi', 'vtarget', 'data_target', 'v'});

if isempty(fmdl)
    error('No fwd_model could be discovered from the target script workspace.');
end

out_dir = char(cfg.output_dir);
if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end

exchange_format = 'eidors_pyeidors_geometry_v2';
schema_version = 2;
index_base = 1;
source_framework = 'eidors';
nodes = double(fmdl.nodes);
elems = double(fmdl.elems);
dimension = double(size(nodes, 2));
if dimension == 2
    cell_type = 'triangle';
    boundary_entity_type = 'edge';
elseif dimension == 3
    cell_type = 'tetrahedron';
    boundary_entity_type = 'triangle';
else
    error('Only 2D triangle and 3D tetrahedron EIDORS models are supported.');
end
if isfield(fmdl, 'boundary')
    boundary_facets = double(fmdl.boundary);
else
    boundary_facets = double(find_boundary(fmdl.elems));
end
boundary_edges = boundary_facets; % v1/MATLAB compatibility alias
[electrode_nodes, electrode_node_counts, contact_impedance] = local_build_electrode_node_arrays(fmdl);
[stim_matrix, meas_matrices, measurement_counts] = local_build_pattern_arrays(fmdl);
n_elec = double(numel(fmdl.electrode));
normalize_measurements = double(local_normalize_measurements(fmdl));
background = local_background_value(img);
truth_elem_data = local_truth_elem_data(img, background, size(elems, 1));
mesh_name = local_mesh_name(fmdl, target_script);
mesh_level = 'script_capture';
scenario_name = local_script_kind(cfg);

save(fullfile(out_dir, 'geometry.mat'), ...
    'exchange_format', ...
    'schema_version', ...
    'index_base', ...
    'source_framework', ...
    'dimension', ...
    'cell_type', ...
    'boundary_entity_type', ...
    'nodes', ...
    'elems', ...
    'boundary_edges', ...
    'boundary_facets', ...
    'electrode_nodes', ...
    'electrode_node_counts', ...
    'stim_matrix', ...
    'meas_matrices', ...
    'measurement_counts', ...
    'n_elec', ...
    'normalize_measurements', ...
    'background', ...
    'truth_elem_data', ...
    'contact_impedance', ...
    'mesh_name', ...
    'mesh_level', ...
    'scenario_name');

if ~isempty(vh) && ~isempty(vi)
    vh_meas = double(vh.meas(:));
    vi_meas = double(vi.meas(:));
    difference = vi_meas - vh_meas;
    T = table(vh_meas, vi_meas, difference, 'VariableNames', ...
        {'meas_homogeneous', 'meas_phantom', 'difference'});
    writetable(T, fullfile(out_dir, 'measurements.csv'));
end

raw_vars = whos;
capture_report = struct();
capture_report.script_path = target_script;
capture_report.script_kind = local_script_kind(cfg);
capture_report.fmdl_found = ~isempty(fmdl);
capture_report.imdl_found = ~isempty(imdl);
capture_report.img_found = ~isempty(img);
capture_report.vh_found = ~isempty(vh);
capture_report.vi_found = ~isempty(vi);
capture_report.workspace_vars = {raw_vars.name};
json_text = jsonencode(capture_report, PrettyPrint=true);
fid = fopen(fullfile(out_dir, 'capture_report.json'), 'w');
fprintf(fid, '%s', json_text);
fclose(fid);
end

function value = local_pick_fmdl()
value = [];
if evalin('caller', 'exist(''fmdl'', ''var'')')
    value = evalin('caller', 'fmdl');
    return;
end
if evalin('caller', 'exist(''imdl'', ''var'')')
    imdl_candidate = evalin('caller', 'imdl');
    if isstruct(imdl_candidate) && isfield(imdl_candidate, 'fwd_model')
        value = imdl_candidate.fwd_model;
        return;
    end
end
if evalin('caller', 'exist(''img'', ''var'')')
    img_candidate = evalin('caller', 'img');
    if isstruct(img_candidate) && isfield(img_candidate, 'fwd_model')
        value = img_candidate.fwd_model;
    end
    return;
end
end

function value = local_pick_imdl()
value = [];
if evalin('caller', 'exist(''imdl'', ''var'')')
    value = evalin('caller', 'imdl');
end
end

function value = local_pick_image()
value = [];
for name = {'img_truth', 'img', 'img_bg'}
    candidate = char(name);
    if evalin('caller', sprintf('exist(''%s'', ''var'')', candidate))
        value = evalin('caller', candidate);
        return;
    end
end
end

function value = local_pick_data(names)
value = [];
for idx = 1:numel(names)
    candidate = char(names{idx});
    if evalin('caller', sprintf('exist(''%s'', ''var'')', candidate))
        temp = evalin('caller', candidate);
        if isstruct(temp) && isfield(temp, 'meas')
            value = temp;
            return;
        end
    end
end
end

function [electrode_nodes, electrode_node_counts, contact_impedance] = local_build_electrode_node_arrays(fmdl)
n_elec = numel(fmdl.electrode);
electrode_node_counts = zeros(n_elec, 1);
max_nodes = 0;
contact_impedance = zeros(n_elec, 1);
for i = 1:n_elec
    nodes = double(fmdl.electrode(i).nodes(:)');
    electrode_node_counts(i) = numel(nodes);
    max_nodes = max(max_nodes, numel(nodes));
    if isfield(fmdl.electrode(i), 'z_contact')
        contact_impedance(i) = double(fmdl.electrode(i).z_contact);
    else
        contact_impedance(i) = 0.01;
    end
end
electrode_nodes = zeros(n_elec, max_nodes);
for i = 1:n_elec
    nodes = double(fmdl.electrode(i).nodes(:)');
    electrode_nodes(i, 1:numel(nodes)) = nodes;
end
if numel(unique(contact_impedance)) == 1
    contact_impedance = contact_impedance(1);
end
end

function [stim_matrix, meas_matrices, measurement_counts] = local_build_pattern_arrays(fmdl)
n_elec = numel(fmdl.electrode);
if ~isfield(fmdl, 'stimulation') || isempty(fmdl.stimulation)
    stim_matrix = zeros(0, n_elec);
    meas_matrices = zeros(0, 0, n_elec);
    measurement_counts = zeros(0, 1);
    return;
end
n_stim = numel(fmdl.stimulation);
measurement_counts = zeros(n_stim, 1);
max_measurements = 0;
stim_matrix = zeros(n_stim, n_elec);
for i = 1:n_stim
    stim_matrix(i, :) = full(double(fmdl.stimulation(i).stim_pattern(:)'));
    measurement_counts(i) = size(fmdl.stimulation(i).meas_pattern, 1);
    max_measurements = max(max_measurements, measurement_counts(i));
end
meas_matrices = zeros(n_stim, max_measurements, n_elec);
for i = 1:n_stim
    count = measurement_counts(i);
    one_meas = full(double(fmdl.stimulation(i).meas_pattern));
    meas_matrices(i, 1:count, :) = reshape(one_meas, 1, count, n_elec);
end
end

function value = local_normalize_measurements(fmdl)
if isfield(fmdl, 'normalize_measurements')
    value = fmdl.normalize_measurements;
else
    value = 0;
end
end

function value = local_background_value(img)
if isempty(img) || ~isfield(img, 'elem_data') || isempty(img.elem_data)
    value = 1.0;
else
    value = double(median(img.elem_data(:)));
end
end

function truth = local_truth_elem_data(img, background, n_elems)
if isempty(img) || ~isfield(img, 'elem_data') || isempty(img.elem_data)
    truth = ones(n_elems, 1) * background;
else
    truth = double(img.elem_data(:));
end
end

function name = local_mesh_name(fmdl, target_script)
if isfield(fmdl, 'name') && ~isempty(fmdl.name)
    name = char(fmdl.name);
else
    [~, stem, ~] = fileparts(target_script);
    name = char(stem);
end
end

function kind = local_script_kind(cfg)
if isfield(cfg, 'script_kind') && ~isempty(cfg.script_kind)
    kind = char(cfg.script_kind);
else
    kind = 'script_capture';
end
end
"""


RUN_IN_EIDORS_TEMPLATE = r"""this_file = mfilename('fullpath');
if isempty(this_file)
    base_dir = pwd;
else
    base_dir = fileparts(this_file);
end
config_path = fullfile(base_dir, 'bridge_runtime.json');
if exist(config_path, 'file') ~= 2
    error('bridge_runtime.json was not found next to this script.');
end

raw_config = fileread(config_path);
if ~isempty(raw_config) && double(raw_config(1)) == 65279
    raw_config = raw_config(2:end);
end
cfg = jsondecode(raw_config);

if exist('eidors_default', 'file') ~= 2
    if isfield(cfg, 'eidors_startup') && exist(cfg.eidors_startup, 'file') == 2
        run(cfg.eidors_startup);
    else
        error('EIDORS startup script not found.');
    end
end

payload = load(cfg.geometry_mat);
nodes = double(payload.nodes);
elems = double(payload.elems);
boundary_edges = double(payload.boundary_edges);
electrode_nodes = double(payload.electrode_nodes);
electrode_counts = double(payload.electrode_node_counts(:));
contact_impedance = double(payload.contact_impedance);
background = double(payload.background);
stim_pattern = local_or_default(cfg, 'stim_pattern', '{ad}');
meas_pattern = local_or_default(cfg, 'meas_pattern', '{ad}');
rotate_meas = logical(local_or_default(cfg, 'rotate_meas', true));
use_meas_current = logical(local_or_default(cfg, 'use_meas_current', false));
drive_value = double(local_or_default(cfg, 'drive_value', 1.0));

n_elec = double(size(electrode_nodes, 1));
fmdl = eidors_obj('fwd_model', 'pyeidors_bridge_geometry');
fmdl.nodes = nodes;
fmdl.elems = elems;
fmdl.boundary = boundary_edges;
fmdl.gnd_node = 1;
fmdl.solve = @fwd_solve_1st_order;
fmdl.system_mat = @system_mat_1st_order;
fmdl.jacobian = @jacobian_adjoint;
fmdl.normalize_measurements = 0;
if isfield(payload, 'normalize_measurements')
    fmdl.normalize_measurements = logical(payload.normalize_measurements);
end

for i = 1:n_elec
    active_nodes = electrode_nodes(i, 1:electrode_counts(i));
    fmdl.electrode(i).nodes = active_nodes(active_nodes > 0);
    if numel(contact_impedance) == 1
        fmdl.electrode(i).z_contact = contact_impedance;
    else
        fmdl.electrode(i).z_contact = contact_impedance(i);
    end
end

if isfield(payload, 'stim_matrix') && ~isempty(payload.stim_matrix) && ...
        isfield(payload, 'meas_matrices') && isfield(payload, 'measurement_counts')
    stim_matrix = double(payload.stim_matrix);
    measurement_counts = double(payload.measurement_counts(:));
    for i = 1:size(stim_matrix, 1)
        fmdl.stimulation(i).stim_pattern = sparse(stim_matrix(i, :)');
        count = measurement_counts(i);
        one_meas = double(payload.meas_matrices(i, 1:count, :));
        fmdl.stimulation(i).meas_pattern = sparse(reshape(one_meas, count, n_elec));
    end
else
    stim_options = {};
    if rotate_meas
        stim_options{end+1} = 'rotate_meas';
    else
        stim_options{end+1} = 'no_rotate_meas';
    end
    if use_meas_current
        stim_options{end+1} = 'meas_current';
    else
        stim_options{end+1} = 'no_meas_current';
    end
    fmdl.stimulation = mk_stim_patterns(n_elec, 1, stim_pattern, meas_pattern, ...
        stim_options, drive_value);
end

if isfield(cfg, 'measurements_csv') && exist(cfg.measurements_csv, 'file') == 2
    T = readtable(cfg.measurements_csv);
    vh_meas = double(T{:, 1});
    vi_meas = double(T{:, 2});
    vh = eidors_obj('data', 'pyeidors_bridge_homogeneous'); %#ok<NASGU>
    vh.meas = vh_meas;
    vi = eidors_obj('data', 'pyeidors_bridge_target'); %#ok<NASGU>
    vi.meas = vi_meas;
    fprintf('EIDORS bridge measurements loaded: %d points\n', numel(vi_meas));
end

fprintf('EIDORS bridge project loaded from %s\n', cfg.geometry_mat);

function value = local_or_default(cfg, field_name, default_value)
if isfield(cfg, field_name)
    value = cfg.(field_name);
else
    value = default_value;
end
end
"""
