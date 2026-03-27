function export_geometry_from_eidors(config_path)
% Export EIDORS geometry in a standardized exchange format for PyEIDORS.

if nargin < 1
    error('export_geometry_from_eidors requires a JSON config path.');
end

raw_config = fileread(config_path);
if ~isempty(raw_config) && double(raw_config(1)) == 65279
    raw_config = raw_config(2:end);
end
cfg = jsondecode(raw_config);

eidorsStartup = 'D:\Program Files\MATLAB\R2023b\toolbox\eidors-v3.12-ng\eidors\startup.m';
if exist('eidors_default', 'file') ~= 2
    if exist('eidors_startup', 'file') == 2
        eidors_startup;
    elseif exist(eidorsStartup, 'file') == 2
        run(eidorsStartup);
    else
        error('EIDORS startup script not found.');
    end
end

mesh_levels = struct('coarse', 3.0, 'medium', 1.6, 'fine', 0.9);
if ~isfield(mesh_levels, cfg.mesh_level)
    error('Unknown mesh level: %s', cfg.mesh_level);
end
maxh = mesh_levels.(cfg.mesh_level);
scenario = get_scenario(cfg.scenario);
n_elec = double(cfg.n_elec);

th = linspace(0, 360, n_elec + 1)';
th(1) = [];
els = (th + 67.5) * [1, 0];
elec_sz = 3;
radius_scale = 30.0;
fmdl = ng_mk_cyl_models([0, radius_scale, maxh], els, [elec_sz, 0, 1]);
for i = 1:n_elec
    fmdl.electrode(i).z_contact = scenario.contact_impedance;
end
fmdl.stimulation = mk_stim_patterns(n_elec, 1, '{ad}', '{ad}', ...
    {'no_meas_current', 'rotate_meas'}, 1.0);

img_bg = mk_image(fmdl, scenario.background);
img_truth = img_bg;
pos = interp_mesh(fmdl) / radius_scale;
idx = (pos(:,1) - scenario.obj_pos(1)).^2 + (pos(:,2) - scenario.obj_pos(2)).^2 < scenario.obj_radius.^2;
img_truth.elem_data(idx) = scenario.obj_sigma;

vh = fwd_solve(img_bg);
vi = fwd_solve(img_truth);
target_diff = calc_difference_data(vh.meas, vi.meas, fmdl);

exchange_format = 'eidors_pyeidors_bridge_v1';
source_framework = 'eidors';
nodes = double(fmdl.nodes);
elems = double(fmdl.elems);
electrodes = fmdl.electrode;
boundary_edges = double(fmdl.boundary);
[electrode_nodes, electrode_node_counts] = build_electrode_node_arrays(electrodes);
n_elec = double(numel(electrodes));
background = double(scenario.background);
truth_elem_data = double(img_truth.elem_data(:));
object_mask = logical(idx(:));
contact_impedance = double(scenario.contact_impedance);
mesh_name = sprintf('eidors_maxh_%.2f', maxh);
mesh_level = char(cfg.mesh_level);
scenario_name = char(cfg.scenario);
obj_pos = double(scenario.obj_pos);
obj_radius = double(scenario.obj_radius);
obj_sigma = double(scenario.obj_sigma);

out_path = char(cfg.output_mat);
out_dir = fileparts(out_path);
if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end

if isfield(cfg, 'forward_export_csv') && ~isempty(cfg.forward_export_csv)
    forward_path = char(cfg.forward_export_csv);
    forward_dir = fileparts(forward_path);
    if exist(forward_dir, 'dir') ~= 7
        mkdir(forward_dir);
    end
    T = table(double(vh.meas(:)), double(vi.meas(:)), double(target_diff(:)), ...
        'VariableNames', {'meas_homogeneous', 'meas_phantom', 'difference'});
    writetable(T, forward_path);
    fprintf('Wrote %s\n', forward_path);
end

save(out_path, ...
    'exchange_format', ...
    'source_framework', ...
    'nodes', ...
    'elems', ...
    'electrodes', ...
    'boundary_edges', ...
    'electrode_nodes', ...
    'electrode_node_counts', ...
    'n_elec', ...
    'background', ...
    'truth_elem_data', ...
    'object_mask', ...
    'contact_impedance', ...
    'mesh_name', ...
    'mesh_level', ...
    'scenario_name', ...
    'radius_scale', ...
    'obj_pos', ...
    'obj_radius', ...
    'obj_sigma');

fprintf('Wrote %s\n', out_path);
end

function [electrode_nodes, electrode_node_counts] = build_electrode_node_arrays(electrodes)
n_elec = numel(electrodes);
electrode_node_counts = zeros(n_elec, 1);
max_nodes = 0;

for i = 1:n_elec
    nodes = double(electrodes(i).nodes(:)');
    electrode_node_counts(i) = numel(nodes);
    max_nodes = max(max_nodes, numel(nodes));
end

electrode_nodes = zeros(n_elec, max_nodes);
for i = 1:n_elec
    nodes = double(electrodes(i).nodes(:)');
    electrode_nodes(i, 1:numel(nodes)) = nodes;
end
end

function scenario = get_scenario(name)
switch char(name)
    case 'low_z'
        scenario = struct( ...
            'contact_impedance', 1e-6, ...
            'background', 1.0, ...
            'obj_sigma', 2.0, ...
            'obj_pos', [0.30, 0.20], ...
            'obj_radius', 0.20);
    case 'high_z'
        scenario = struct( ...
            'contact_impedance', 1e-2, ...
            'background', 1.0, ...
            'obj_sigma', 2.0, ...
            'obj_pos', [0.25, -0.22], ...
            'obj_radius', 0.18);
    otherwise
        error('Unknown scenario: %s', name);
end
end
