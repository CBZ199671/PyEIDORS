function import_geometry_from_pyeidors(config_path)
% Import a PyEIDORS geometry into EIDORS and run a same-geometry difference solve.

if nargin < 1
    error('import_geometry_from_pyeidors requires a JSON config path.');
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

payload = load(char(cfg.mesh_mat));
if isfield(payload, 'exchange_format')
    exchange_format = char(payload.exchange_format);
else
    exchange_format = 'eidors_pyeidors_bridge_v1';
end
if isfield(payload, 'source_framework')
    source_framework = char(payload.source_framework);
else
    source_framework = 'pyeidors';
end
nodes = double(payload.nodes);
elems = double(payload.elems);
if isfield(payload, 'boundary_edges')
    boundary_edges = double(payload.boundary_edges);
else
    boundary_edges = double(find_boundary(elems));
end
electrode_nodes = double(payload.electrode_nodes);
electrode_counts = double(payload.electrode_node_counts(:));
background = double(payload.background);
truth_elem_data = double(payload.truth_elem_data(:));
contact_impedance = double(payload.contact_impedance);

n_elec = double(size(electrode_nodes, 1));
fmdl = eidors_obj('fwd_model', 'imported_pyeidors_geometry');
fmdl.nodes = nodes;
fmdl.elems = elems;
fmdl.boundary = boundary_edges;
fmdl.gnd_node = choose_ground_node(nodes, electrode_nodes, electrode_counts);
fmdl.solve = @fwd_solve_1st_order;
fmdl.system_mat = @system_mat_1st_order;
fmdl.jacobian = @jacobian_adjoint;
fmdl.normalize_measurements = 0;

for i = 1:n_elec
    active_nodes = electrode_nodes(i, 1:electrode_counts(i));
    fmdl.electrode(i).nodes = active_nodes(active_nodes > 0);
    fmdl.electrode(i).z_contact = contact_impedance;
end

fmdl.stimulation = mk_stim_patterns(n_elec, 1, '{ad}', '{ad}', ...
    {'no_meas_current', 'rotate_meas'}, 1.0);

img_bg = mk_image(fmdl, background);
img_truth = img_bg;
img_truth.elem_data = truth_elem_data;

T = readtable(char(cfg.input_csv));
vh_meas = double(T.meas_homogeneous);
vi_meas = double(T.meas_phantom);
if ismember('difference', T.Properties.VariableNames)
    target_diff = double(T.difference);
else
    target_diff = vi_meas - vh_meas;
end

imdl = eidors_obj('inv_model', 'imported_pyeidors_diff');
imdl.fwd_model = fmdl;
imdl.rec_model = fmdl;
imdl.reconst_type = 'difference';
imdl.solve = @inv_solve_diff_GN_one_step;
imdl.jacobian_bkgnd.value = 1.0;
imdl.hyperparameter.value = 0.01;
imdl.RtR_prior = @prior_noser;
imdl.inv_solve_diff_GN_one_step.calc_step_size = true;
imdl.inv_solve_diff_GN_one_step.bounds = [1e-5, 10];

t0 = tic;
rimg = inv_solve(imdl, vh_meas, vi_meas);
runtime = toc(t0);

img_rec = img_bg;
img_rec.elem_data = img_rec.elem_data + rimg.elem_data;
pred_vh = fwd_solve(img_bg);
pred_vi = fwd_solve(img_rec);
pred_diff = calc_difference_data(pred_vh.meas, pred_vi.meas, fmdl);

metrics = voltage_metrics(target_diff, pred_diff);
cmetrics = conductivity_metrics(truth_elem_data, img_rec.elem_data);
metrics.conductivity_mae = cmetrics.conductivity_mae;
metrics.conductivity_rmse = cmetrics.conductivity_rmse;
metrics.conductivity_relative_error_pct = cmetrics.conductivity_relative_error_pct;

[userView, ~] = memory;

row = struct();
row.study = 'same_geometry_cross_generation';
row.source_framework = source_framework;
row.framework = 'eidors';
row.exchange_format = exchange_format;
row.task = 'difference';
row.mesh_level = char(payload.mesh_level);
row.mesh_name = char(payload.mesh_name);
row.scenario = char(payload.scenario_name);
row.n_nodes = size(nodes, 1);
row.n_elements = size(elems, 1);
row.n_frames = 1;
row.device = 'cpu';
row.warmups = 0;
row.repeats = 1;
if isfield(cfg, 'commit')
    row.commit = char(cfg.commit);
else
    row.commit = 'unknown';
end
row.peak_rss_mb = double(userView.MemUsedMATLAB) / 1024 / 1024;
row.mean = runtime;
row.std = 0;
row.median = runtime;
row.iqr = 0;
row.mean_sec = runtime;
row.std_sec = 0;
row.median_sec = runtime;
row.iqr_sec = 0;
row.source_csv = char(cfg.input_csv);
row.imported_same_geometry = true;
if isfield(payload, 'electrode_coverage')
    row.electrode_coverage = double(payload.electrode_coverage);
else
    row.electrode_coverage = [];
end

metric_fields = fieldnames(metrics);
for i = 1:numel(metric_fields)
    row.(metric_fields{i}) = metrics.(metric_fields{i});
end

json_text = jsonencode(row, PrettyPrint=true);
fid = fopen(cfg.output_json, 'w');
fprintf(fid, '%s', json_text);
fclose(fid);
fprintf('Wrote %s\n', cfg.output_json);

if isfield(cfg, 'details_mat') && ~isempty(cfg.details_mat)
    details_path = char(cfg.details_mat);
    details_dir = fileparts(details_path);
    if exist(details_dir, 'dir') ~= 7
        mkdir(details_dir);
    end
    exchange_format = row.exchange_format;
    source_framework = row.source_framework;
    framework = row.framework;
    mesh_name = row.mesh_name;
    recon_elem_data = double(img_rec.elem_data(:));
    predicted_diff = double(pred_diff(:));
    target_diff = double(target_diff(:));
    voltage_rmse = double(row.voltage_rmse);
    conductivity_rmse = double(row.conductivity_rmse);
    save(details_path, ...
        'exchange_format', ...
        'source_framework', ...
        'framework', ...
        'mesh_name', ...
        'truth_elem_data', ...
        'recon_elem_data', ...
        'target_diff', ...
        'predicted_diff', ...
        'voltage_rmse', ...
        'conductivity_rmse');
    fprintf('Wrote %s\n', details_path);
end
end

function gnd_node = choose_ground_node(nodes, electrode_nodes, electrode_counts)
mask = false(size(nodes, 1), 1);
for i = 1:numel(electrode_counts)
    active = electrode_nodes(i, 1:electrode_counts(i));
    active = active(active > 0);
    mask(active) = true;
end

free_nodes = find(~mask);
if isempty(free_nodes)
    gnd_node = 1;
    return;
end

r2 = sum(nodes(free_nodes, :).^2, 2);
[~, idx] = min(r2);
gnd_node = free_nodes(idx);
end

function metrics = voltage_metrics(target, predicted)
target = double(target(:));
predicted = double(predicted(:));
err = predicted - target;
metrics = struct();
metrics.voltage_rmse = sqrt(mean(err.^2));
metrics.voltage_mae = mean(abs(err));
metrics.voltage_relative_error_pct = norm(err) / max(norm(target), eps) * 100;
end

function metrics = conductivity_metrics(target, predicted)
target = double(target(:));
predicted = double(predicted(:));
err = predicted - target;
metrics = struct();
metrics.conductivity_mae = mean(abs(err));
metrics.conductivity_rmse = sqrt(mean(err.^2));
metrics.conductivity_relative_error_pct = norm(err) / max(norm(target), eps) * 100;
end
