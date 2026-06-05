function complex_eidors_pyeidors_step_compare(out_root, eidors_startup_path)
% EIDORS side of the complex-admittance PyEIDORS parity harness.
%
% This function must not create its own mesh or patterns. It reads the exact
% payload exported by complex_eidors_pyeidors_step_compare.py, solves the same
% background/target complex admittivity case, and writes eidors_result.mat.

clc;
set(0, 'DefaultAxesFontName', 'Times New Roman');

if nargin < 2
    eidors_startup_path = '';
end
ensure_eidors_started(eidors_startup_path);

script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(script_dir));
if nargin < 1 || isempty(out_root)
    out_root = fullfile(default_pyeidors_output_root(), 'complex_eidors_pyeidors_step_compare');
end

case_dir = fullfile(out_root, 'complex_3d_8x2_center_sphere');
payload_path = fullfile(case_dir, 'payload.mat');
if exist(payload_path, 'file') ~= 2
    error('Missing payload.mat: %s', payload_path);
end

payload = load(payload_path);
fprintf('[EIDORS] payload: %s\n', payload_path);

fmdl = eidors_obj('fwd_model', 'complex_3d_8x2_center_sphere');
fmdl.nodes = double(payload.nodes);
fmdl.elems = double(payload.elems);
fmdl.boundary = double(payload.boundary);
fmdl.gnd_node = choose_ground_node(fmdl.nodes, double(payload.electrode_nodes), ...
    double(payload.electrode_node_counts(:)));
fmdl.solve = @fwd_solve_1st_order;
fmdl.system_mat = @system_mat_1st_order;
fmdl.jacobian = @jacobian_adjoint;
fmdl.normalize_measurements = 1;

contact_z = payload.contact_impedance(1);
electrode_nodes = double(payload.electrode_nodes);
electrode_counts = double(payload.electrode_node_counts(:));
for elec_idx = 1:size(electrode_nodes, 1)
    active = electrode_nodes(elec_idx, 1:electrode_counts(elec_idx));
    active = active(active > 0);
    fmdl.electrode(elec_idx).nodes = active;
    fmdl.electrode(elec_idx).z_contact = contact_z;
end
fmdl.stimulation = build_stimulation_from_payload(payload);
fmdl = mdl_normalize(fmdl, 1);

base_sigma = payload.base_sigma(1);
truth_elem_data = payload.truth_elem_data(:);
img_bg = mk_image(fmdl, base_sigma);
img_truth = img_bg;
img_truth.elem_data = truth_elem_data;

stage_status = struct();
stage_status.forward_ok = false;
stage_status.jacobian_ok = false;
stage_status.inverse_ok = false;
stage_status.forward_error = '';
stage_status.jacobian_error = '';
stage_status.inverse_error = '';

vh = [];
vi = [];
dv_raw_tmr = [];
dv_raw_rmt = [];
dv_norm_tmr = [];
dv_norm_rmt = [];
jacobian_projected_norm_tmr = [];
noser_prior_diag = [];
rm_matrix = [];
rec_delta = [];
rec_sigma = [];

try
    t_forward = tic;
    vh_obj = fwd_solve(img_bg);
    vi_obj = fwd_solve(img_truth);
    forward_seconds = toc(t_forward);
    vh = vh_obj.meas(:);
    vi = vi_obj.meas(:);
    dv_raw_tmr = vi - vh;
    dv_raw_rmt = vh - vi;
    dv_norm_tmr = calc_difference_data(vh, vi, fmdl);
    dv_norm_rmt = -dv_norm_tmr;
    stage_status.forward_ok = true;
catch ME
    forward_seconds = NaN;
    stage_status.forward_error = getReport(ME, 'basic', 'hyperlinks', 'off');
end

if stage_status.forward_ok
    try
        t_jac = tic;
        jacobian_projected_norm_tmr = calc_jacobian(img_bg);
        jacobian_seconds = toc(t_jac);
        stage_status.jacobian_ok = true;
    catch ME
        jacobian_seconds = NaN;
        stage_status.jacobian_error = getReport(ME, 'basic', 'hyperlinks', 'off');
        jacobian_projected_norm_tmr = [];
    end

    try
        t_inverse = tic;
        imdl = eidors_obj('inv_model', 'complex_3d_8x2_center_sphere_noser');
        imdl.fwd_model = fmdl;
        imdl.rec_model = fmdl;
        imdl.reconst_type = 'difference';
        imdl.solve = @inv_solve_diff_GN_one_step;
        imdl.RtR_prior = @prior_noser;
        imdl.hyperparameter.value = double(payload.hyperparameter);
        imdl.jacobian_bkgnd.value = base_sigma;
        imdl.inv_solve_diff_GN_one_step.calc_step_size = false;
        jacobian_for_rm = calc_jacobian(img_bg);
        W = calc_meas_icov(imdl);
        RtR = calc_RtR_prior(imdl);
        noser_prior_diag = full(diag(RtR));
        rm_matrix = left_divide((jacobian_for_rm' * W * jacobian_for_rm + ...
            double(payload.hyperparameter)^2 * RtR), jacobian_for_rm' * W);
        img_rec = inv_solve(imdl, vh_obj, vi_obj);
        inverse_seconds = toc(t_inverse);
        rec_delta = img_rec.elem_data(:);
        rec_sigma = base_sigma + rec_delta;
        stage_status.inverse_ok = true;
    catch ME
        inverse_seconds = NaN;
        stage_status.inverse_error = getReport(ME, 'basic', 'hyperlinks', 'off');
    end
else
    jacobian_seconds = NaN;
    inverse_seconds = NaN;
end

save(fullfile(case_dir, 'eidors_result.mat'), ...
    'truth_elem_data', 'vh', 'vi', 'dv_raw_tmr', 'dv_raw_rmt', ...
    'dv_norm_tmr', 'dv_norm_rmt', 'jacobian_projected_norm_tmr', ...
    'noser_prior_diag', 'rm_matrix', 'rec_delta', 'rec_sigma', ...
    'forward_seconds', 'jacobian_seconds', ...
    'inverse_seconds', 'stage_status', '-v7');

fprintf('[EIDORS] wrote %s\n', fullfile(case_dir, 'eidors_result.mat'));
end

function root = default_pyeidors_output_root()
root = strtrim(getenv('PYEIDORS_OUTPUT_ROOT'));
if ~isempty(root)
    return;
end
data_root = strtrim(getenv('PYEIDORS_DATA_ROOT'));
if ~isempty(data_root)
    root = fullfile(data_root, 'outputs');
    return;
end
xdg_data = strtrim(getenv('XDG_DATA_HOME'));
if ~isempty(xdg_data)
    root = fullfile(xdg_data, 'pyeidors', 'outputs');
    return;
end
local_appdata = strtrim(getenv('LOCALAPPDATA'));
if ~isempty(local_appdata)
    root = fullfile(local_appdata, 'pyeidors', 'outputs');
    return;
end
appdata = strtrim(getenv('APPDATA'));
if ~isempty(appdata)
    root = fullfile(appdata, 'pyeidors', 'outputs');
    return;
end
home_dir = strtrim(getenv('HOME'));
if isempty(home_dir)
    home_dir = strtrim(getenv('USERPROFILE'));
end
if isempty(home_dir)
    home_dir = pwd;
end
root = fullfile(home_dir, '.local', 'share', 'pyeidors', 'outputs');
end

function ensure_eidors_started(eidors_startup_path)
if exist('eidors_default', 'file') == 2
    return;
end
if ~isempty(eidors_startup_path) && exist(eidors_startup_path, 'file') == 2
    run(eidors_startup_path);
    return;
end
env_startup = getenv('EIDORS_STARTUP');
if ~isempty(env_startup) && exist(env_startup, 'file') == 2
    run(env_startup);
    return;
end
if exist('eidors_startup', 'file') == 2
    eidors_startup;
    return;
end
error(['EIDORS startup not found. Pass eidors_startup_path or set ', ...
       'EIDORS_STARTUP.']);
end

function stim = build_stimulation_from_payload(payload)
stim_matrix = double(payload.stim_matrix);
meas_concat = double(payload.meas_matrix_concat);
meas_start = double(payload.meas_start(:));
meas_counts = double(payload.meas_counts(:));
stim = struct('stim_pattern', {}, 'meas_pattern', {});
for idx = 1:size(stim_matrix, 1)
    start_idx = meas_start(idx);
    count = meas_counts(idx);
    rows = start_idx:(start_idx + count - 1);
    stim(idx).stim_pattern = sparse(stim_matrix(idx, :)');
    stim(idx).meas_pattern = sparse(meas_concat(rows, :));
end
end

function gnd_node = choose_ground_node(nodes, electrode_nodes, electrode_counts)
mask = false(size(nodes, 1), 1);
for idx = 1:numel(electrode_counts)
    active = electrode_nodes(idx, 1:electrode_counts(idx));
    active = active(active > 0);
    mask(active) = true;
end
free_nodes = find(~mask);
if isempty(free_nodes)
    gnd_node = 1;
    return;
end
[~, local_idx] = min(sum(nodes(free_nodes, :).^2, 2));
gnd_node = free_nodes(local_idx);
end
