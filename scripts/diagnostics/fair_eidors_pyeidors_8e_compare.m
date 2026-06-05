function fair_eidors_pyeidors_8e_compare(out_root)
% Fair 8-electrode EIDORS side of the PyEIDORS comparison.

clc;
set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultTextFontName', 'Microsoft YaHei');
set(0, 'DefaultTextInterpreter', 'none');

eidorsStartup = 'D:\Program Files\MATLAB\R2023b\toolbox\eidors-v3.12-ng\eidors\startup.m';
if exist('eidors_default', 'file') ~= 2
    if exist('eidors_startup', 'file') == 2
        eidors_startup;
    elseif exist(eidorsStartup, 'file') == 2
        run(eidorsStartup);
    else
        error('EIDORS startup script not found. Please update eidorsStartup.');
    end
end

script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(script_dir));
if nargin < 1 || isempty(out_root)
    out_root = fullfile(default_pyeidors_output_root(), 'eidors_fair_8e_layers');
end

cases = {'2d_8e', '3d_8x2', '3d_8x3'};
for idx = 1:numel(cases)
    run_case(fullfile(out_root, cases{idx}));
end
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

function run_case(case_dir)
payload_path = fullfile(case_dir, 'payload.mat');
if exist(payload_path, 'file') ~= 2
    error('Missing payload: %s', payload_path);
end
payload = load(payload_path);
case_name = char(string(payload.case_name));
dim = double(payload.dim);
fprintf('[EIDORS] %s: loading %s\n', case_name, payload_path);

fmdl = eidors_obj('fwd_model', ['fair_' case_name]);
fmdl.nodes = double(payload.nodes);
fmdl.elems = double(payload.elems);
fmdl.boundary = double(payload.boundary);
fmdl.gnd_node = choose_ground_node(fmdl.nodes, double(payload.electrode_nodes), ...
    double(payload.electrode_node_counts(:)));
fmdl.solve = @fwd_solve_1st_order;
fmdl.system_mat = @system_mat_1st_order;
fmdl.jacobian = @jacobian_adjoint;
fmdl.normalize_measurements = 1;

electrode_nodes = double(payload.electrode_nodes);
electrode_counts = double(payload.electrode_node_counts(:));
for elec_idx = 1:size(electrode_nodes, 1)
    active = electrode_nodes(elec_idx, 1:electrode_counts(elec_idx));
    fmdl.electrode(elec_idx).nodes = active(active > 0);
    fmdl.electrode(elec_idx).z_contact = double(payload.contact_impedance);
end
fmdl.stimulation = build_stimulation_from_payload(payload);
fmdl = mdl_normalize(fmdl, 1);

base_sigma = double(payload.base_sigma);
img_bg = mk_image(fmdl, base_sigma);
img_truth = img_bg;
img_truth.elem_data = double(payload.truth_elem_data(:));

t_forward = tic;
vh = fwd_solve(img_bg);
vi = fwd_solve(img_truth);
forward_seconds = toc(t_forward);

imdl = eidors_obj('inv_model', ['fair_' case_name '_noser']);
imdl.fwd_model = fmdl;
imdl.rec_model = fmdl;
imdl.reconst_type = 'difference';
imdl.solve = @inv_solve_diff_GN_one_step;
imdl.RtR_prior = @prior_noser;
imdl.hyperparameter.value = double(payload.hyperparameter);
imdl.jacobian_bkgnd.value = base_sigma;
imdl.inv_solve_diff_GN_one_step.calc_step_size = false;

t_inverse = tic;
img_rec = inv_solve(imdl, vh, vi);
inverse_seconds = toc(t_inverse);

rec_delta = real(img_rec.elem_data(:));
rec_sigma = base_sigma + rec_delta;
img_pred = img_bg;
img_pred.elem_data = max(rec_sigma, 1e-6);
pred_vi = fwd_solve(img_pred);

dv_meas = real(calc_difference_data(vh.meas, vi.meas, fmdl));
dv_pred = real(calc_difference_data(vh.meas, pred_vi.meas, fmdl));
truth_delta = img_truth.elem_data(:) - base_sigma;
cond_rel_l2 = norm(rec_delta - truth_delta) / max(norm(truth_delta), eps);
cond_corr = safe_corr(truth_delta, rec_delta);
fit_rmse = sqrt(mean((dv_meas(:) - dv_pred(:)).^2));
fit_corr = safe_corr(dv_meas(:), dv_pred(:));

writematrix(real(vh.meas(:)), fullfile(case_dir, 'eidors_vh_background.csv'));
writematrix(real(vi.meas(:)), fullfile(case_dir, 'eidors_vi_target.csv'));
writematrix(dv_meas(:), fullfile(case_dir, 'eidors_dv_measured_normalized.csv'));
writematrix(dv_pred(:), fullfile(case_dir, 'eidors_dv_predicted_normalized.csv'));
writematrix(rec_delta(:), fullfile(case_dir, 'eidors_recon_delta_sigma.csv'));
writematrix(rec_sigma(:), fullfile(case_dir, 'eidors_recon_sigma.csv'));

save(fullfile(case_dir, 'eidors_result.mat'), ...
    'case_name', 'dim', 'rec_delta', 'rec_sigma', 'dv_meas', 'dv_pred', ...
    'forward_seconds', 'inverse_seconds', 'cond_rel_l2', 'cond_corr', ...
    'fit_rmse', 'fit_corr', '-v7');

fprintf(['[EIDORS] %s: dim=%d forward=%.3fs inverse=%.3fs ', ...
    'cond_corr=%.4f fit_corr=%.4f meas=%d\n'], ...
    case_name, dim, forward_seconds, inverse_seconds, cond_corr, fit_corr, numel(dv_meas));
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
    stim(idx).stim_pattern = sparse(stim_matrix(idx, :)'); %#ok<AGROW>
    stim(idx).meas_pattern = sparse(meas_concat(rows, :)); %#ok<AGROW>
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

function rho = safe_corr(a, b)
a = real(a(:));
b = real(b(:));
mask = isfinite(a) & isfinite(b);
if nnz(mask) < 3 || std(a(mask)) <= eps || std(b(mask)) <= eps
    rho = NaN;
    return;
end
c = corrcoef(a(mask), b(mask));
rho = c(1, 2);
end
