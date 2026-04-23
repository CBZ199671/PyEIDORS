function manifest = capture_eidors_greit_fixture(varargin)
%CAPTURE_EIDORS_GREIT_FIXTURE Export official EIDORS GREIT parity fixtures.
%
% Usage from MATLAB after EIDORS is on the path:
%
%   capture_eidors_greit_fixture('out_dir', 'reports/eidors_greit_fixtures')
%   capture_eidors_greit_fixture('case_id', 'tiny_3d_cylinder')
%
% Optional name/value arguments:
%   out_dir          directory for *.mat fixtures and manifest.json
%   case_id          one case id, or "all" (default)
%   eidors_startup   startup.m path to run before capture
%   overwrite        true/false, default false
%
% Each fixture is saved with -v7.3 so MATLAB stores it as HDF5. Required
% root variables: vh, vi, xyzr, D, Y, PJt, M, noiselev, RM, weight.

opts = parse_capture_options(varargin{:});
if ~isempty(opts.eidors_startup)
    run(opts.eidors_startup);
end

cases = eidors_greit_fixture_cases();
selected = select_fixture_cases(cases, opts.case_id);
if isempty(selected)
    error('capture_eidors_greit_fixture:case_id', ...
        'No EIDORS GREIT fixture case matched "%s".', opts.case_id);
end

if ~exist(opts.out_dir, 'dir')
    mkdir(opts.out_dir);
end

source_map = eidors_greit_source_map();
manifest = struct();
manifest.schema = 'pyeidors-eidors-greit-fixture-manifest-v1';
manifest.generated_by = mfilename;
manifest.source_map = source_map;
manifest.fixtures = {};

for idx = 1:numel(selected)
    case_def = selected(idx);
    out_file = fullfile(opts.out_dir, [case_def.case_id '_eidors_greit_fixture.mat']);
    if exist(out_file, 'file') && ~opts.overwrite
        error('capture_eidors_greit_fixture:exists', ...
            'Fixture already exists: %s. Pass overwrite=true to replace it.', out_file);
    end

    payload = capture_one_eidors_greit_case(case_def, source_map);
    save(out_file, '-struct', 'payload', '-v7.3');

    info = struct();
    info.case_id = case_def.case_id;
    info.path = out_file;
    info.required_exports = source_map.required_exports;
    info.weight = payload.weight;
    info.n_targets = size(payload.xyzr, 2);
    info.n_measurements = numel(payload.vh);
    info.rm_shape = size(payload.RM);
    info.tolerance_status = 'unknown_until_first_official_fixture';
    manifest.fixtures{end + 1} = info; %#ok<AGROW>
end

write_json_file(fullfile(opts.out_dir, 'manifest.json'), manifest);
end


function opts = parse_capture_options(varargin)
opts = struct();
opts.out_dir = fullfile('reports', 'eidors_greit_fixtures');
opts.case_id = 'all';
opts.eidors_startup = '';
opts.overwrite = false;

if mod(numel(varargin), 2) ~= 0
    error('capture_eidors_greit_fixture:args', 'Arguments must be name/value pairs.');
end

for idx = 1:2:numel(varargin)
    name = lower(string(varargin{idx}));
    value = varargin{idx + 1};
    switch name
        case "out_dir"
            opts.out_dir = char(value);
        case "case_id"
            opts.case_id = char(value);
        case "eidors_startup"
            opts.eidors_startup = char(value);
        case "overwrite"
            opts.overwrite = logical(value);
        otherwise
            error('capture_eidors_greit_fixture:args', 'Unknown option "%s".', name);
    end
end
end


function selected = select_fixture_cases(cases, case_id)
if strcmpi(case_id, 'all')
    selected = cases;
    return;
end
mask = strcmp({cases.case_id}, case_id);
selected = cases(mask);
end


function source_map = eidors_greit_source_map()
source_map = struct();
source_map.schema = 'pyeidors-eidors-greit-source-map-v1';
source_map.required_exports = { ...
    'vh', 'vi', 'xyzr', 'D', 'Y', 'PJt', 'M', 'noiselev', 'RM', 'weight'};
source_map.official_functions = { ...
    'GREIT3D_distribution', ...
    'mk_GREIT_model', ...
    'mk_GREIT_model/stim_targets', ...
    'simulate_movement', ...
    'calc_GREIT_RM', ...
    'calc_GREIT_RM/calc_PJt', ...
    'ng_mk_cyl_models'};
source_map.urls = struct();
source_map.urls.GREIT3D_distribution = ...
    'https://eidors3d.sourceforge.net/doc/eidors/models/GREIT3D_distribution.html';
source_map.urls.mk_GREIT_model = ...
    'https://eidors3d.sourceforge.net/doc/eidors/models/mk_GREIT_model.html';
source_map.urls.simulate_movement = ...
    'https://eidors3d.sourceforge.net/doc/eidors/models/simulate_movement.html';
source_map.urls.calc_GREIT_RM = ...
    'https://eidors3d.sourceforge.net/doc/eidors/solvers/inverse/calc_GREIT_RM.html';
source_map.urls.ng_mk_cyl_models = ...
    'https://eidors3d.sourceforge.net/doc/eidors/meshing/netgen/ng_mk_cyl_models.html';
end


function cases = eidors_greit_fixture_cases()
cases = repmat(empty_case(), 1, 2);

cases(1).case_id = 'tiny_3d_cylinder';
cases(1).n_elec = 8;
cases(1).n_rings = 1;
cases(1).cyl_shape = [1.0, 1.0, 0.35];
cases(1).elec_pos = [8, 0.5];
cases(1).elec_shape = [0.12];
cases(1).background = 1.0;
cases(1).radius = 0.20;
cases(1).weight = 0.02;
cases(1).normalize = 1;
cases(1).vopt = struct( ...
    'imgsz', [5, 5], ...
    'zvec', [0.15, 0.35, 0.55, 0.75], ...
    'downsample', [2, 0]);

cases(2).case_id = 'reduced_48e_5936';
cases(2).n_elec = 48;
cases(2).n_rings = 3;
cases(2).cyl_shape = [0.16, 0.18, 0.055];
cases(2).elec_pos = three_ring_electrode_positions(16, [0.15, 0.50, 0.85]);
cases(2).elec_shape = [0.012, 0, 0.006];
cases(2).background = 1.0;
cases(2).radius = 0.035;
cases(2).weight = 0.02;
cases(2).normalize = 1;
cases(2).vopt = struct( ...
    'xvec', [-0.16, -0.08, 0.0, 0.08, 0.16], ...
    'yvec', [-0.16, -0.08, 0.0, 0.08, 0.16], ...
    'zvec', [0.02, 0.06, 0.10, 0.14], ...
    'downsample', [2, 0]);
end


function case_def = empty_case()
case_def = struct();
case_def.case_id = '';
case_def.n_elec = 0;
case_def.n_rings = 0;
case_def.cyl_shape = [];
case_def.elec_pos = [];
case_def.elec_shape = [];
case_def.background = 1.0;
case_def.radius = 0.0;
case_def.weight = 0.0;
case_def.normalize = 1;
case_def.vopt = struct();
end


function elec_pos = three_ring_electrode_positions(per_ring, z_levels)
elec_pos = zeros(numel(z_levels) * per_ring, 2);
row = 1;
for zi = 1:numel(z_levels)
    for ei = 1:per_ring
        elec_pos(row, :) = [360 * (ei - 1) / per_ring, z_levels(zi)];
        row = row + 1;
    end
end
end


function payload = capture_one_eidors_greit_case(case_def, source_map)
assert_required_eidors_functions();

[fmdl, mat_idx] = ng_mk_cyl_models( ...
    case_def.cyl_shape, case_def.elec_pos, case_def.elec_shape);
fmdl.name = ['PyEIDORS parity ' case_def.case_id];
fmdl.stimulation = mk_stim_patterns( ...
    case_def.n_elec, case_def.n_rings, '{ad}', '{ad}', {'no_meas_current'}, 1);
fmdl = mdl_normalize(fmdl, case_def.normalize);

homogeneous = mk_image(fmdl, case_def.background);
[imdl, distr] = GREIT3D_distribution(fmdl, case_def.vopt);
imdl.name = ['PyEIDORS GREIT parity ' case_def.case_id];
opt = case_def.vopt;
opt.distr = distr;
opt.rec_model = imdl.rec_model;
opt.keep_model_components = true;
opt.normalize = case_def.normalize;
opt.noise_covar = 1;

[vi, vh, xyzr] = capture_finite_target_responses(homogeneous, opt, case_def.radius);
xyz = xyzr(1:3, :);
[RM, PJt, M, noiselev] = calc_GREIT_RM( ...
    vh, vi, xyz, case_def.radius, case_def.weight, opt);
[D, Y] = capture_desired_and_response_matrices( ...
    vh, vi, xyz, case_def.radius, opt);

[official_imdl, official_weight] = mk_GREIT_model( ...
    imdl, case_def.radius, case_def.weight, opt);

payload = struct();
payload.schema = 'pyeidors-eidors-greit-fixture-v1';
payload.case_id = case_def.case_id;
payload.source_map = source_map;
payload.case_definition = case_def;
payload.mat_idx = mat_idx;
payload.vh = vh;
payload.vi = vi;
payload.xyzr = xyzr;
payload.D = D;
payload.Y = Y;
payload.PJt = PJt;
payload.M = M;
payload.noiselev = noiselev;
payload.RM = RM;
payload.weight = official_weight;
payload.requested_weight = case_def.weight;
payload.official_RM = official_imdl.solve_use_matrix.RM;
payload.official_PJt = official_imdl.solve_use_matrix.PJt;
payload.official_M = official_imdl.solve_use_matrix.M;
payload.official_noiselev = official_imdl.solve_use_matrix.noiselev;
payload.diagnostics = capture_diagnostics(payload);
end


function assert_required_eidors_functions()
names = {'ng_mk_cyl_models', 'mk_stim_patterns', 'mdl_normalize', ...
    'mk_image', 'GREIT3D_distribution', 'calc_GREIT_RM', ...
    'mk_GREIT_model', 'simulate_movement'};
missing = {};
for idx = 1:numel(names)
    if exist(names{idx}, 'file') ~= 2
        missing{end + 1} = names{idx}; %#ok<AGROW>
    end
end
if ~isempty(missing)
    error('capture_eidors_greit_fixture:missing_eidors', ...
        'Missing EIDORS functions on MATLAB path: %s', strjoin(missing, ', '));
end
end


function [vi, vh, xyzr] = capture_finite_target_responses(imgs, opt, radius)
xyzr = opt.distr;
if size(xyzr, 1) < 3
    error('capture_eidors_greit_fixture:distr', ...
        'GREIT3D_distribution returned invalid target distribution.');
end
if size(xyzr, 1) == 3
    xyzr(4, :) = radius;
else
    xyzr(4, :) = radius;
end
[vh, vi, xyzr] = simulate_movement(imgs, xyzr);
end


function [D, Y] = capture_desired_and_response_matrices(vh, vi, xyz, radius, opt)
if isfield(opt, 'normalize') && opt.normalize
    Y = calc_difference_data(vi, vh, 'ratio');
else
    Y = calc_difference_data(vi, vh);
end
if isfield(opt, 'desired_solution_fn')
    fn = opt.desired_solution_fn;
else
    fn = eidors_default('get', 'calc_GREIT_RM_desired_img');
end
D = feval(fn, xyz, radius, opt);
end


function diagnostics = capture_diagnostics(payload)
diagnostics = struct();
diagnostics.required_exports = payload.source_map.required_exports;
diagnostics.rm_shape = size(payload.RM);
diagnostics.y_shape = size(payload.Y);
diagnostics.d_shape = size(payload.D);
diagnostics.pjt_shape = size(payload.PJt);
diagnostics.m_shape = size(payload.M);
diagnostics.n_targets = size(payload.xyzr, 2);
diagnostics.n_measurements = numel(payload.vh);
diagnostics.official_rm_relative_error = relative_matrix_error( ...
    payload.RM, payload.official_RM);
diagnostics.official_pjt_relative_error = relative_matrix_error( ...
    payload.PJt, payload.official_PJt);
diagnostics.tolerance_status = 'unknown_until_first_official_fixture';
end


function err = relative_matrix_error(a, b)
if isempty(a) || isempty(b) || any(size(a) ~= size(b))
    err = inf;
    return;
end
denom = max(1.0, norm(b(:), inf));
err = norm(a(:) - b(:), inf) / denom;
end


function write_json_file(path, payload)
fid = fopen(path, 'w');
if fid < 0
    error('capture_eidors_greit_fixture:manifest', ...
        'Unable to write manifest: %s', path);
end
cleaner = onCleanup(@() fclose(fid));
fprintf(fid, '%s', jsonencode(payload, 'PrettyPrint', true));
clear cleaner;
end
