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
if isfield(cfg, 'work_dir') && exist(char(cfg.work_dir), 'dir') == 7
    addpath(script_dir);
    cd(char(cfg.work_dir));
else
    cd(script_dir);
end
run(target_script);

catalog = local_discover_workspace();
[img_bg, img_bg_source, img_bg_selection] = local_select_optional_role( ...
    catalog.images, cfg, 'background_image_var', ...
    {'img_bg', 'img_bkgnd', 'img_background', 'img_homogeneous', ...
     'background_image'}, {}, false);
[img_target, img_target_source, img_target_selection] = local_select_optional_role( ...
    catalog.images, cfg, 'target_image_var', ...
    {'img_truth', 'img_target', 'img_phantom', 'target_image', 'img'}, ...
    {img_bg_source}, true);
[fmdl, fmdl_source, fmdl_selection] = local_select_model( ...
    catalog.models, cfg, {img_bg, img_target});
[vh, vh_source, vh_selection] = local_select_optional_role( ...
    catalog.data, cfg, 'homogeneous_data_var', ...
    {'vh', 'vhom', 'data_homogeneous', 'data_background'}, {}, false);
[vi, vi_source, vi_selection] = local_select_optional_role( ...
    catalog.data, cfg, 'target_data_var', ...
    {'vi', 'vtarget', 'data_target', 'data_phantom', 'v'}, ...
    {vh_source}, false);

out_dir = char(cfg.output_dir);
if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end

exchange_format = 'eidors_pyeidors_geometry_v3';
schema_version = 3;
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
boundary_edges = boundary_facets; % MATLAB compatibility alias
runtime = fwd_model_parameters(fmdl, 'skip_VOLUME');
N2E = full(runtime.N2E);
QQ = local_runtime_matrix(runtime, 'QQ', size(nodes, 1), 0);
VV = local_runtime_matrix(runtime, 'VV', size(nodes, 1), 0);
v2meas = local_runtime_matrix(runtime, 'v2meas', 0, size(nodes, 1));
runtime_boundary = double(runtime.boundary);
runtime_normalize = logical(runtime.normalize);
CEM_boundary = zeros(0, dimension);
if isfield(fmdl, 'system_mat_fields') && ...
        isfield(fmdl.system_mat_fields, 'CEM_boundary')
    CEM_boundary = double(fmdl.system_mat_fields.CEM_boundary);
end
coarse2fine = zeros(size(elems, 1), 0);
coarse2fine_present = isfield(fmdl, 'coarse2fine') && ...
    ~isempty(fmdl.coarse2fine);
if coarse2fine_present
    coarse2fine = full(fmdl.coarse2fine);
end
meas_select = zeros(0, 1);
meas_select_present = isfield(fmdl, 'meas_select') && ...
    ~isempty(fmdl.meas_select);
if meas_select_present
    meas_select = full(fmdl.meas_select);
end
[electrode_nodes, electrode_node_counts, electrode_faces, ...
 electrode_face_counts, electrode_model, contact_impedance, ...
 contact_impedance_present, electrode_projection_required, ...
 pem_node_weights, electrode_boundary_kind, cem_face_nodes, ...
 cem_face_node_counts, cem_face_electrode] = ...
    local_build_electrode_arrays( ...
        fmdl, boundary_facets, runtime_boundary, N2E, dimension);
[stim_matrix_raw, stim_matrix, meas_matrices, measurement_counts, ...
 volt_matrix, volt_pattern_present, interior_sources, ...
 interior_source_counts, stimulation_labels, current_density, ...
 current_density_present, current_density_applied, ...
 stim_positive_current, stim_negative_current, stim_net_current, ...
 stim_max_abs_current, stim_balanced, stimulation_supported] = ...
    local_build_pattern_arrays(fmdl);
n_elec = double(numel(fmdl.electrode));
[normalize_measurements, normalize_measurements_present, ...
 normalize_measurements_source] = local_normalize_measurements(fmdl);
if normalize_measurements ~= runtime_normalize
    error('EIDORS runtime normalize disagrees with mdl_normalize.');
end
[gnd_node, gnd_node_present, effective_gnd_node, ...
 effective_gnd_node_source] = local_ground_node(fmdl);
background_image = local_resolve_image(img_bg, fmdl, img_bg_source, ...
    img_bg_selection, 'background');
target_image = local_resolve_image(img_target, fmdl, img_target_source, ...
    img_target_selection, 'target');
background_present = background_image.scalar_present;
background = background_image.scalar_value;
background_elem_data_present = background_image.present;
background_elem_data = background_image.elem_data;
truth_elem_data_present = target_image.present;
truth_elem_data = target_image.elem_data;
target_elem_data = target_image.elem_data;
mesh_name = local_mesh_name(fmdl, target_script);
mesh_level = 'script_capture';
scenario_name = local_script_kind(cfg);
[model_valid, model_validation_error] = local_validate_fwd_model(fmdl);
model_solver = local_function_name(local_field_or_empty(fmdl, 'solve'));
effective_model_solver = local_effective_function_name(model_solver, 'fwd_solve');
model_system_mat = local_function_name(local_field_or_empty(fmdl, 'system_mat'));
effective_model_system_mat = local_effective_function_name( ...
    model_system_mat, 'calc_system_mat');
model_jacobian = local_function_name(local_field_or_empty(fmdl, 'jacobian'));
model_measured_quantity = local_text_field(fmdl, 'measured_quantity', ...
    'unspecified');
model_coordinate_units = local_text_field(fmdl, 'units', 'unspecified');
contact_impedance_unit = local_contact_impedance_unit(dimension, ...
    model_coordinate_units);
[forward_blockers, forward_warnings] = local_forward_readiness( ...
    model_valid, model_validation_error, contact_impedance_present, ...
    electrode_model, electrode_projection_required, background_image, target_image, ...
    stimulation_supported, effective_model_solver, gnd_node_present, ...
    effective_gnd_node_source);

capture_metadata = struct();
capture_metadata.schema = 'eidors_pyeidors_capture_semantics_v3';
capture_metadata.eidors_version = eidors_obj('eidors_version');
capture_metadata.selected.fwd_model = local_selection_record( ...
    fmdl_source, fmdl_selection);
capture_metadata.selected.background_image = local_selection_record( ...
    img_bg_source, img_bg_selection);
capture_metadata.selected.target_image = local_selection_record( ...
    img_target_source, img_target_selection);
capture_metadata.selected.homogeneous_data = local_selection_record( ...
    vh_source, vh_selection);
capture_metadata.selected.target_data = local_selection_record( ...
    vi_source, vi_selection);
capture_metadata.model.valid = model_valid;
capture_metadata.model.validation_error = model_validation_error;
capture_metadata.model.solver_declared = model_solver;
capture_metadata.model.solver_effective = effective_model_solver;
capture_metadata.model.system_mat_declared = model_system_mat;
capture_metadata.model.system_mat_effective = effective_model_system_mat;
capture_metadata.model.jacobian_declared = model_jacobian;
capture_metadata.model.measured_quantity = model_measured_quantity;
capture_metadata.model.coordinate_units = model_coordinate_units;
capture_metadata.model.contact_impedance_unit = contact_impedance_unit;
capture_metadata.model.coarse2fine = local_field_shape_record( ...
    fmdl, 'coarse2fine');
capture_metadata.model.background = local_field_shape_record( ...
    fmdl, 'background');
capture_metadata.model.model_reduction = local_field_shape_record( ...
    fmdl, 'model_reduction');
capture_metadata.fields.contact_impedance = local_presence_record( ...
    contact_impedance_present, 'electrode(i).z_contact', ...
    'EIDORS valid_fwd_model requires the field; no solver default exists.');
capture_metadata.fields.normalize_measurements = local_runtime_record( ...
    normalize_measurements_present, normalize_measurements_source);
capture_metadata.fields.gnd_node = local_runtime_record( ...
    gnd_node_present, effective_gnd_node_source);
capture_metadata.fields.stimulation = local_presence_record( ...
    ~isempty(stim_matrix_raw), 'fwd_model.stimulation', ...
    'Raw and EIDORS-effective patterns are stored separately.');
capture_metadata.runtime_operators.N2E = size(N2E);
capture_metadata.runtime_operators.QQ = size(QQ);
capture_metadata.runtime_operators.VV = size(VV);
capture_metadata.runtime_operators.v2meas = size(v2meas);
capture_metadata.runtime_operators.CEM_boundary = size(CEM_boundary);
capture_metadata.runtime_operators.meas_select_present = meas_select_present;
capture_metadata.runtime_operators.coarse2fine_present = coarse2fine_present;
capture_metadata.fields.background_image = local_image_record(background_image);
capture_metadata.fields.target_image = local_image_record(target_image);
capture_metadata.electrode_models = electrode_model;
capture_metadata.current_density.present = current_density_present;
capture_metadata.current_density.value = current_density;
capture_metadata.current_density.applied = current_density_applied;
capture_metadata.forward_ready = isempty(forward_blockers);
capture_metadata.forward_blockers = forward_blockers;
capture_metadata.forward_warnings = forward_warnings;
capture_metadata_json = jsonencode(capture_metadata);

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
    'electrode_faces', ...
    'electrode_face_counts', ...
    'electrode_model', ...
    'electrode_boundary_kind', ...
    'pem_node_weights', ...
    'cem_face_nodes', ...
    'cem_face_node_counts', ...
    'cem_face_electrode', ...
    'electrode_projection_required', ...
    'CEM_boundary', ...
    'runtime_boundary', ...
    'stim_matrix_raw', ...
    'stim_matrix', ...
    'meas_matrices', ...
    'measurement_counts', ...
    'N2E', ...
    'QQ', ...
    'VV', ...
    'v2meas', ...
    'meas_select', ...
    'meas_select_present', ...
    'volt_matrix', ...
    'volt_pattern_present', ...
    'interior_sources', ...
    'interior_source_counts', ...
    'stimulation_labels', ...
    'current_density', ...
    'current_density_present', ...
    'current_density_applied', ...
    'stim_positive_current', ...
    'stim_negative_current', ...
    'stim_net_current', ...
    'stim_max_abs_current', ...
    'stim_balanced', ...
    'stimulation_supported', ...
    'n_elec', ...
    'normalize_measurements', ...
    'normalize_measurements_present', ...
    'normalize_measurements_source', ...
    'gnd_node', ...
    'gnd_node_present', ...
    'effective_gnd_node', ...
    'effective_gnd_node_source', ...
    'background', ...
    'background_present', ...
    'background_elem_data', ...
    'background_elem_data_present', ...
    'truth_elem_data', ...
    'truth_elem_data_present', ...
    'target_elem_data', ...
    'coarse2fine', ...
    'coarse2fine_present', ...
    'contact_impedance', ...
    'contact_impedance_present', ...
    'contact_impedance_unit', ...
    'model_solver', ...
    'effective_model_solver', ...
    'model_system_mat', ...
    'effective_model_system_mat', ...
    'model_jacobian', ...
    'model_measured_quantity', ...
    'model_coordinate_units', ...
    'model_valid', ...
    'model_validation_error', ...
    'forward_blockers', ...
    'forward_warnings', ...
    'capture_metadata_json', ...
    'mesh_name', ...
    'mesh_level', ...
    'scenario_name');

if ~isempty(vh) && ~isempty(vi)
    vh_meas = double(vh.meas(:));
    vi_meas = double(vi.meas(:));
    difference = vi_meas - vh_meas;
    if isreal(vh_meas) && isreal(vi_meas)
        T = table(vh_meas, vi_meas, difference, 'VariableNames', ...
            {'meas_homogeneous', 'meas_phantom', 'difference'});
        writetable(T, fullfile(out_dir, 'measurements.csv'));
    else
        homogeneous = vh_meas;
        target = vi_meas;
        save(fullfile(out_dir, 'measurements.mat'), ...
            'homogeneous', 'target', 'difference');
    end
end

raw_vars = whos;
capture_report = struct();
capture_report.script_path = target_script;
capture_report.script_kind = local_script_kind(cfg);
capture_report.fmdl_found = true;
capture_report.fmdl_source = fmdl_source;
capture_report.model_candidates = local_candidate_paths(catalog.models);
capture_report.image_candidates = local_candidate_paths(catalog.images);
capture_report.data_candidates = local_candidate_paths(catalog.data);
capture_report.background_image_source = img_bg_source;
capture_report.target_image_source = img_target_source;
capture_report.vh_found = ~isempty(vh);
capture_report.vi_found = ~isempty(vi);
capture_report.vh_source = vh_source;
capture_report.vi_source = vi_source;
capture_report.forward_ready = isempty(forward_blockers);
capture_report.forward_blockers = forward_blockers;
capture_report.forward_warnings = forward_warnings;
capture_report.workspace_vars = {raw_vars.name};
json_text = jsonencode(capture_report, PrettyPrint=true);
fid = fopen(fullfile(out_dir, 'capture_report.json'), 'w');
fprintf(fid, '%s', json_text);
fclose(fid);
end

function catalog = local_discover_workspace()
catalog.models = struct('path', {}, 'aliases', {}, 'value', {});
catalog.images = struct('path', {}, 'aliases', {}, 'value', {});
catalog.data = struct('path', {}, 'aliases', {}, 'value', {});
vars = evalin('caller', 'whos');
for idx = 1:numel(vars)
    name = vars(idx).name;
    value = evalin('caller', name);
    if ~isstruct(value) || numel(value) ~= 1
        continue;
    end
    object_type = local_object_type(value);
    if strcmp(object_type, 'fwd_model') || ...
            (isfield(value, 'nodes') && isfield(value, 'elems'))
        catalog.models = local_append_candidate(catalog.models, name, value);
    end
    if strcmp(object_type, 'image') || local_has_image_data(value)
        catalog.images = local_append_candidate(catalog.images, name, value);
    end
    if strcmp(object_type, 'data') || isfield(value, 'meas')
        catalog.data = local_append_candidate(catalog.data, name, value);
    end
    if isfield(value, 'fwd_model') && isstruct(value.fwd_model) && ...
            numel(value.fwd_model) == 1
        nested_path = [name, '.fwd_model'];
        catalog.models = local_append_candidate( ...
            catalog.models, nested_path, value.fwd_model);
    end
end
end

function object_type = local_object_type(value)
object_type = '';
if isstruct(value) && isfield(value, 'type') && ...
        (ischar(value.type) || isstring(value.type))
    object_type = char(value.type);
end
end

function tf = local_has_image_data(value)
tf = isfield(value, 'fwd_model') && any(isfield(value, ...
    {'elem_data', 'node_data', 'conductivity', 'resistivity', ...
     'log_conductivity', 'log10_conductivity', ...
     'log_resistivity', 'log10_resistivity'}));
end

function candidates = local_append_candidate(candidates, path, value)
for idx = 1:numel(candidates)
    if isequaln(candidates(idx).value, value)
        candidates(idx).aliases{end + 1} = path;
        return;
    end
end
record.path = path;
record.aliases = {path};
record.value = value;
candidates(end + 1) = record;
end

function [value, path, method] = local_select_required( ...
        candidates, cfg, selector_field, label)
[value, path, method] = local_select_by_selector( ...
    candidates, cfg, selector_field, label);
if ~isempty(value)
    return;
end
if isempty(candidates)
    error('No standard EIDORS %s object was discovered.', label);
end
if numel(candidates) > 1
    error(['Multiple EIDORS %s objects were discovered: %s. ', ...
           'Select one with %s.'], label, ...
          strjoin(local_candidate_paths(candidates), ', '), selector_field);
end
value = candidates(1).value;
path = candidates(1).path;
method = 'unique_standard_object';
end

function [value, path, method] = local_select_model(candidates, cfg, images)
[value, path, method] = local_select_by_selector( ...
    candidates, cfg, 'fwd_model_var', 'forward model');
if ~isempty(value)
    return;
end
related = [];
for image_idx = 1:numel(images)
    image = images{image_idx};
    if isempty(image) || ~isstruct(image) || ~isfield(image, 'fwd_model')
        continue;
    end
    for candidate_idx = 1:numel(candidates)
        if isequaln(candidates(candidate_idx).value, image.fwd_model)
            related(end + 1) = candidate_idx; %#ok<AGROW>
        end
    end
end
related = unique(related);
if numel(related) == 1
    value = candidates(related).value;
    path = candidates(related).path;
    method = 'referenced_by_selected_image';
    return;
elseif numel(related) > 1
    error(['Selected EIDORS images reference different forward models: %s. ', ...
           'Use fwd_model_var to choose explicitly.'], ...
          strjoin(local_candidate_paths(candidates(related)), ', '));
end
[value, path, method] = local_select_required( ...
    candidates, cfg, 'fwd_model_var', 'forward model');
end

function [value, path, method] = local_select_optional_role( ...
        candidates, cfg, selector_field, preferred_names, excluded_paths, ...
        use_single_fallback)
[value, path, method] = local_select_by_selector( ...
    candidates, cfg, selector_field, selector_field);
if ~isempty(value)
    return;
end
matches = [];
for idx = 1:numel(candidates)
    if any(strcmp(candidates(idx).path, excluded_paths))
        continue;
    end
    aliases = lower(string(candidates(idx).aliases));
    if any(ismember(aliases, lower(string(preferred_names))))
        matches(end + 1) = idx; %#ok<AGROW>
    end
end
if numel(matches) > 1
    error(['Multiple candidates match inferred role %s: %s. ', ...
           'Use an explicit selector.'], selector_field, ...
          strjoin(local_candidate_paths(candidates(matches)), ', '));
elseif numel(matches) == 1
    value = candidates(matches).value;
    path = candidates(matches).path;
    method = 'inferred_from_conventional_variable_name';
    return;
end
available = [];
for idx = 1:numel(candidates)
    if ~any(strcmp(candidates(idx).path, excluded_paths))
        available(end + 1) = idx; %#ok<AGROW>
    end
end
if use_single_fallback && numel(available) == 1
    value = candidates(available).value;
    path = candidates(available).path;
    method = 'inferred_from_single_unassigned_candidate';
else
    value = [];
    path = '';
    method = 'missing';
end
end

function [value, path, method] = local_select_by_selector( ...
        candidates, cfg, selector_field, label)
value = [];
path = '';
method = '';
if ~isfield(cfg, selector_field) || isempty(cfg.(selector_field))
    return;
end
selector = char(cfg.(selector_field));
for idx = 1:numel(candidates)
    if any(strcmp(selector, candidates(idx).aliases))
        value = candidates(idx).value;
        path = candidates(idx).path;
        method = 'explicit_selector';
        return;
    end
end
error('Requested %s selector "%s" was not discovered. Candidates: %s', ...
    label, selector, strjoin(local_candidate_paths(candidates), ', '));
end

function paths = local_candidate_paths(candidates)
paths = cell(1, numel(candidates));
for idx = 1:numel(candidates)
    paths{idx} = strjoin(candidates(idx).aliases, '|');
end
end

function [electrode_nodes, electrode_node_counts, electrode_faces, ...
          electrode_face_counts, electrode_model, contact_impedance, ...
          contact_impedance_present, projection_required, ...
          pem_node_weights, electrode_boundary_kind, cem_face_nodes, ...
          cem_face_node_counts, cem_face_electrode] = ...
          local_build_electrode_arrays( ...
              fmdl, boundary_facets, runtime_boundary, N2E, dimension)
n_elec = numel(fmdl.electrode);
n_node = size(fmdl.nodes, 1);
electrode_node_counts = zeros(n_elec, 1);
max_nodes = 0;
max_faces = 0;
node_lists = cell(n_elec, 1);
weight_lists = cell(n_elec, 1);
face_lists = cell(n_elec, 1);
electrode_model = cell(n_elec, 1);
electrode_boundary_kind = cell(n_elec, 1);
contact_impedance = NaN(n_elec, 1);
contact_impedance_present = false(n_elec, 1);
projection_required = false(n_elec, 1);
for i = 1:n_elec
    elec = fmdl.electrode(i);
    if ~isfield(elec, 'nodes') || ischar(elec.nodes)
        error(['Electrode %d does not expose numeric mesh nodes. ', ...
               'Instrument electrodes are audit-only in Bridge v3.'], i);
    end
    runtime_row = full(double(N2E(i, :)));
    runtime_columns = find(runtime_row ~= 0);
    if isempty(runtime_columns)
        error('Electrode %d has no active N2E entries.', i);
    end
    nodes = double(elec.nodes(:)');
    weights = zeros(1, 0);
    faces = zeros(0, dimension);
    if any(runtime_columns > n_node)
        if isfield(elec, 'faces') && ~isempty(elec.faces)
            faces = double(elec.faces);
            electrode_model{i} = 'cem_faces';
        else
            complete = all(ismember(runtime_boundary, nodes), 2);
            faces = runtime_boundary(complete, :);
            electrode_model{i} = 'cem';
        end
        if isempty(faces) || size(faces, 2) ~= dimension
            error('CEM electrode %d has no exact width-%d faces.', i, dimension);
        end
        nodes = unique([nodes(:); faces(:)])';
        if local_all_faces_in_set(faces, boundary_facets)
            electrode_boundary_kind{i} = 'exterior';
        else
            electrode_boundary_kind{i} = 'interior';
        end
    else
        nodes = runtime_columns;
        weights = runtime_row(runtime_columns);
        if numel(nodes) == 1
            electrode_model{i} = 'point';
        else
            electrode_model{i} = 'distributed_point';
        end
        electrode_boundary_kind{i} = 'none';
        tolerance = 1e-12 * max(1, max(abs(weights)));
        if abs(sum(weights) - 1) > tolerance
            error('PEM electrode %d N2E weights do not sum to one.', i);
        end
    end
    node_lists{i} = nodes;
    weight_lists{i} = weights;
    face_lists{i} = faces;
    electrode_node_counts(i) = numel(nodes);
    electrode_face_counts(i, 1) = size(faces, 1); %#ok<AGROW>
    max_nodes = max(max_nodes, numel(nodes));
    max_faces = max(max_faces, size(faces, 1));
    if isfield(elec, 'z_contact') && isnumeric(elec.z_contact) && ...
            isscalar(elec.z_contact)
        contact_impedance(i) = double(elec.z_contact);
        contact_impedance_present(i) = true;
    end
end
electrode_nodes = zeros(n_elec, max_nodes);
pem_node_weights = zeros(n_elec, max_nodes);
electrode_faces = zeros(n_elec, max_faces, dimension);
total_cem_faces = sum(electrode_face_counts);
cem_face_nodes = zeros(total_cem_faces, dimension);
cem_face_node_counts = dimension * ones(total_cem_faces, 1);
cem_face_electrode = zeros(total_cem_faces, 1);
face_cursor = 0;
for i = 1:n_elec
    nodes = node_lists{i};
    electrode_nodes(i, 1:numel(nodes)) = nodes;
    weights = weight_lists{i};
    if ~isempty(weights)
        pem_node_weights(i, 1:numel(weights)) = weights;
    end
    faces = face_lists{i};
    electrode_faces(i, 1:size(faces, 1), :) = ...
        reshape(faces, 1, size(faces, 1), dimension);
    if ~isempty(faces)
        rows = face_cursor + (1:size(faces, 1));
        cem_face_nodes(rows, :) = faces;
        cem_face_electrode(rows) = i;
        face_cursor = face_cursor + size(faces, 1);
    end
end
end

function result = local_all_faces_in_set(faces, reference)
if isempty(faces)
    result = false;
    return;
end
sorted_reference = sort(reference, 2);
sorted_faces = sort(faces, 2);
result = all(ismember(sorted_faces, sorted_reference, 'rows'));
end

function value = local_runtime_matrix(runtime, field_name, n_rows, n_cols)
if isfield(runtime, field_name) && ~isempty(runtime.(field_name))
    value = full(runtime.(field_name));
else
    value = zeros(n_rows, n_cols);
end
end

function [stim_matrix_raw, stim_matrix, meas_matrices, measurement_counts, ...
          volt_matrix, volt_pattern_present, interior_sources, ...
          interior_source_counts, stimulation_labels, current_density, ...
          current_density_present, current_density_applied, ...
          positive_current, negative_current, net_current, ...
          max_abs_current, balanced, supported] = ...
          local_build_pattern_arrays(fmdl)
n_elec = numel(fmdl.electrode);
current_density = NaN;
current_density_present = isfield(fmdl, 'current_density') && ...
    ~isempty(fmdl.current_density);
current_density_applied = false;
if current_density_present && isnumeric(fmdl.current_density) && ...
        isscalar(fmdl.current_density)
    current_density = double(fmdl.current_density);
    current_density_applied = isfinite(current_density) && current_density > 0;
end
if ~isfield(fmdl, 'stimulation') || isempty(fmdl.stimulation)
    stim_matrix_raw = zeros(0, n_elec);
    stim_matrix = zeros(0, n_elec);
    meas_matrices = zeros(0, 0, n_elec);
    measurement_counts = zeros(0, 1);
    volt_matrix = zeros(0, n_elec);
    volt_pattern_present = false(0, 1);
    interior_sources = zeros(0, 0);
    interior_source_counts = zeros(0, 1);
    stimulation_labels = cell(0, 1);
    positive_current = zeros(0, 1);
    negative_current = zeros(0, 1);
    net_current = zeros(0, 1);
    max_abs_current = zeros(0, 1);
    balanced = false(0, 1);
    supported = false;
    return;
end
n_stim = numel(fmdl.stimulation);
measurement_counts = zeros(n_stim, 1);
interior_source_counts = zeros(n_stim, 1);
max_measurements = 0;
max_interior = 0;
stim_matrix_raw = zeros(n_stim, n_elec);
volt_matrix = zeros(n_stim, n_elec);
volt_pattern_present = false(n_stim, 1);
stimulation_labels = cell(n_stim, 1);
supported = true;
for i = 1:n_stim
    stim = fmdl.stimulation(i);
    if ~isfield(stim, 'stim_pattern') || ...
            numel(stim.stim_pattern) ~= n_elec
        supported = false;
        continue;
    end
    stim_matrix_raw(i, :) = full(double(stim.stim_pattern(:)'));
    if isfield(stim, 'meas_pattern')
        measurement_counts(i) = size(stim.meas_pattern, 1);
    else
        supported = false;
    end
    max_measurements = max(max_measurements, measurement_counts(i));
    if isfield(stim, 'volt_pattern') && ~isempty(stim.volt_pattern)
        if numel(stim.volt_pattern) == n_elec
            volt_matrix(i, :) = full(double(stim.volt_pattern(:)'));
        end
        volt_pattern_present(i) = true;
        supported = false;
    end
    if isfield(stim, 'interior_sources') && ~isempty(stim.interior_sources)
        interior_source_counts(i) = numel(stim.interior_sources);
        max_interior = max(max_interior, interior_source_counts(i));
        supported = false;
    end
    if isfield(stim, 'stimulation') && ~isempty(stim.stimulation)
        stimulation_labels{i} = char(stim.stimulation);
    else
        stimulation_labels{i} = 'unspecified';
    end
end
stim_matrix = stim_matrix_raw;
if current_density_applied
    stim_matrix = stim_matrix ./ current_density;
end
meas_matrices = zeros(n_stim, max_measurements, n_elec);
interior_sources = zeros(n_stim, max_interior);
for i = 1:n_stim
    count = measurement_counts(i);
    if count > 0
        one_meas = full(double(fmdl.stimulation(i).meas_pattern));
        meas_matrices(i, 1:count, :) = reshape(one_meas, 1, count, n_elec);
    end
    interior_count = interior_source_counts(i);
    if interior_count > 0
        interior_sources(i, 1:interior_count) = ...
            full(double(fmdl.stimulation(i).interior_sources(:)'));
    end
end
if ~isreal(stim_matrix) || ~isreal(meas_matrices)
    supported = false;
end
positive_current = NaN(n_stim, 1);
negative_current = NaN(n_stim, 1);
net_current = sum(stim_matrix, 2);
max_abs_current = max(abs(stim_matrix), [], 2);
balanced = false(n_stim, 1);
if isreal(stim_matrix)
    positive_current = sum(max(stim_matrix, 0), 2);
    negative_current = -sum(min(stim_matrix, 0), 2);
    tolerance = 1e-12 * max(1, max_abs_current);
    balanced = abs(net_current) <= tolerance;
end
end

function [value, present, source] = local_normalize_measurements(fmdl)
present = isfield(fmdl, 'normalize_measurements') || isfield(fmdl, 'normalize');
value = double(mdl_normalize(fmdl));
if present
    source = 'exact_model_field';
else
    source = 'eidors_runtime_default_mdl_normalize';
end
end

function [gnd, present, effective, source] = local_ground_node(fmdl)
present = isfield(fmdl, 'gnd_node') && ~isempty(fmdl.gnd_node);
if present
    gnd = double(fmdl.gnd_node);
    effective = gnd;
    source = 'exact_model_field';
else
    gnd = NaN;
    solver_name = local_effective_function_name( ...
        local_function_name(local_field_or_empty(fmdl, 'solve')), 'fwd_solve');
    if strcmp(solver_name, 'fwd_solve_1st_order')
        center = mean(fmdl.nodes, 1);
        distance2 = sum(bsxfun(@minus, fmdl.nodes, center).^2, 2);
        [~, effective] = min(distance2);
        effective = double(effective);
        source = 'derived_eidors_fwd_solve_1st_order_center_node';
    else
        effective = NaN;
        source = 'missing_unknown_custom_solver_behavior';
    end
end
end

function result = local_resolve_image(img, fmdl, source_path, ...
        selection_method, role)
result.present = false;
result.scalar_present = false;
result.scalar_value = NaN;
result.elem_data = zeros(size(fmdl.elems, 1), 0);
result.source_path = source_path;
result.selection_method = selection_method;
result.role = role;
result.parameterization = 'missing';
result.mapping = 'not_run';
result.error = '';
result.coarse2fine_applied = false;
result.model_background_applied = false;
if isempty(img)
    return;
end
try
    working = img;
    working.fwd_model = fmdl;
    if isfield(working, 'params_mapping') && ...
            isfield(working.params_mapping, 'function')
        working = feval(working.params_mapping.function, working);
        result.mapping = 'params_mapping_function';
    end
    if isfield(fmdl, 'coarse2fine') && isfield(working, 'elem_data')
        c2f = fmdl.coarse2fine;
        if size(working.elem_data, 1) == size(c2f, 2)
            working.elem_data = c2f * working.elem_data;
            result.coarse2fine_applied = true;
            if isfield(fmdl, 'background')
                working.elem_data = working.elem_data + fmdl.background;
                result.model_background_applied = true;
            end
        end
    end
    mapped = data_mapper(working);
    if isfield(mapped, 'current_params') && ~isempty(mapped.current_params)
        result.parameterization = char(mapped.current_params);
    else
        result.parameterization = 'unspecified';
    end
    converted = convert_img_units(mapped, 'conductivity');
    if ~isfield(converted, 'elem_data') || ...
            size(converted.elem_data, 1) ~= size(fmdl.elems, 1)
        error(['Mapped image does not provide one conductivity value per ', ...
               'forward-model element.']);
    end
    result.elem_data = double(converted.elem_data);
    result.present = true;
    if strcmp(result.mapping, 'not_run')
        result.mapping = 'data_mapper_then_convert_img_units_to_conductivity';
    else
        result.mapping = [result.mapping, ...
            '+data_mapper+convert_img_units_to_conductivity'];
    end
    values = result.elem_data(:);
    if ~isempty(values) && all(isfinite(values))
        tolerance = 1e-12 * max(1, max(abs(values)));
        if max(abs(values - values(1))) <= tolerance
            result.scalar_present = true;
            result.scalar_value = values(1);
        end
    end
catch err
    result.error = err.message;
    result.mapping = 'unsupported_or_failed';
end
end

function [valid, message] = local_validate_fwd_model(fmdl)
valid = false;
message = '';
try
    [valid, message] = valid_fwd_model(fmdl);
catch err
    message = err.message;
end
valid = logical(valid);
end

function value = local_field_or_empty(object, field_name)
if isfield(object, field_name)
    value = object.(field_name);
else
    value = [];
end
end

function name = local_function_name(value)
if isa(value, 'function_handle')
    name = func2str(value);
elseif ischar(value) || isstring(value)
    name = char(value);
elseif isnumeric(value)
    name = 'numeric_matrix';
else
    name = 'missing';
end
end

function name = local_effective_function_name(declared, default_kind)
name = declared;
if strcmp(declared, 'eidors_default')
    try
        value = eidors_default('get', default_kind);
        name = local_function_name(value);
    catch
        name = 'eidors_default_unresolved';
    end
end
end

function value = local_text_field(object, field_name, fallback)
if isfield(object, field_name) && ...
        (ischar(object.(field_name)) || isstring(object.(field_name)))
    value = char(object.(field_name));
else
    value = fallback;
end
end

function record = local_field_shape_record(object, field_name)
record.present = isfield(object, field_name) && ~isempty(object.(field_name));
if ~record.present
    record.class = 'missing';
    record.size = [];
    return;
end
value = object.(field_name);
record.class = class(value);
record.size = size(value);
if isa(value, 'function_handle')
    record.function = func2str(value);
else
    record.function = '';
end
end

function unit = local_contact_impedance_unit(dimension, coordinate_units)
if strcmp(coordinate_units, 'unspecified')
    length_unit = 'source_length_unit';
else
    length_unit = coordinate_units;
end
if dimension == 2
    unit = ['ohm*', length_unit];
else
    unit = ['ohm*', length_unit, '^2'];
end
end

function [blockers, warnings] = local_forward_readiness( ...
        model_valid, validation_error, contact_present, electrode_model, ...
        projection_required, background_image, target_image, ...
        stimulation_supported, solver_name, gnd_present, effective_gnd_source)
blockers = {};
warnings = {};
if ~model_valid
    warnings{end + 1} = ['EIDORS valid_fwd_model rejected the source: ', ...
        validation_error];
end
cem_mask = strcmp(electrode_model, 'cem') | ...
    strcmp(electrode_model, 'cem_faces');
if any(~contact_present & cem_mask)
    blockers{end + 1} = ...
        'contact_impedance_missing_no_eidors_default';
end
if any(projection_required)
    blockers{end + 1} = ...
        'unexpected_electrode_projection_requested';
end
if ~background_image.present
    blockers{end + 1} = 'background_image_missing_or_unmappable';
end
if ~target_image.present
    warnings{end + 1} = 'No target image was selected; geometry/background only.';
end
if ~stimulation_supported
    blockers{end + 1} = ...
        'stimulation_missing_or_unsupported_voltage_interior_complex_pattern';
end
if ~strcmp(solver_name, 'fwd_solve_1st_order')
    blockers{end + 1} = ...
        'custom_eidors_forward_solver_semantics_not_portable';
end
if ~gnd_present && strcmp(effective_gnd_source, ...
        'missing_unknown_custom_solver_behavior')
    blockers{end + 1} = 'ground_node_missing_custom_solver_behavior_unknown';
elseif ~gnd_present
    warnings{end + 1} = ...
        'gnd_node missing; EIDORS first-order center-node fallback was recorded.';
end
end

function record = local_selection_record(source_path, method)
record.source_path = source_path;
record.method = method;
if isempty(source_path)
    record.status = 'missing';
elseif strcmp(method, 'explicit_selector') || ...
        strcmp(method, 'unique_standard_object')
    record.status = 'exact';
else
    record.status = 'inferred';
end
end

function record = local_presence_record(presence, source_path, note)
record.source_path = source_path;
record.present = logical(presence);
record.note = note;
if all(presence)
    record.status = 'exact';
elseif any(presence)
    record.status = 'partial';
else
    record.status = 'missing';
end
end

function record = local_runtime_record(present, runtime_source)
record.source_present = logical(present);
record.effective_source = runtime_source;
if present
    record.status = 'exact';
elseif startsWith(runtime_source, 'missing')
    record.status = 'missing';
else
    record.status = 'runtime_default';
end
end

function record = local_image_record(image)
record.source_path = image.source_path;
record.selection_method = image.selection_method;
record.role = image.role;
record.parameterization = image.parameterization;
record.mapping = image.mapping;
record.coarse2fine_applied = image.coarse2fine_applied;
record.model_background_applied = image.model_background_applied;
record.error = image.error;
if image.present
    if startsWith(image.selection_method, 'inferred')
        record.status = 'derived';
    else
        record.status = 'exact';
    end
elseif isempty(image.error)
    record.status = 'missing';
else
    record.status = 'unsupported';
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
if ~isfield(cfg, 'protocol_mat') || exist(cfg.protocol_mat, 'file') ~= 2
    error('Bridge v3 protocol.mat was not found.');
end
protocol = load(cfg.protocol_mat);
if ~isfield(cfg, 'fields_mat') || exist(cfg.fields_mat, 'file') ~= 2
    error('Bridge v3 fields.mat was not found.');
end
fields_payload = load(cfg.fields_mat);
nodes = double(payload.nodes);
elems = double(payload.elems);
if isfield(payload, 'boundary_facets')
    boundary_edges = double(payload.boundary_facets);
else
    boundary_edges = double(payload.boundary_edges);
end
electrode_nodes = double(payload.electrode_nodes);
electrode_counts = double(payload.electrode_node_counts(:));
contact_impedance = double(payload.contact_impedance);
electrode_models = repmat({'cem'}, size(electrode_nodes, 1), 1);
if isfield(payload, 'electrode_model')
    electrode_models = local_text_vector(payload.electrode_model);
end
if isfield(payload, 'contact_impedance_present')
    contact_present = logical(payload.contact_impedance_present(:));
    if numel(contact_present) == 1
        contact_present = repmat(contact_present, size(electrode_nodes, 1), 1);
    end
    cem_mask = strcmp(strtrim(electrode_models), 'cem') | ...
        strcmp(strtrim(electrode_models), 'cem_faces');
    if numel(contact_present) ~= size(electrode_nodes, 1) || ...
            any(~contact_present & cem_mask)
        error(['Bridge geometry has missing contact impedance. ', ...
               'EIDORS has no universal z_contact default.']);
    end
end
background = double(payload.background);
stim_pattern = local_or_default(cfg, 'stim_pattern', '{ad}');
meas_pattern = local_or_default(cfg, 'meas_pattern', '{ad}');
rotate_meas = logical(local_or_default(cfg, 'rotate_meas', true));
use_meas_current = logical(local_or_default(cfg, 'use_meas_current', false));
drive_value = double(local_or_default(cfg, 'drive_value', 1.0));

n_logical_elec = double(size(electrode_nodes, 1));
[logical_to_physical, physical_nodes, physical_logical, ...
 logical_primary_physical] = local_expand_logical_electrodes( ...
    payload, electrode_nodes, electrode_counts, electrode_models);
n_elec = double(size(logical_to_physical, 1));
fmdl = eidors_obj('fwd_model', 'pyeidors_bridge_geometry');
fmdl.nodes = nodes;
fmdl.elems = elems;
fmdl.boundary = boundary_edges;
if isfield(payload, 'effective_gnd_node') && ...
        isfinite(double(payload.effective_gnd_node))
    fmdl.gnd_node = double(payload.effective_gnd_node);
elseif isfield(payload, 'gnd_node') && isfinite(double(payload.gnd_node))
    fmdl.gnd_node = double(payload.gnd_node);
else
    error('Bridge geometry has no explicit/effective EIDORS ground-node semantics.');
end
fmdl.solve = @fwd_solve_1st_order;
fmdl.system_mat = @system_mat_1st_order;
fmdl.jacobian = @jacobian_adjoint;
if isfield(protocol, 'normalize_measurements')
    fmdl.normalize_measurements = logical(protocol.normalize_measurements);
else
    error('Bridge v3 protocol has no normalize_measurements semantics.');
end

for physical_idx = 1:n_elec
    logical_idx = physical_logical(physical_idx);
    fmdl.electrode(physical_idx).nodes = physical_nodes{physical_idx};
    model_kind = strtrim(electrode_models{logical_idx});
    is_pem = strcmp(model_kind, 'point') || ...
        strcmp(model_kind, 'distributed_point') || strcmp(model_kind, 'pem');
    if is_pem
        fmdl.electrode(physical_idx).z_contact = 1;
        fmdl.electrode(physical_idx).pyeidors_z_contact_nonphysical = true;
        fmdl.electrode(physical_idx).pyeidors_logical_electrode = logical_idx;
        fmdl.electrode(physical_idx).pyeidors_logical_weight = ...
            logical_to_physical(physical_idx, logical_idx);
    elseif numel(contact_impedance) == 1
        fmdl.electrode(physical_idx).z_contact = contact_impedance;
    else
        fmdl.electrode(physical_idx).z_contact = contact_impedance(logical_idx);
    end
    if ~is_pem && isfield(payload, 'electrode_face_counts') && ...
            isfield(payload, 'electrode_faces')
        face_count = double(payload.electrode_face_counts(logical_idx));
        if face_count > 0
            one_faces = double(payload.electrode_faces( ...
                logical_idx, 1:face_count, :));
            fmdl.electrode(physical_idx).faces = reshape( ...
                one_faces, face_count, size(boundary_edges, 2));
            fmdl.electrode(physical_idx).nodes = [];
        end
    end
end
fmdl.pyeidors_bridge.logical_electrode_count = n_logical_elec;
fmdl.pyeidors_bridge.logical_to_physical = logical_to_physical;
fmdl.pyeidors_bridge.physical_to_logical = physical_logical;
fmdl.pyeidors_bridge.logical_primary_physical = logical_primary_physical;

if isfield(payload, 'electrode_boundary_kind') && ...
        isfield(payload, 'cem_face_nodes') && ...
        isfield(payload, 'cem_face_electrode')
    boundary_kind = local_text_vector(payload.electrode_boundary_kind);
    face_electrode = double(payload.cem_face_electrode(:));
    all_faces = double(payload.cem_face_nodes);
    interior_faces = zeros(0, size(boundary_edges, 2));
    for logical_idx = 1:n_logical_elec
        rows = face_electrode == logical_idx;
        if any(rows)
            physical_idx = logical_primary_physical(logical_idx);
            fmdl.electrode(physical_idx).faces = all_faces(rows, :);
            fmdl.electrode(physical_idx).nodes = [];
            if strcmp(strtrim(boundary_kind{logical_idx}), 'interior')
                interior_faces = [interior_faces; all_faces(rows, :)]; %#ok<AGROW>
            end
        end
    end
    if ~isempty(interior_faces)
        fmdl.system_mat_fields.CEM_boundary = interior_faces;
    end
end

if isfield(protocol, 'stim_matrix') && ~isempty(protocol.stim_matrix) && ...
        isfield(protocol, 'meas_matrices') && ...
        isfield(protocol, 'measurement_counts')
    stim_matrix = double(protocol.stim_matrix);
    measurement_counts = double(protocol.measurement_counts(:));
    if size(stim_matrix, 2) ~= n_logical_elec
        error('Bridge v3 stimulation width does not match logical electrodes.');
    end
    for i = 1:size(stim_matrix, 1)
        physical_stim = logical_to_physical * stim_matrix(i, :)';
        fmdl.stimulation(i).stim_pattern = sparse(physical_stim);
        count = measurement_counts(i);
        one_meas = double(protocol.meas_matrices(i, 1:count, :));
        logical_meas = reshape(one_meas, count, n_logical_elec);
        fmdl.stimulation(i).meas_pattern = sparse( ...
            logical_meas * logical_to_physical');
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
    logical_stimulation = mk_stim_patterns(n_logical_elec, 1, ...
        stim_pattern, meas_pattern, stim_options, drive_value);
    for i = 1:numel(logical_stimulation)
        fmdl.stimulation(i) = logical_stimulation(i);
        fmdl.stimulation(i).stim_pattern = sparse( ...
            logical_to_physical * logical_stimulation(i).stim_pattern);
        fmdl.stimulation(i).meas_pattern = sparse( ...
            logical_stimulation(i).meas_pattern * logical_to_physical');
    end
end
if isfield(protocol, 'meas_select') && ~isempty(protocol.meas_select)
    fmdl.meas_select = protocol.meas_select;
end
if isfield(fields_payload, 'coarse2fine') && ...
        ~isempty(fields_payload.coarse2fine)
    fmdl.coarse2fine = fields_payload.coarse2fine;
end
if isfield(fields_payload, 'background_elem_data') && ...
        ~isempty(fields_payload.background_elem_data)
    img_background = eidors_obj('image', 'pyeidors_bridge_background'); %#ok<NASGU>
    img_background.fwd_model = fmdl;
    img_background.elem_data = fields_payload.background_elem_data(:);
end
if isfield(fields_payload, 'target_elem_data') && ...
        ~isempty(fields_payload.target_elem_data)
    img_target = eidors_obj('image', 'pyeidors_bridge_target'); %#ok<NASGU>
    img_target.fwd_model = fmdl;
    img_target.elem_data = fields_payload.target_elem_data(:);
end

runtime_roundtrip = fwd_model_parameters(fmdl, 'skip_VOLUME');
local_assert_logical_n2e( ...
    protocol, runtime_roundtrip.N2E, logical_to_physical);
local_assert_runtime_operator(protocol, 'QQ', runtime_roundtrip.QQ);
local_assert_runtime_operator(protocol, 'VV', runtime_roundtrip.VV);
local_assert_logical_v2meas( ...
    protocol, runtime_roundtrip.v2meas, logical_to_physical, ...
    numel(fmdl.stimulation));

if isfield(cfg, 'measurements_csv') && exist(cfg.measurements_csv, 'file') == 2
    T = readtable(cfg.measurements_csv);
    vh_meas = double(T{:, 1});
    vi_meas = double(T{:, 2});
elseif isfield(cfg, 'measurements_mat') && exist(cfg.measurements_mat, 'file') == 2
    measurement_payload = load(cfg.measurements_mat);
    if ~isfield(measurement_payload, 'homogeneous') || ...
            ~isfield(measurement_payload, 'target')
        error('measurements.mat must contain homogeneous and target arrays.');
    end
    vh_meas = measurement_payload.homogeneous(:);
    vi_meas = measurement_payload.target(:);
else
    vh_meas = [];
    vi_meas = [];
end
if ~isempty(vh_meas) && ~isempty(vi_meas)
    vh = eidors_obj('data', 'pyeidors_bridge_homogeneous'); %#ok<NASGU>
    vh.meas = vh_meas;
    vi = eidors_obj('data', 'pyeidors_bridge_target'); %#ok<NASGU>
    vi.meas = vi_meas;
    fprintf('EIDORS bridge measurements loaded: %d points\n', numel(vi_meas));
end

fprintf('EIDORS bridge project loaded from %s\n', cfg.geometry_mat);

function [logical_to_physical, physical_nodes, physical_logical, ...
          logical_primary] = local_expand_logical_electrodes( ...
          payload, electrode_nodes, electrode_counts, electrode_models)
n_logical = size(electrode_nodes, 1);
pem_weights = zeros(size(electrode_nodes));
if isfield(payload, 'pem_node_weights')
    pem_weights = double(payload.pem_node_weights);
end
physical_nodes = {};
physical_logical = zeros(0, 1);
physical_weights = zeros(0, 1);
logical_primary = zeros(n_logical, 1);
for logical_idx = 1:n_logical
    count = electrode_counts(logical_idx);
    active_nodes = electrode_nodes(logical_idx, 1:count);
    model_kind = strtrim(electrode_models{logical_idx});
    is_pem = strcmp(model_kind, 'point') || ...
        strcmp(model_kind, 'distributed_point') || strcmp(model_kind, 'pem');
    if is_pem
        weights = pem_weights(logical_idx, 1:count);
        if count == 1 && all(weights == 0)
            weights = 1;
        end
        if ~isreal(weights) || any(~isfinite(weights)) || any(weights < 0) || ...
                abs(sum(weights) - 1) > 1e-12 * max(1, max(abs(weights)))
            error('Bridge v3 PEM electrode %d has invalid exact weights.', ...
                logical_idx);
        end
        for node_idx = 1:count
            physical_idx = numel(physical_nodes) + 1;
            physical_nodes{physical_idx, 1} = active_nodes(node_idx); %#ok<AGROW>
            physical_logical(physical_idx, 1) = logical_idx; %#ok<AGROW>
            physical_weights(physical_idx, 1) = weights(node_idx); %#ok<AGROW>
            if logical_primary(logical_idx) == 0
                logical_primary(logical_idx) = physical_idx;
            end
        end
    else
        physical_idx = numel(physical_nodes) + 1;
        physical_nodes{physical_idx, 1} = active_nodes(active_nodes > 0); %#ok<AGROW>
        physical_logical(physical_idx, 1) = logical_idx; %#ok<AGROW>
        physical_weights(physical_idx, 1) = 1; %#ok<AGROW>
        logical_primary(logical_idx) = physical_idx;
    end
end
logical_to_physical = sparse( ...
    (1:numel(physical_nodes))', physical_logical, physical_weights, ...
    numel(physical_nodes), n_logical);
end

function local_assert_runtime_operator(protocol, field_name, actual)
if ~isfield(protocol, field_name)
    return;
end
expected = full(protocol.(field_name));
actual = full(actual);
if ~isequal(size(expected), size(actual)) || ...
        norm(double(expected(:) - actual(:))) > ...
        1e-12 * max(1, norm(double(expected(:))))
    error('Bridge v3 runtime operator %s did not round-trip.', field_name);
end
end

function local_assert_logical_n2e(protocol, actual, logical_to_physical)
if ~isfield(protocol, 'N2E')
    return;
end
logical_actual = logical_to_physical' * full(actual);
local_assert_runtime_operator(protocol, 'N2E', logical_actual);
end

function local_assert_logical_v2meas( ...
        protocol, actual, logical_to_physical, n_stim)
if ~isfield(protocol, 'v2meas')
    return;
end
lift = kron(speye(n_stim), logical_to_physical);
logical_expected = full(protocol.v2meas);
physical_expected = lift * logical_expected;
if ~isequal(size(physical_expected), size(actual)) || ...
        norm(double(physical_expected(:) - actual(:))) > ...
        1e-12 * max(1, norm(double(physical_expected(:))))
    error('Bridge v3 logical v2meas did not round-trip after PEM expansion.');
end
end

function value = local_or_default(cfg, field_name, default_value)
if isfield(cfg, field_name)
    value = cfg.(field_name);
else
    value = default_value;
end
end

function values = local_text_vector(raw)
if iscell(raw)
    values = cellfun(@(value) strtrim(char(value)), raw(:), ...
        'UniformOutput', false);
elseif isstring(raw)
    values = cellstr(raw(:));
elseif ischar(raw)
    values = cellstr(raw);
else
    error('Bridge v3 text-vector field has an unsupported MATLAB type.');
end
end
"""
