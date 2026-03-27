function render_eidors_native_reconstruction(config_path)
% Render an EIDORS-native reconstruction figure from bridge artifacts.

if nargin < 1
    error('render_eidors_native_reconstruction requires a JSON config path.');
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

geom = load(char(cfg.geometry_mat));
details = load(char(cfg.details_mat));
fmdl = build_fwd_model_from_exchange(geom);

background = double(geom.background);
img_truth = mk_image(fmdl, background);
img_truth.elem_data = double(details.truth_elem_data(:));

img_recon = mk_image(fmdl, background);
img_recon.elem_data = double(details.recon_elem_data(:));

fig = figure('Visible', 'off', 'Color', 'white', 'Position', [100, 100, 1200, 520]);

subplot(1, 2, 1);
show_fem(img_truth, [1, 0, 0]);
axis equal off;

subplot(1, 2, 2);
show_fem(img_recon, [1, 0, 0]);
axis equal off;
cond_rmse = double(details.conductivity_rmse);

annotation(fig, 'textbox', [0.36, 0.02, 0.28, 0.06], ...
    'String', sprintf('Conductivity RMSE = %.4e', cond_rmse), ...
    'Interpreter', 'none', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', ...
    'FontWeight', 'bold', ...
    'FontSize', 12, ...
    'LineWidth', 0.8, ...
    'BackgroundColor', [1, 1, 1], ...
    'EdgeColor', [0.6, 0.6, 0.6], ...
    'FitBoxToText', 'off');

out_path = char(cfg.output_png);
out_dir = fileparts(out_path);
if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end
exportgraphics(fig, out_path, 'Resolution', 300, 'BackgroundColor', 'white');
close(fig);
fprintf('Wrote %s\n', out_path);
end

function fmdl = build_fwd_model_from_exchange(payload)
nodes = double(payload.nodes);
elems = double(payload.elems);
if isfield(payload, 'boundary_edges')
    boundary_edges = double(payload.boundary_edges);
else
    boundary_edges = double(find_boundary(elems));
end
electrode_nodes = double(payload.electrode_nodes);
electrode_counts = double(payload.electrode_node_counts(:));
contact_impedance = double(payload.contact_impedance);

n_elec = double(size(electrode_nodes, 1));
fmdl = eidors_obj('fwd_model', 'bridge_native_render');
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
