%% EIDORS 选定案例：经典 CEM 与 Robin CEM / Selected-case walkthrough
% 中文：在 MATLAB 编辑器中逐节运行；关键变量会保留在工作区，可在变量编辑器
% 或断点处检查。
% English: Run section-by-section in the MATLAB Editor. Important variables
% remain in the workspace for the Variable Editor and breakpoints.

%% 1. 用户设置 / User settings
case_id = "X01";
eidors_startup = "";  % Example: "C:\eidors-v3.12-ng\eidors\startup.m"
rebuild_eidors_blocks = true;
save_debug_workspace = false;

script_path = mfilename("fullpath");
package_dir = fileparts(script_path);
repo_root = fileparts(fileparts(package_dir));
suite_output = fullfile(repo_root, "output", "cem_exact_extension");
expected_metrics_path = fullfile( ...
    package_dir, "expected", "cem_exact_extension_metrics.csv");

%% 2. 加载 EIDORS / Load EIDORS
if exist('eidors_default', 'file') ~= 2
    if strlength(eidors_startup) == 0 || exist(eidors_startup, 'file') ~= 2
        error([ ...
            "EIDORS is not on the MATLAB path. Set eidors_startup in ", ...
            "Section 1 to the EIDORS startup.m file." ...
        ]);
    end
    run(eidors_startup);
end
fprintf("EIDORS version: %s\n", eidors_obj('eidors_version'));

%% 3. 定位精确共享 MAT 夹具 / Resolve the exact shared MAT fixture
case_dirs = dir(fullfile(suite_output, "cases", case_id + "_*"));
case_dirs = case_dirs([case_dirs.isdir]);
if isempty(case_dirs) && case_id == "X01"
    case_dir = fullfile(package_dir, "fixtures", "X01");
    mesh_mat = fullfile( ...
        case_dir, "common_mesh", "cem_exact_extension_p1.mat");
else
    if numel(case_dirs) ~= 1
        error("Expected exactly one directory for %s.", case_id);
    end
    case_dir = fullfile(case_dirs(1).folder, case_dirs(1).name);
    mesh_mat = fullfile( ...
        case_dir, "common_mesh", "cem_exact_extension_p1.mat");
end
if exist(mesh_mat, "file") ~= 2
    error("Shared MAT fixture not found: %s", mesh_mat);
end
mesh_json = replace(mesh_mat, ".mat", ".json");
if exist(mesh_json, "file") ~= 2
    error("Shared JSON metadata not found: %s", mesh_json);
end
payload = load(mesh_mat);
mesh_metadata = jsondecode(fileread(mesh_json));

nodes = double(payload.nodes);
elements = double(payload.elems);
boundary_edges = double(payload.boundary_edges);
tagged_boundary_edges = double(payload.tagged_boundary_edges);
electrode_nodes = double(payload.electrode_nodes);
electrode_counts = double(payload.electrode_node_counts(:));
currents = double(payload.current_patterns);
electrode_count = double(payload.n_elec);
contact_impedance = double(payload.contact_impedance);
if isfield(payload, "truth_elem_data")
    cell_conductivity = double(payload.truth_elem_data(:));
else
    cell_conductivity = repmat( ...
        double(payload.background), size(elements, 1), 1);
end

fprintf( ...
    "%s: nodes=%d, triangles=%d, electrodes=%d, RHS=%d, P%d, %s\n", ...
    case_id, size(nodes, 1), size(elements, 1), ...
    electrode_count, size(currents, 2), ...
    mesh_metadata.potential_order, mesh_metadata.scalar_dtype);
fprintf("Mesh fingerprint: %s\n", mesh_metadata.mesh_fingerprint);
fprintf("Cell conductivity values: ");
fprintf("%.12g ", unique(cell_conductivity));
fprintf("\nFirst current-column sum: %.3e\n", sum(currents(:, 1)));

%% 4. 变量字典 / Variable dictionary
% N = 体节点数；K = 三角形数；L = 电极数；P = 电流模式数。
% N = body-node count; K = triangle count; L = electrode count;
% P = number of current-pattern right-hand sides.
variable_dictionary = table( ...
    ["nodes"; "elements"; "tagged_boundary_edges"; ...
     "cell_conductivity"; "currents"; "A_R"; "C"; "D"; ...
     "classic_body_potential"; "classic_electrode_voltage"; ...
     "Q"; "response_basis"; "reduced_map"], ...
    ["N-by-2"; "K-by-3"; "Eb-by-3"; "K-by-1"; "L-by-P"; ...
     "N-by-N"; "N-by-L"; "L-by-L"; "N-by-P"; "L-by-P"; ...
     "L-by-(L-1)"; "N-by-(L-1)"; "(L-1)-by-(L-1)"], ...
    ["节点坐标 / node coordinates"; ...
     "P1 三角形 / P1 triangles"; ...
     "边界边与电极标签 / boundary edges and electrode labels"; ...
     "逐单元电导率 / per-cell conductivity"; ...
     "边界注入电流 / boundary injected currents"; ...
     "体刚度加 Robin 边界质量 / body stiffness plus Robin mass"; ...
     "体节点/电极耦合 / body-electrode coupling"; ...
     "电极边界积分块 / electrode boundary block"; ...
     "体内节点电势 / body nodal potential"; ...
     "边界电极电压 / boundary electrode voltage"; ...
     "电极零和正交基 / orthonormal zero-sum basis"; ...
     "A_R^{-1}CQ 的求解结果 / solved response basis"; ...
     "Robin 零和约化矩阵 / Robin reduced map"], ...
    'VariableNames', {'Variable', 'Shape', 'Meaning'});
disp(variable_dictionary);

%% 5. 显示相同网格、电导率和边界电流 / Display fair forward inputs
% X01 是均匀背景正问题，没有内部异常物。输入是网格、sigma、z 和 I，
% 输出是体电势 u 与边界电极电压 U；这里不是逆问题重构。
% X01 is a uniform-background forward problem without an interior anomaly.
% The mesh, sigma, z, and I are inputs; body potential u and boundary
% electrode voltage U are outputs. This is not an inverse reconstruction.
plot_forward_fixture_matlab( ...
    nodes, elements, tagged_boundary_edges, cell_conductivity, ...
    currents, case_id, string(mesh_metadata.mesh_fingerprint));

%% 6. 用 EIDORS 组装 A_R、C、D / Assemble A_R, C, and D with EIDORS
% A_R = K(sigma) + 边界 Robin 质量矩阵 / boundary Robin mass
% C   = 体节点/电极耦合 / body-electrode coupling
% D   = 电极对角块 / electrode diagonal block
fmdl = eidors_obj('fwd_model', 'professor_cem_walkthrough');
fmdl.nodes = nodes;
fmdl.elems = elements;
fmdl.boundary = boundary_edges;
fmdl.gnd_node = choose_ground_node( ...
    nodes, electrode_nodes, electrode_counts);
fmdl.solve = @fwd_solve_1st_order;
fmdl.system_mat = @system_mat_1st_order;
fmdl.jacobian = @jacobian_adjoint;
fmdl.normalize_measurements = 0;

for electrode = 1:electrode_count
    active = electrode_nodes( ...
        electrode, 1:electrode_counts(electrode));
    fmdl.electrode(electrode).nodes = active(active > 0);
    fmdl.electrode(electrode).z_contact = contact_impedance;
end
for pattern = 1:size(currents, 2)
    stimulation(pattern).stim_pattern = ...
        sparse(currents(:, pattern)); %#ok<SAGROW>
    stimulation(pattern).meas_pattern = ...
        speye(electrode_count); %#ok<SAGROW>
end
fmdl.stimulation = stimulation;

img = mk_image(fmdl, double(payload.background));
if isfield(payload, "truth_elem_data")
    img.elem_data = double(payload.truth_elem_data(:));
end
img.fwd_solve.get_all_nodes = 1;

if rebuild_eidors_blocks
    eidors_system_matrix = calc_system_mat(img);
    full_eidors_matrix = eidors_system_matrix.E;
else
    stored_blocks = load(fullfile( ...
        case_dir, "eidors_assembled_blocks.mat")); %#ok<UNRCH>
    full_eidors_matrix = [];
end

node_count = size(nodes, 1);
if rebuild_eidors_blocks
    A_R = full_eidors_matrix(1:node_count, 1:node_count);
    C = full_eidors_matrix( ...
        1:node_count, node_count + (1:electrode_count));
    D = full_eidors_matrix( ...
        node_count + (1:electrode_count), ...
        node_count + (1:electrode_count));
else
    A_R = stored_blocks.A_R; %#ok<UNRCH>
    C = stored_blocks.C;
    D = stored_blocks.D;
end

%% 7. 传统经典 CEM：一次增广块求解 / One augmented block solve
% [ A_R   C    0 ] [u     ]   [0]
% [ C'    D    1 ] [U     ] = [I]
% [ 0     1'   0 ] [lambda]   [0]
gauge_column = sparse(ones(electrode_count, 1));
classic_matrix = [ ...
    A_R, C, sparse(node_count, 1); ...
    C.', D, gauge_column; ...
    sparse(1, node_count), gauge_column.', sparse(1, 1) ...
];
classic_rhs = zeros( ...
    node_count + electrode_count + 1, size(currents, 2));
classic_rhs(node_count + (1:electrode_count), :) = currents;

classic_factor = decomposition(classic_matrix, "lu");
classic_solution = classic_factor \ classic_rhs;
classic_body_potential = classic_solution(1:node_count, :);
classic_electrode_voltage = classic_solution( ...
    node_count + (1:electrode_count), :);

%% 8. Robin CEM：消去体场并在零和电极子空间求解 / Reduced solve
% Q 张成 {U : 1' U = 0}。 / Q spans the zero-sum electrode subspace.
% R   = A_R^{-1} C Q
% T_r = Q' (D - C' A_R^{-1} C) Q
% T_r y = Q' I, U = Q y, u = -R y
Q = helmert_zero_sum_basis(electrode_count);
body_factor = decomposition(A_R, "lu");
coupling_basis = C * Q;
response_basis = body_factor \ coupling_basis;
schur_action_basis = D * Q - C.' * response_basis;
reduced_map = Q.' * schur_action_basis;
reduced_rhs = Q.' * currents;
reduced_factor = decomposition(full(reduced_map), "lu");
robin_coefficients = reduced_factor \ reduced_rhs;
robin_electrode_voltage = Q * robin_coefficients;
robin_body_potential = -(response_basis * robin_coefficients);

%% 9. Classic/Robin 正问题求解结果可视化 / Visualize solved forward results
% 上排显示相同注流模式下的 Classic/Robin 体电势和带符号差值。
% 下排显示 Classic/Robin 电极电压和舍入级电压差。
% The top row shows Classic/Robin body potentials and their signed delta
% for the same drive. The bottom row shows electrode voltages and the
% roundoff-level voltage difference.
plot_forward_solution_matlab( ...
    nodes, elements, ...
    classic_body_potential, robin_body_potential, ...
    classic_electrode_voltage, robin_electrode_voltage, ...
    currents, case_id);

%% 10. MATLAB 工作区中的直接检查 / Direct workspace checks
electrode_voltage_relative_l2 = relative_l2( ...
    robin_electrode_voltage, classic_electrode_voltage);
body_potential_relative_l2 = relative_l2( ...
    robin_body_potential, classic_body_potential);
classic_scaled_backward_residual = scaled_backward_residual( ...
    classic_matrix, classic_solution, classic_rhs);
robin_scaled_backward_residual = scaled_backward_residual( ...
    reduced_map, robin_coefficients, reduced_rhs);
classic_voltage_gauge_max_abs = max( ...
    abs(sum(classic_electrode_voltage, 1)));
robin_voltage_gauge_max_abs = max( ...
    abs(sum(robin_electrode_voltage, 1)));

fprintf("Classic vs Robin electrode relative L2: %.12e\n", ...
    electrode_voltage_relative_l2);
fprintf("Classic scaled backward residual:       %.12e\n", ...
    classic_scaled_backward_residual);
fprintf("Robin scaled backward residual:         %.12e\n", ...
    robin_scaled_backward_residual);
fprintf("Classic/Robin voltage gauge max abs:    %.3e / %.3e\n", ...
    classic_voltage_gauge_max_abs, robin_voltage_gauge_max_abs);

%% 11. 对接冻结的 38 案例有理数精确报告 / Connect to the exact-QQ report
expected_metrics = readtable(expected_metrics_path, "TextType", "string");
eidors_expected = expected_metrics( ...
    expected_metrics.case_id == case_id & ...
    expected_metrics.solver == "EIDORS", :);
disp(eidors_expected(:, { ...
    'case_id', 'formulation', 'truth_relative_l2', ...
    'exact_reduced_scaled_backward_residual', ...
    'voltage_gauge_relative_residual', ...
    'classic_robin_relative_l2' ...
}));

%% 12. 可选工作区快照 / Optional workspace snapshot
if save_debug_workspace
    debug_output = fullfile( ...
        repo_root, "output", "cem_professor_walkthrough", "eidors"); %#ok<UNRCH>
    if exist(debug_output, "dir") ~= 7
        mkdir(debug_output);
    end
    save(fullfile(debug_output, case_id + "_workspace.mat"));
end

%% 局部辅助函数 / Local helper functions
function plot_forward_fixture_matlab( ...
    nodes, elements, tagged_edges, conductivity, ...
    currents, case_id, mesh_fingerprint)
figure( ...
    'Name', case_id + " shared forward problem", ...
    'Color', 'white', ...
    'Position', [100, 100, 1250, 560]);
layout = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

mesh_axis = nexttile(layout);
patch( ...
    mesh_axis, ...
    'Faces', elements, ...
    'Vertices', nodes, ...
    'FaceVertexCData', conductivity, ...
    'FaceColor', 'flat', ...
    'EdgeColor', [0.75, 0.78, 0.82], ...
    'LineWidth', 0.75);
axis(mesh_axis, 'equal');
xlim(mesh_axis, [-1.14, 1.14]);
ylim(mesh_axis, [-1.14, 1.14]);
colormap(mesh_axis, parula);
colorbar(mesh_axis);
title(mesh_axis, "网格与单元电导率", ...
    'FontName', 'Microsoft YaHei');
subtitle(mesh_axis, "Mesh and cell conductivity", ...
    'FontName', 'Times New Roman');
xlabel(mesh_axis, "x", 'FontName', 'Times New Roman');
ylabel(mesh_axis, "y", 'FontName', 'Times New Roman');
set(mesh_axis, 'FontName', 'Times New Roman');
hold(mesh_axis, 'on');

drive_axis = nexttile(layout);
triplot( ...
    drive_axis, elements, nodes(:, 1), nodes(:, 2), ...
    'Color', [0.82, 0.84, 0.87], ...
    'LineWidth', 0.75);
axis(drive_axis, 'equal');
xlim(drive_axis, [-1.14, 1.14]);
ylim(drive_axis, [-1.14, 1.14]);
title(drive_axis, "第一个边界注入电流模式", ...
    'FontName', 'Microsoft YaHei');
subtitle(drive_axis, "First boundary-current drive pattern", ...
    'FontName', 'Times New Roman');
xlabel(drive_axis, "x", 'FontName', 'Times New Roman');
ylabel(drive_axis, "y", 'FontName', 'Times New Roman');
set(drive_axis, 'FontName', 'Times New Roman');
hold(drive_axis, 'on');

drive = currents(:, 1);
for edge_index = 1:size(tagged_edges, 1)
    label = tagged_edges(edge_index, 3);
    if label <= 0
        continue;
    end
    vertices = tagged_edges(edge_index, 1:2);
    coordinates = nodes(vertices, :);
    midpoint = mean(coordinates, 1);
    plot(mesh_axis, coordinates(:, 1), coordinates(:, 2), ...
        'Color', [0.07, 0.09, 0.13], 'LineWidth', 2.2);
    text(mesh_axis, midpoint(1) * 1.075, midpoint(2) * 1.075, ...
        string(label), 'HorizontalAlignment', 'center', ...
        'FontName', 'Times New Roman', 'FontSize', 8);
    current = drive(label);
    if current > 0
        color = [0.85, 0.47, 0.02];
        width = 5;
    elseif current < 0
        color = [0.11, 0.31, 0.72];
        width = 5;
    else
        color = [0.42, 0.45, 0.50];
        width = 2;
    end
    plot(drive_axis, coordinates(:, 1), coordinates(:, 2), ...
        'Color', color, 'LineWidth', width);
    text(drive_axis, midpoint(1) * 1.075, midpoint(2) * 1.075, ...
        string(label), 'HorizontalAlignment', 'center', ...
        'Color', color, 'FontName', 'Times New Roman', 'FontSize', 8);
end
sgtitle(layout, sprintf( ...
    '%s shared forward problem | mesh %s...', ...
    case_id, extractBefore(mesh_fingerprint, 17)), ...
    'FontName', 'Times New Roman');
end


function plot_forward_solution_matlab( ...
    nodes, elements, ...
    classic_body_all, robin_body_all, ...
    classic_voltage_all, robin_voltage_all, ...
    currents, case_id)
drive_index = 1;
classic_body = classic_body_all(:, drive_index);
robin_body = robin_body_all(:, drive_index);
body_delta = robin_body - classic_body;
classic_voltage = classic_voltage_all(:, drive_index);
robin_voltage = robin_voltage_all(:, drive_index);
voltage_delta = robin_voltage - classic_voltage;

body_min = min([classic_body; robin_body]);
body_max = max([classic_body; robin_body]);
if body_max <= body_min
    padding = max(abs(body_min), 1) * 1e-12;
    body_min = body_min - padding;
    body_max = body_max + padding;
end
body_delta_limit = max(max(abs(body_delta)), eps);
voltage_min = min([classic_voltage; robin_voltage]);
voltage_max = max([classic_voltage; robin_voltage]);
voltage_padding = max(voltage_max - voltage_min, 1) * 0.08;
voltage_delta_limit = max(max(abs(voltage_delta)), eps);

figure( ...
    'Name', case_id + " Classic/Robin forward results", ...
    'Color', 'white', ...
    'Position', [80, 60, 1500, 880]);
layout = tiledlayout(2, 3, ...
    'TileSpacing', 'compact', 'Padding', 'compact');

body_values = {classic_body, robin_body, body_delta};
body_titles_zh = [ ...
    "Classic 体电势", ...
    "Robin 体电势", ...
    "体电势差值 Robin - Classic"];
body_titles_en = [ ...
    "Classic body potential", ...
    "Robin body potential", ...
    "Body-potential difference"];
for panel = 1:3
    result_axis = nexttile(layout);
    patch( ...
        result_axis, ...
        'Faces', elements, ...
        'Vertices', nodes, ...
        'FaceVertexCData', body_values{panel}, ...
        'FaceColor', 'interp', ...
        'EdgeColor', [0.78, 0.82, 0.88], ...
        'LineWidth', 0.55);
    axis(result_axis, 'equal');
    xlim(result_axis, [-1.08, 1.08]);
    ylim(result_axis, [-1.08, 1.08]);
    xlabel(result_axis, "x", 'FontName', 'Times New Roman');
    ylabel(result_axis, "y", 'FontName', 'Times New Roman');
    title(result_axis, body_titles_zh(panel), ...
        'FontName', 'Microsoft YaHei');
    subtitle(result_axis, body_titles_en(panel), ...
        'FontName', 'Times New Roman');
    set(result_axis, 'FontName', 'Times New Roman');
    if panel < 3
        colormap(result_axis, parula);
        clim(result_axis, [body_min, body_max]);
    else
        colormap(result_axis, signed_blue_orange_colormap(257));
        clim(result_axis, [-body_delta_limit, body_delta_limit]);
    end
    colorbar(result_axis);
end

electrode_indices = 1:size(classic_voltage_all, 1);
classic_axis = nexttile(layout);
plot(classic_axis, electrode_indices, classic_voltage, ...
    '-o', 'Color', [0.85, 0.47, 0.02], ...
    'MarkerSize', 4.5, 'LineWidth', 1.8);
yline(classic_axis, 0, 'Color', [0.58, 0.64, 0.72]);
ylim(classic_axis, [ ...
    voltage_min - voltage_padding, voltage_max + voltage_padding]);
title(classic_axis, "Classic 电极电压", ...
    'FontName', 'Microsoft YaHei');
subtitle(classic_axis, "Classic electrode voltage", ...
    'FontName', 'Times New Roman');

robin_axis = nexttile(layout);
plot(robin_axis, electrode_indices, robin_voltage, ...
    '--s', 'Color', [0.11, 0.31, 0.72], ...
    'MarkerFaceColor', 'white', ...
    'MarkerSize', 4.5, 'LineWidth', 1.8);
yline(robin_axis, 0, 'Color', [0.58, 0.64, 0.72]);
ylim(robin_axis, [ ...
    voltage_min - voltage_padding, voltage_max + voltage_padding]);
title(robin_axis, "Robin 电极电压", ...
    'FontName', 'Microsoft YaHei');
subtitle(robin_axis, "Robin electrode voltage", ...
    'FontName', 'Times New Roman');

delta_axis = nexttile(layout);
plot(delta_axis, electrode_indices, voltage_delta, ...
    '-d', 'Color', [0.49, 0.23, 0.93], ...
    'MarkerFaceColor', 'white', ...
    'MarkerSize', 4.2, 'LineWidth', 1.6);
yline(delta_axis, 0, 'Color', [0.28, 0.35, 0.44]);
ylim(delta_axis, [ ...
    -1.12 * voltage_delta_limit, 1.12 * voltage_delta_limit]);
title(delta_axis, "电极电压差值 Robin - Classic", ...
    'FontName', 'Microsoft YaHei');
subtitle(delta_axis, "Electrode-voltage difference", ...
    'FontName', 'Times New Roman');

for voltage_axis = [classic_axis, robin_axis, delta_axis]
    xlabel(voltage_axis, "电极编号 / Electrode index", ...
        'FontName', 'Microsoft YaHei');
    ylabel(voltage_axis, "电压 / Voltage", ...
        'FontName', 'Microsoft YaHei');
    xticks(voltage_axis, electrode_indices);
    grid(voltage_axis, 'on');
    set(voltage_axis, 'FontName', 'Times New Roman');
end

positive = find(currents(:, drive_index) > 0);
negative = find(currents(:, drive_index) < 0);
sgtitle(layout, sprintf( ...
    ['%s Classic/Robin 体电势与边界电压 | ', ...
     'drive %d, +I E%s, -I E%s'], ...
    case_id, drive_index, ...
    join(string(positive), ","), join(string(negative), ",")), ...
    'FontName', 'Microsoft YaHei');
end


function map = signed_blue_orange_colormap(count)
blue = [0.11, 0.31, 0.72];
white = [1.00, 1.00, 1.00];
orange = [0.85, 0.47, 0.02];
left_count = ceil(count / 2);
right_count = count - left_count + 1;
left = [ ...
    linspace(blue(1), white(1), left_count).', ...
    linspace(blue(2), white(2), left_count).', ...
    linspace(blue(3), white(3), left_count).'];
right = [ ...
    linspace(white(1), orange(1), right_count).', ...
    linspace(white(2), orange(2), right_count).', ...
    linspace(white(3), orange(3), right_count).'];
map = [left; right(2:end, :)];
end


function Q = helmert_zero_sum_basis(electrode_count)
Q = zeros(electrode_count, electrode_count - 1);
for column = 1:(electrode_count - 1)
    scale = sqrt(column * (column + 1));
    Q(1:column, column) = 1 / scale;
    Q(column + 1, column) = -column / scale;
end
end


function value = relative_l2(candidate, reference)
value = norm(candidate - reference, "fro") / ...
    max(norm(reference, "fro"), eps);
end


function value = scaled_backward_residual(matrix, solution, rhs)
numerator = norm(matrix * solution - rhs, "fro");
denominator = norm(matrix, "fro") * norm(solution, "fro") + ...
    norm(rhs, "fro");
value = numerator / max(denominator, eps);
end


function gnd_node = choose_ground_node( ...
    nodes, electrode_nodes, electrode_counts)
mask = false(size(nodes, 1), 1);
for index = 1:numel(electrode_counts)
    active = electrode_nodes(index, 1:electrode_counts(index));
    active = active(active > 0);
    mask(active) = true;
end
free_nodes = find(~mask);
if isempty(free_nodes)
    gnd_node = 1;
else
    [~, local_index] = min(sum(nodes(free_nodes, :) .^ 2, 2));
    gnd_node = free_nodes(local_index);
end
end
