function count = run_cem_exact_suite(manifest_path)
%% Run every prepared rational exact-suite fixture in one MATLAB/EIDORS process.
if nargin < 1 || isempty(manifest_path)
    manifest_path = getenv('CEM_EXACT_MANIFEST');
end
if isempty(manifest_path) || exist(manifest_path, 'file') ~= 2
    error('Exact suite manifest not found: %s', manifest_path);
end
manifest = jsondecode(fileread(manifest_path));
fixtures = manifest.cases;
script_dir = fileparts(mfilename('fullpath'));
case_script = fullfile(script_dir, 'compare_cem_formulations.m');
escaped_case_script = strrep(case_script, '''', '''''');
for index = 1:numel(fixtures)
    output_dir = wsl_path_to_unc(fixtures(index).case_dir);
    mesh_mat = wsl_path_to_unc(fixtures(index).mat);
    setenv('CEM_BENCHMARK_OUTPUT_DIR', output_dir);
    setenv('CEM_COMMON_MESH_MAT', mesh_mat);
    % The legacy case script starts with `clear`; isolate that in base so the
    % manifest loop remains intact in this function workspace.
    evalin('base', sprintf('run(''%s'')', escaped_case_script));
end
count = numel(fixtures);
fprintf('EIDORS exact CEM reports: %d cases\n', count);
end


function converted = wsl_path_to_unc(path_value)
converted = char(path_value);
prefix = '/home/';
if startsWith(converted, prefix)
    converted = ['\\wsl.localhost\Ubuntu-22.04', strrep(converted, '/', '\')];
end
end
