function count = run_cem_exact_extension_suite(manifest_path)
%% Run every preregistered rational extension fixture in one EIDORS process.
if nargin < 1 || isempty(manifest_path)
    manifest_path = getenv('CEM_EXACT_EXTENSION_MANIFEST');
end
if isempty(manifest_path) || exist(manifest_path, 'file') ~= 2
    error('Extension manifest not found: %s', manifest_path);
end
manifest = jsondecode(fileread(manifest_path));
fixtures = manifest.cases;
script_dir = fileparts(mfilename('fullpath'));
case_script = fullfile(script_dir, 'compare_cem_exact_extension.m');
escaped_case_script = strrep(case_script, '''', '''''');
for index = 1:numel(fixtures)
    output_dir = wsl_path_to_unc(fixtures(index).case_dir);
    mesh_mat = wsl_path_to_unc(fixtures(index).mat);
    setenv('CEM_BENCHMARK_OUTPUT_DIR', output_dir);
    setenv('CEM_COMMON_MESH_MAT', mesh_mat);
    evalin('base', sprintf('run(''%s'')', escaped_case_script));
end
count = numel(fixtures);
fprintf('EIDORS extension reports: %d cases\n', count);
end


function converted = wsl_path_to_unc(path_value)
converted = char(path_value);
prefix = '/home/';
if startsWith(converted, prefix)
    converted = ['\\wsl.localhost\Ubuntu-22.04', strrep(converted, '/', '\')];
end
end
