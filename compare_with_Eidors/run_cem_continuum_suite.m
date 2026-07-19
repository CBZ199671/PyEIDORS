function count = run_cem_continuum_suite(manifest_path)
%% Run every prepared true-circle fixture in one MATLAB/EIDORS process.
if nargin < 1 || isempty(manifest_path)
    manifest_path = getenv('CEM_CONTINUUM_MANIFEST');
end
if isempty(manifest_path) || exist(manifest_path, 'file') ~= 2
    error('Continuum suite manifest not found: %s', manifest_path);
end
manifest = jsondecode(fileread(manifest_path));
fixtures = manifest.fixtures;
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
for index = 1:numel(fixtures)
    output_dir = wsl_path_to_unc(fixtures(index).case_dir);
    mesh_mat = wsl_path_to_unc(fixtures(index).mat);
    compare_cem_continuum(output_dir, mesh_mat);
end
count = numel(fixtures);
fprintf('EIDORS continuum reports: %d fixtures\n', count);
end


function converted = wsl_path_to_unc(path_value)
converted = char(path_value);
prefix = '/home/';
if startsWith(converted, prefix)
    converted = ['\\wsl.localhost\Ubuntu-22.04', strrep(converted, '/', '\')];
end
end
