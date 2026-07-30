function imported = pyeidors_import_v3(package_dir, varargin)
%PYEIDORS_IMPORT_V3 Validate and load a Bridge Package v3 into EIDORS.
%
% imported = pyeidors_import_v3(package_dir)
% imported.fwd_model
% imported.background_image
% imported.target_image
% imported.homogeneous_data
% imported.target_data

parser = inputParser;
parser.addRequired('package_dir', @(value) ischar(value) || isstring(value));
parser.addParameter('Cli', 'pyeidors-interop', ...
    @(value) ischar(value) || isstring(value));
parser.parse(package_dir, varargin{:});
package_dir = char(parser.Results.package_dir);

manifest_path = fullfile(package_dir, 'manifest.json');
if exist(manifest_path, 'file') ~= 2
    error('Bridge v3 manifest.json was not found in %s.', package_dir);
end
manifest = jsondecode(fileread(manifest_path));
if ~isfield(manifest, 'exchange_format') || ...
        ~strcmp(char(manifest.exchange_format), 'eidors_pyeidors_bridge_v3')
    error('Only eidors_pyeidors_bridge_v3 packages are supported.');
end

command = strjoin({local_quote(char(parser.Results.Cli)), ...
    'validate', local_quote(package_dir)}, ' ');
[status, output] = system(command);
if status ~= 0
    error('Bridge v3 integrity validation failed:\n%s', output);
end

loader = fullfile(package_dir, 'run_in_eidors.m');
if exist(loader, 'file') ~= 2
    error(['run_in_eidors.m is missing. Re-export the package with ', ...
        'include_scripts enabled.']);
end
run(loader);

imported = struct();
imported.fwd_model = fmdl;
imported.background_image = local_workspace_value('img_background');
imported.target_image = local_workspace_value('img_target');
imported.homogeneous_data = local_workspace_value('vh');
imported.target_data = local_workspace_value('vi');
imported.model_id = char(manifest.model_id);
imported.forward_fingerprint = char(manifest.forward_fingerprint);
imported.protocol_layout_hash = char(manifest.protocol_layout_hash);
imported.protocol_physics_hash = char(manifest.protocol_physics_hash);
end

function value = local_workspace_value(name)
if evalin('caller', sprintf('exist(''%s'', ''var'')', name))
    value = evalin('caller', name);
else
    value = [];
end
end

function quoted = local_quote(value)
if contains(value, '"')
    error('Bridge command arguments cannot contain a double-quote character.');
end
quoted = ['"', value, '"'];
end
