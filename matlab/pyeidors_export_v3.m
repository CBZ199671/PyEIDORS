function package_dir = pyeidors_export_v3(source, out_dir, varargin)
%PYEIDORS_EXPORT_V3 Export an EIDORS object as a Bridge Package v3.
%
% package_dir = pyeidors_export_v3(fmdl, out_dir)
% package_dir = pyeidors_export_v3(inv_model, out_dir, ...
%     'Background', img_h, 'Target', img_i)
%
% The function uses the public pyeidors-interop CLI and a fresh MATLAB
% process.  Pass 'EidorsStartup' when it cannot be discovered automatically.

parser = inputParser;
parser.addRequired('source', @(value) isstruct(value) && isscalar(value));
parser.addRequired('out_dir', @(value) ischar(value) || isstring(value));
parser.addParameter('Background', [], @(value) isempty(value) || ...
    isnumeric(value) || (isstruct(value) && isscalar(value)));
parser.addParameter('Target', [], @(value) isempty(value) || ...
    isnumeric(value) || (isstruct(value) && isscalar(value)));
parser.addParameter('EidorsStartup', '', @(value) ischar(value) || isstring(value));
parser.addParameter('Matlab', fullfile(matlabroot, 'bin', 'matlab'), ...
    @(value) ischar(value) || isstring(value));
parser.addParameter('Cli', 'pyeidors-interop', ...
    @(value) ischar(value) || isstring(value));
parser.parse(source, out_dir, varargin{:});
options = parser.Results;

source_kind = local_source_kind(source);
eidors_startup = char(options.EidorsStartup);
if isempty(eidors_startup)
    eidors_startup = which('eidors_startup');
end
if isempty(eidors_startup) || exist(eidors_startup, 'file') ~= 2
    error(['EIDORS startup was not discovered. Pass ', ...
        '''EidorsStartup'', ''<path-to-eidors/startup.m>''.']);
end

package_dir = char(out_dir);
temporary_dir = tempname;
mkdir(temporary_dir);
cleanup = onCleanup(@() local_remove_temp(temporary_dir)); %#ok<NASGU>
source_mat = fullfile(temporary_dir, 'bridge_source.mat');
driver_script = fullfile(temporary_dir, 'bridge_source_driver.m');
bridge_source = source; %#ok<NASGU>
bridge_background = options.Background; %#ok<NASGU>
bridge_target = options.Target; %#ok<NASGU>
save(source_mat, 'bridge_source', 'bridge_background', 'bridge_target', ...
    'source_kind', '-v7');
local_write_driver(driver_script, source_mat);

command_parts = { ...
    local_quote(char(options.Cli)), ...
    'capture', local_quote(driver_script), ...
    '--output', local_quote(package_dir), ...
    '--matlab', local_quote(char(options.Matlab)), ...
    '--eidors-startup', local_quote(eidors_startup), ...
    '--fwd-model-var', local_quote('fmdl')};
if ~isempty(options.Background)
    command_parts(end + 1:end + 2) = { ...
        '--background-image-var', local_quote('img_bg')};
end
if strcmp(source_kind, 'image') || ~isempty(options.Target)
    command_parts(end + 1:end + 2) = { ...
        '--target-image-var', local_quote('img_target')};
end
command = strjoin(command_parts, ' ');
[status, output] = system(command);
if status ~= 0
    error('PyEIDORS Bridge v3 export failed:\n%s', output);
end
if exist(fullfile(package_dir, 'manifest.json'), 'file') ~= 2
    error('Bridge v3 export returned without manifest.json:\n%s', output);
end
end

function kind = local_source_kind(source)
kind = '';
if isfield(source, 'type') && (ischar(source.type) || isstring(source.type))
    kind = char(source.type);
end
if strcmp(kind, 'fwd_model') || ...
        (isfield(source, 'nodes') && isfield(source, 'elems'))
    kind = 'fwd_model';
elseif strcmp(kind, 'inv_model')
    if ~isfield(source, 'fwd_model')
        error('The EIDORS inv_model has no fwd_model.');
    end
    kind = 'inv_model';
elseif strcmp(kind, 'image') || ...
        (isfield(source, 'fwd_model') && any(isfield(source, ...
        {'elem_data', 'node_data', 'conductivity', 'resistivity'})))
    kind = 'image';
else
    error('source must be an EIDORS fwd_model, inv_model, or image.');
end
end

function local_write_driver(script_path, source_mat)
source_literal = strrep(source_mat, '''', '''''');
lines = {
    sprintf('load(''%s'');', source_literal)
    'img_bg = [];'
    'img_target = [];'
    'switch source_kind'
    '    case ''fwd_model'''
    '        fmdl = bridge_source;'
    '    case ''inv_model'''
    '        inv_model = bridge_source;'
    '        fmdl = inv_model.fwd_model;'
    '    case ''image'''
    '        img_target = bridge_source;'
    '        fmdl = img_target.fwd_model;'
    '    otherwise'
    '        error(''Unsupported bridge source kind.'');'
    'end'
    'if ~isempty(bridge_background)'
    '    if isnumeric(bridge_background)'
    '        img_bg = mk_image(fmdl, bridge_background);'
    '    else'
    '        img_bg = bridge_background;'
    '    end'
    'end'
    'if ~isempty(bridge_target)'
    '    if isnumeric(bridge_target)'
    '        img_target = mk_image(fmdl, bridge_target);'
    '    else'
    '        img_target = bridge_target;'
    '    end'
    'end'
    };
fid = fopen(script_path, 'w');
if fid < 0
    error('Could not create the temporary Bridge v3 driver script.');
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', lines{:});
end

function quoted = local_quote(value)
if contains(value, '"')
    error('Bridge command arguments cannot contain a double-quote character.');
end
quoted = ['"', value, '"'];
end

function local_remove_temp(path)
if exist(path, 'dir') == 7
    rmdir(path, 's');
end
end
