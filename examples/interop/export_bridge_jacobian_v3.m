function jacobian = export_bridge_jacobian_v3(run_script, output_path)
%EXPORT_BRIDGE_JACOBIAN_V3 Export a Bridge v3 background Jacobian from EIDORS.
%
% jacobian = export_bridge_jacobian_v3(run_script, output_path)
%
% `run_script` is the generated Bridge v3 `run_in_eidors.m`. The helper
% reconstructs the exact EIDORS forward model, evaluates the canonical
% background Jacobian in the source P1 element-conductivity parameter space,
% and writes a MATLAB v7 MAT file for the cross-runtime acceptance gate.

if nargin < 2 || isempty(output_path)
    error('A Bridge v3 run script and output MAT path are required.');
end
run_script = char(run_script);
output_path = char(output_path);
if exist(run_script, 'file') ~= 2
    error('Bridge v3 run script was not found: %s', run_script);
end

run(run_script);
if ~exist('fmdl', 'var') || ~exist('img_background', 'var')
    error('Bridge v3 import did not create fmdl and img_background.');
end

jacobian = calc_jacobian(img_background);
if size(jacobian, 2) ~= size(fmdl.elems, 1)
    error(['EIDORS Jacobian parameter columns do not match the source ', ...
           'P1 element-conductivity space.']);
end
save(output_path, 'jacobian', '-v7');
fprintf('EIDORS Bridge v3 Jacobian exported to %s\n', output_path);
end
