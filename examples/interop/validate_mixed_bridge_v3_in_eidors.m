function report = validate_mixed_bridge_v3_in_eidors(package_dir, report_path)
%VALIDATE_MIXED_BRIDGE_V3_IN_EIDORS Validate exact logical PEM expansion.

if nargin < 2
    report_path = '';
end
run(fullfile(package_dir, 'run_in_eidors.m'));

mapping = full(fmdl.pyeidors_bridge.logical_to_physical);
expected_mapping = [1, 0; 0, 0.25; 0, 0.75];
if ~isequal(size(mapping), size(expected_mapping)) || ...
        norm(mapping - expected_mapping, 'fro') > 1e-14
    error('Logical-to-physical weighted PEM mapping is incorrect.');
end
if numel(fmdl.electrode) ~= 3 || ...
        fmdl.pyeidors_bridge.logical_electrode_count ~= 2
    error('Weighted PEM was not expanded to the expected EIDORS electrodes.');
end

eidors_homogeneous = fwd_solve(img_background);
eidors_target = fwd_solve(img_target);
homogeneous_rel_l2 = local_rel_l2(eidors_homogeneous.meas, vh.meas);
target_rel_l2 = local_rel_l2(eidors_target.meas, vi.meas);
if homogeneous_rel_l2 > 5e-4 || target_rel_l2 > 5e-4
    error('Mixed Bridge v3 voltage parity exceeded relL2 <= 5e-4.');
end

runtime = fwd_model_parameters(fmdl, 'skip_VOLUME');
logical_n2e = mapping' * full(runtime.N2E);
report = struct();
report.schema = 'pyeidors_mixed_bridge_v3_eidors_acceptance_v1';
report.logical_electrodes = 2;
report.physical_electrodes = numel(fmdl.electrode);
report.logical_to_physical = mapping;
report.logical_n2e = logical_n2e;
report.homogeneous_relative_l2 = homogeneous_rel_l2;
report.target_relative_l2 = target_rel_l2;
report.passed = true;

if ~isempty(report_path)
    fid = fopen(report_path, 'w');
    if fid < 0
        error('Could not create mixed Bridge v3 acceptance report.');
    end
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%s', jsonencode(report, PrettyPrint=true));
end
end

function value = local_rel_l2(actual, expected)
actual = actual(:);
expected = expected(:);
value = norm(actual - expected) / max(norm(expected), eps);
end
