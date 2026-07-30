function run_eidors_38_cases(repo_root, eidors_startup, timing_repeats)
%RUN_EIDORS_38_CASES 运行全部认证案例 / Run all certified EIDORS cases.
%
% 示例 / Example:
%   run_eidors_38_cases( ...
%       "\\wsl.localhost\Ubuntu-22.04\home\tom\workspace\PyEidors_wsl2", ...
%       "C:\eidors-v3.12-ng\eidors\startup.m", ...
%       11);

if nargin < 1 || strlength(string(repo_root)) == 0
    package_dir = fileparts(mfilename("fullpath"));
    repo_root = fileparts(fileparts(package_dir));
end
if nargin < 2
    eidors_startup = "";
end
if nargin < 3
    timing_repeats = 11;
end
if timing_repeats < 3
    error("Fair timing requires at least three repetitions.");
end

if exist("eidors_default", "file") ~= 2
    if strlength(string(eidors_startup)) == 0 || ...
            exist(eidors_startup, "file") ~= 2
        error("Set eidors_startup to a valid EIDORS startup.m file.");
    end
    run(eidors_startup);
end

suite_output = fullfile(repo_root, "output", "cem_exact_extension");
manifest_path = fullfile(suite_output, "suite_manifest.json");
if exist(manifest_path, "file") ~= 2
    error([ ...
        "Suite manifest not found. Run the PyEIDORS prepare step first: ", ...
        manifest_path ...
    ]);
end
manifest = jsondecode(fileread(manifest_path));
entrypoint = fullfile( ...
    repo_root, "compare_with_Eidors", "compare_cem_exact_extension.m");
if exist(entrypoint, "file") ~= 2
    error("EIDORS benchmark entry point not found: %s", entrypoint);
end

old_output = getenv("CEM_BENCHMARK_OUTPUT_DIR");
old_mesh = getenv("CEM_COMMON_MESH_MAT");
old_repeats = getenv("CEM_TIMING_REPEATS");
cleanup = onCleanup(@() restore_environment( ...
    old_output, old_mesh, old_repeats));

for index = 1:numel(manifest.cases)
    fixture = manifest.cases(index);
    case_name = string(fixture.case_id) + "_" + string(fixture.label);
    case_dir = fullfile(suite_output, "cases", case_name);
    mesh_mat = fullfile( ...
        case_dir, "common_mesh", "cem_exact_extension_p1.mat");
    if exist(mesh_mat, "file") ~= 2
        error("Missing common MAT for %s: %s", fixture.case_id, mesh_mat);
    end

    setenv("CEM_BENCHMARK_OUTPUT_DIR", case_dir);
    setenv("CEM_COMMON_MESH_MAT", mesh_mat);
    setenv("CEM_TIMING_REPEATS", string(timing_repeats));
    fprintf( ...
        "[%02d/%02d] EIDORS %s\n", ...
        index, numel(manifest.cases), fixture.case_id);

    % 既有脚本以 clear 开头，因此在 base 工作区运行，避免清除本函数的循环变量
    % 和环境恢复状态。 / The existing script starts with clear, so run it in
    % the base workspace to isolate this function's loop and cleanup state.
    command = "run('" + escape_matlab_path(entrypoint) + "');";
    evalin("base", command);
end
fprintf("Completed %d EIDORS cases in %s\n", ...
    numel(manifest.cases), suite_output);
end


function escaped = escape_matlab_path(path)
escaped = replace(string(path), "'", "''");
end


function restore_environment(output, mesh, repeats)
setenv("CEM_BENCHMARK_OUTPUT_DIR", output);
setenv("CEM_COMMON_MESH_MAT", mesh);
setenv("CEM_TIMING_REPEATS", repeats);
end
