%% EIDORS entry point for preregistered exact-rational extension cases.
% compare_cem_formulations consumes truth_elem_data/current_patterns,
% verifies conductivity_digest metadata, and exports assembled_blocks.
script_dir = fileparts(mfilename('fullpath'));
run(fullfile(script_dir, 'compare_cem_formulations.m'));
