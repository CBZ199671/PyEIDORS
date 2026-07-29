%% EIDORS 3D point-electrode source-semantics example
% This example intentionally exercises:
%   - one-node EIDORS point electrodes (PEM);
%   - resistivity image parametrization converted by EIDORS at solve time;
%   - fwd_model.current_density scaling of the raw stimulation vectors.

imdl = mk_common_model('a3cr', 16);
fmdl = imdl.fwd_model;
fmdl.name = 'EIDORS 3D point-electrode semantics quickstart';
fmdl.stimulation = mk_stim_patterns( ...
    numel(fmdl.electrode), 1, '{ad}', '{ad}', ...
    {'no_meas_current'}, 0.02);
fmdl.current_density = 2.0;

n_elem = size(fmdl.elems, 1);
img_bg = mk_image(fmdl, 0.5 * ones(n_elem, 1), 'resistivity');
img_truth = img_bg;
img_truth.resistivity.elem_data(1) = 0.25;
img_bg = data_mapper(img_bg);
img_truth = data_mapper(img_truth);

vh = fwd_solve(img_bg);
vi = fwd_solve(img_truth);
