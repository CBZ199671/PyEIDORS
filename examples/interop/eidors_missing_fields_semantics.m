%% EIDORS missing-field source-semantics example
% This model is intentionally invalid under valid_fwd_model because required
% source fields are removed. The capture must preserve that fact instead of
% inventing contact impedance or source-field values.

imdl = mk_common_model('c2C2', 8);
fmdl = imdl.fwd_model;
fmdl.name = 'EIDORS missing-field semantics example';
fmdl.stimulation = mk_stim_patterns( ...
    numel(fmdl.electrode), 1, '{ad}', '{ad}', ...
    {'no_meas_current'}, 0.01);
fmdl.electrode = rmfield(fmdl.electrode, 'z_contact');
fmdl = rmfield(fmdl, 'gnd_node');
fmdl = rmfield(fmdl, 'normalize_measurements');

img_bg = mk_image(fmdl, 1.5);
img_truth = img_bg;
img_truth.elem_data(1) = 2.5;
