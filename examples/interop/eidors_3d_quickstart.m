%% Ordinary EIDORS 3D surface-electrode model for Bridge Package v2 capture
[fmdl, ~] = ng_mk_cyl_models([1.0, 1.0, 0.35], [8, 0.5], [0.12]);
fmdl.name = 'EIDORS 3D surface-electrode quickstart';
fmdl.stimulation = mk_stim_patterns( ...
    numel(fmdl.electrode), 1, '{ad}', '{ad}', ...
    {'no_meas_current'}, 1.0);

img_bg = mk_image(fmdl, 1.0);
img_truth = img_bg;
select_fun = inline( ...
    '(x-0.20).^2 + (y+0.10).^2 + (z-0.55).^2 < 0.15^2', ...
    'x', 'y', 'z');
target_mask = elem_select(fmdl, select_fun);
img_truth.elem_data = 1.0 + target_mask;

vh = fwd_solve(img_bg);
vi = fwd_solve(img_truth);
