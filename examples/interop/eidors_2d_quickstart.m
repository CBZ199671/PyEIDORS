%% Ordinary EIDORS 2D model for Bridge Package v2 capture
imdl = mk_common_model('c2c2', 16);
fmdl = imdl.fwd_model;
fmdl.name = 'EIDORS 2D quickstart';
fmdl.stimulation = mk_stim_patterns( ...
    numel(fmdl.electrode), 1, '{ad}', '{ad}', ...
    {'no_meas_current'}, 1.0);

img_bg = mk_image(fmdl, 1.0);
img_truth = img_bg;
select_fun = inline('(x-0.25).^2 + (y+0.10).^2 < 0.18^2', 'x', 'y', 'z');
target_mask = elem_select(fmdl, select_fun);
img_truth.elem_data = 1.0 + target_mask;

vh = fwd_solve(img_bg);
vi = fwd_solve(img_truth);
