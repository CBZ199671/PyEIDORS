%% EIDORS multiple-model discovery example
% The capture must reject this workspace unless fwd_model_var is explicit.

imdl_a = mk_common_model('c2C2', 8);
fmdl_a = imdl_a.fwd_model;
fmdl_a.name = 'EIDORS ambiguous model A (8 electrodes)';

imdl_b = mk_common_model('c2C2', 16);
fmdl_b = imdl_b.fwd_model;
fmdl_b.name = 'EIDORS ambiguous model B (16 electrodes)';
