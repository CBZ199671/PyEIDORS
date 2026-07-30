# Bridge v3 examples

这里只包含 `eidors_pyeidors_bridge_v3` 示例；不再提供 v1/v2 或独立
`geometry.mat` 兼容路线。

## EIDORS → PyEIDORS

- `eidors_2d_quickstart.m`：2D CEM。
- `eidors_3d_quickstart.m`：3D 外部 CEM。
- `eidors_3d_point_electrode_quickstart.m`：3D 原生 PEM。
- `eidors_multiple_models_requires_selector.m`：多候选与显式变量选择。
- `eidors_missing_fields_semantics.m`：缺失/阻断字段的 fail-closed 示例。

```bash
nix develop .#complex64-cuda --command pyeidors-interop capture \
  examples/interop/eidors_3d_point_electrode_quickstart.m \
  --output output/eidors_pem_v3 \
  --matlab matlab \
  --eidors-startup /path/to/eidors/startup.m

nix develop .#complex64-cuda --command pyeidors-interop validate \
  output/eidors_pem_v3
nix develop .#complex64-cuda --command pyeidors-interop verify-numerics \
  output/eidors_pem_v3
```

多候选时追加 `--fwd-model-var`、`--background-image-var` 和
`--target-image-var` 等完整路径选择器。

## PyEIDORS → EIDORS

- `pyeidors_3d_export.py`：原生 PyEIDORS 3D 模型。
- `pyeidors_mixed_export.py`：CEM + 多节点加权 PEM 混合模型。
- `validate_bridge_in_eidors.m`：在真实 EIDORS 中验证常规模型。
- `validate_mixed_bridge_v3_in_eidors.m`：验证混合电极模型。

```bash
nix develop .#complex64-cuda --command python \
  examples/interop/pyeidors_mixed_export.py \
  --output output/pyeidors_mixed_v3 \
  --eidors-startup /path/to/eidors/startup.m
```

MATLAB：

```matlab
run('/path/to/eidors/startup.m');
addpath('/path/to/PyEIDORS/matlab');
imported = pyeidors_import_v3('/path/to/output/pyeidors_mixed_v3');
```

完整 GUI、CLI、协议、电极和实时映射说明见
[`docs/EIDORS_PYEIDORS_INTEROP.md`](../../docs/EIDORS_PYEIDORS_INTEROP.md)。
