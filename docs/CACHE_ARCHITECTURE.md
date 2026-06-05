# Cache Architecture (EIDORS-Style Semantic Cache)

PyEIDORS uses a two-layer cache inspired by EIDORS `eidors_cache`, with semantic
dependency signatures, deterministic invalidation, and rank-aware eviction.

## Layers

1. **L1 Process cache**
   - In-memory rank-aware store.
   - `score_eff = round(10*log10(effort * use_count)) + priority`
   - `score_size = round(10*log10(size_bytes / 1024))`
   - Eviction removes low-priority entries first via key:
     - retention rank: `(-score_eff, score_size, -last_access)`
     - eviction rank: `(score_eff, -score_size, last_access)`
   - Optimized for repeated solves in one Python process.
   - Resident size estimation recognizes NumPy arrays, sparse-matrix payload
     arrays, containers, and simple object attributes before falling back to
     pickle sizing, avoiding full serialized copies for large array-backed cache
     values.
   - Process-cache admission is checked before mutating the LRU: entries larger
     than the whole process budget, or entries that would be evicted
     immediately by score-aware eviction, are returned to the caller but not
     retained. Stats expose `process_admission_rejections`,
     `process_admission_rejected_bytes`, and rejection reasons so oversized 3D
     objects are visible without flushing hotter L1 entries first.
   - Default size budget: `3 GB`.

2. **L2 Disk cache**
   - Runtime object store rooted at `.pyeidors_cache/v2`. By default, supported dev shells place the effective disk cache under `.pyeidors_cache/v2/.sessions/<session-id>`.
   - sqlite index (`index.sqlite`) + object payload files.
   - Maintains `name/namespace/effort/use_count/priority/score_eff/score_size/score` metadata.
   - Object payloads stream through `pickle.dump/load` directly on the target file handle, optionally wrapped in gzip, so large Jacobian/RM cache entries do not create extra whole-payload `pickle.dumps`/`Path.read_bytes` copies on disk put/get.
   - Default size budget: `20 GB`. Session caches are terminal-scoped by default and are cleaned automatically when the owning `nix develop` shell exits; use `cache_lifecycle="persistent"` to opt into long-lived cross-terminal storage.

## 3D First-Load Observability

GUI forward results include phase timing metadata so cold 3D runs can be
attributed before adding more cache layers:

- `forward_timing_schema = eit_app_forward_timing_v1`
- `forward_timing_ms` and `forward_timing_phase_order`
- `forward_timing_total_ms`
- worker transport timings when an external backend is used:
  `backend_worker_request_write_ms`, `backend_worker_request_duration_ms` or
  `backend_worker_subprocess_duration_ms`, and `backend_worker_result_read_ms`
- GUI render/update timing:
  `gui_forward_visualization_update_ms`

Backend worker HDF5 IPC arrays use fast `lzf` by default and enable the HDF5
shuffle filter for numeric arrays, which reduces repeated typed-byte patterns in
large 3D coordinates, connectivity, conductivity, and voltage payloads without
switching to CPU-heavy gzip. `EIT_APP_BACKEND_WORKER_HDF5_COMPRESSION=off`
disables compression entirely, and `EIT_APP_BACKEND_WORKER_HDF5_SHUFFLE=off`
keeps `lzf` while disabling shuffle for low-level HDF5 debugging.
Compressed numeric datasets use explicit row-major chunks targeting 1MiB by
default. Set `EIT_APP_BACKEND_WORKER_HDF5_CHUNK_BYTES=<bytes>` to tune the target
chunk size, or set it to `off` to let HDF5 choose chunks.
Readers use HDF5 `read_direct` for non-scalar numeric datasets so large arrays
land directly in their final C-order NumPy buffers.
Optional result arrays are omitted when absent (`homogeneous_voltages`,
`measured`, and `simulated`) instead of being written as empty placeholder
datasets; readers remain compatible with older files that contain those
placeholders.
Simulation result views preserve `float32`/`complex64` display channel precision
instead of widening single-precision conductivity arrays to `float64` solely for
visualization.
Complex-mode detection scans imaginary components in bounded chunks, so enabling
the GUI channel selector does not copy the full finite imaginary part of a large
3D `complex64` conductivity array.
The same scan reuses bounded finite and absolute-value work buffers, avoiding
per-chunk `isfinite`/`abs` temporaries while preserving non-finite exclusion and
tolerance-based early exit.
The composite amplitude/phase display channel is also chunked; it produces only
the final display array plus a bounded work buffer instead of full magnitude and
phase temporaries.
3D anomaly highlighting and point-cloud sampling reuse the residual score buffer
for positive, negative, and absolute anomaly modes, avoiding another full
cell-count score array during large 3D display updates.
Positive/negative robust MAD thresholding also uses that same buffer: the score
array is temporarily converted to absolute residuals in-place for the median,
then restored to the signed mode score without allocating a second full abs
array.
Crowded anomaly percentile tightening reuses the existing finite mask as an
invalid-position work mask and marks invalid score entries as NaN in-place, so
the crowded branch no longer copies `score[finite_values]` before percentile
calculation.
When a large anomaly region already exceeds the point-cloud display cap, anomaly
indices are sampled by rank in bounded chunks instead of materializing every
true index before capping.
When all remaining background points fit the point-cloud display budget,
background indices are direct-filled from the anomaly mask instead of building a
full inverted mask for `flatnonzero`.
That background sampler trusts the already materialized full anomaly-index
count, so rare-anomaly point-cloud first draw avoids a second whole-mask
`count_nonzero` pass before filling evenly spaced background points.
All-retained true/anomaly indices use the same direct-fill pattern, reserving
chunked `flatnonzero` only for bounded chunk sampling when anomaly count exceeds
the display cap.
Point-cloud highlight rendering then uses one helper to direct-fill highlighted
centers/sigma from the bool mask, rather than repeatedly boolean-indexing the
same display arrays in each renderer or materializing a full `flatnonzero`
highlight index vector.
3D electrode overlay patch geometry direct-fills both point coordinates and
triangle indices into final arrays, avoiding tuple-list staging before PyVista
polydata construction.
The 3D conductivity widget no longer provides a Matplotlib 3D electrode
fallback; electrode overlays use PyVista polydata built from direct-filled patch
and face buffers.
Conductivity color limits and anomaly-mask setup scan finite status in bounded
chunks; the common all-finite 3D payload path does not allocate a full finite
boolean mask, while NaN/Inf payloads still materialize one mask for the legacy
non-finite handling path.
When 3D color limits do encounter non-finite values, median calculation marks
invalid entries as NaN in a private value copy and avoids `values[finite_mask]`
finite-subset copies.
The anomaly-mask candidate bool array is also reused as the final threshold mask
through an opt-in helper path, while direct score/peak helper calls keep their
count/peak return contract. Non-finite filtering reuses the existing finite mask
instead of allocating a second `isfinite(score)` array.
Dual-mesh coarse-to-fine cell lookup reuses a one-dimensional bounding-box
candidate mask per fine point instead of creating a `(n_cells, dim)` boolean
comparison matrix before candidate simplex checks.
Candidate simplex checks iterate that mask directly, so the locator also avoids
allocating a `flatnonzero` candidate-index vector for every fine point.
Candidate simplex vertices are copied into one reusable work buffer instead of
rebuilding a `coords[cells[cell_idx]]` gather array for each candidate.
VoxelGrid coarse meshes use the same low-copy style: inside/outside masks are
built axis-by-axis with one reusable bool buffer, row-major voxel indices are
filled directly, and outside nearest queries compact/scatter rows without
boolean row indexing. The outside mask is counted once and that count is passed
to row compaction, so nearest-outside fallback does not rescan the same mask
after deciding the branch is needed.
Their point-to-voxel scaling also runs through one float work buffer with
`subtract/divide/floor` `out=` calls instead of chained expression temporaries.
PyVista volume-grid cell buffers and electrode face buffers are passed as
C-order `.ravel()` views instead of forced `.flatten()` copies when constructing
VTK geometry.
Shared mesh cell-to-node display projection divides into its accumulator buffer
and fills orphan/NaN nodes in-place, avoiding finite-subset copies and whole-array
`np.where` replacement buffers. Orphan and NaN fill operations now use
`np.copyto(..., where=mask)` with the existing bool masks instead of boolean
left-hand-side assignment.
The shared mesh finite scanner also writes into a reusable chunk bool buffer
with `np.isfinite(..., out=...)`, so display helpers avoid repeated chunk-sized
finite-mask allocations while checking fallback fill values.
Simulation metrics nearest-resample now direct-fills masked target rows from the
nearest source index stream, avoiding a `mapped_values` gather vector before
writing back to the NaN-padded output.
Finite-pair metric reductions also reuse two chunk-sized bool buffers for
ground-truth/reconstruction validity, avoiding a fresh
`isfinite(gt_chunk) & isfinite(rc_chunk)` mask allocation on every chunk.
When source or target geometry contains non-finite rows, the same nearest
resample path compacts valid source/query rows by preallocating the compact
payloads and direct-filling them row by row instead of boolean-indexing large
coordinate/value arrays.
Finite-row detection itself scans coordinate axes into two reusable 1D bool
buffers, avoiding chunk-sized `(rows, dim)` finite masks before deciding whether
the all-finite fast path can reuse original arrays.
The hardware equipotential PyVista surface follows the same copy budget: triangle
faces use `.ravel()` views, and warp scaling scans finite min/max in chunks
instead of copying the finite node subset.
Realtime acquisition keeps its shared-memory IPC layout stable while trimming
per-frame copies: `FrameRingBuffer.write()` checks component shape without
pre-widening the source array, copies safely into the float64 slot views, and
the GUI poller emits the private arrays returned by `read_latest()` directly
instead of copying them again.
The same float32 display budget covers GUI routing helpers: absolute-value
threshold scans use a single-precision work buffer for `float32`/`complex64`
payloads, and GREIT 2D rec-model centers padded for 3D hexa display keep the
center dtype instead of forcing an intermediate double-precision cloud.
Normalized-difference reference-floor handling follows the same rule: near-zero
reference scans and in-place clamping size their absolute-value work buffer to
the measurement dtype, so `float32`/`complex64` online RM paths avoid a hidden
float64 chunk while retaining the existing float64 default path.
`FrameData` vector extraction keeps the same ownership contract with fewer
temporaries: complex vectors fill one complex output array directly, while
magnitude and calibrated amplitude use `np.hypot` instead of building temporary
complex or squared real/imag arrays.
HDF5 checksum verification now mirrors the writer's streaming philosophy:
numeric datasets are hashed in row chunks directly from `h5py.Dataset`, so
large RM/GREIT artifacts can be verified without a full `np.asarray(dataset)`
materialization. The digest bytes remain the legacy `dtype|shape|raw-bytes`
contract, and string/object datasets keep the canonical full-read fallback.
Legacy HDF5 artifact manifest repair follows the same path: when an old dataset
is missing a stored `sha256`, the manifest builder streams `_dataset_array_digest`
instead of materializing the dataset only to compute the artifact key.
The general cache-key array digest helper follows the same bounded-copy rule:
numeric non-contiguous arrays are hashed as legacy C-order payload bytes by
leading-axis chunks rather than by first building one full contiguous copy. This
applies to mesh, RM, Jacobian, sigma, and GUI geometry signatures that reuse
`hash_array_payload` / `update_digest_with_array_payload`.
GUI RM mesh signatures now pass coordinate/connectivity views directly into that
streaming helper, so reconstruction artifact lookup does not add a separate
local contiguous copy before the shared digest path.
GREIT cache-signature digests follow the same pattern for numeric arrays:
signature metadata records the original dtype/shape while the shared streaming
helper supplies byte-stable C-order payload hashing without a GREIT-local full
contiguous copy.
Forward mesh-content hashes, TV-IRLS prior digests, and GN dense-regularization
cache hashes also pass dtype-cast arrays directly into the shared streaming
helper; any required dtype conversion is still explicit, but non-contiguous
views no longer get an additional local contiguous copy just for hashing.
The forward KSP session-reuse benchmark follows the same rule for sigma-sequence
hashes, keeping report keys byte-stable without staging a full contiguous copy
of non-contiguous benchmark arrays.
Hardware reconstruction display grids defer interpolation work buffers until the
first frame provides the node-value dtype; float32 reconstruction frames reuse
float32 interpolated/absolute/normalized buffers instead of allocating fixed
float64 display grids during mesh-cache preparation.
Boundary-voltage plots also keep y-axis range updates streaming: each series uses
finite-mask `where=` reductions, not finite-subset copies or concatenated series.
The same y-range path preserves real floating input dtype and does not widen
`float32` display curves to `float64` solely for axis scaling.
PyVista volume highlight extraction uses direct `np.flatnonzero` anomaly indices
instead of `np.where(mask)[0]` wrappers.
Those PyVista highlight paths now reuse the same `flatnonzero` result for the
branch check and `extract_cells`, avoiding a separate `np.any(inhom_mask)` scan
before index extraction.
Simulation reconstructed-voltage fitting in the main window validates finite
absolute voltages through a bounded scan buffer instead of materializing a full
`np.isfinite(reconstructed)` vector.
Forward/runtime complex-input guards now share a bounded imaginary-component
scanner, so backend routing, forward setup keys, CEM scalar coercion, and
Gauss-Newton diagnostics do not build full `imag`/`abs`/comparison temporaries
just to decide whether a payload is meaningfully complex.
Forward finite guards share the same bounded scan pattern for CEM scalar coercion
and the CUDA structured top-left diagonal check, avoiding full finite-bool
payloads on large 3D conductivity vectors and assembled system diagonals.
Reconstruction-controller fit and geometry guards use the shared bounded
finite/imaginary scanners for RM fit Jacobians, center-cloud reconstruction
geometry, streamed HDF5 RM outputs, and simulated voltage fits.
Cell-to-node averaging reuses the existing touched-node mask for final NaN
replacement and computes the replacement mean with a bounded finite scan, so
large node arrays do not allocate separate NaN and finite masks.
Conductivity image square-limit scans reuse two bounded bool buffers for finite
x/y coordinate detection instead of building `np.isfinite(x) & np.isfinite(y)`
temporaries for each chunk.
Reconstruction-matrix helpers also use bounded finite scans for measurement
vectors, frame batches, dense RM/J/RtR matrices, measurement regularization,
built one-step RM payloads, and RM application outputs.
Joint sigma/contact block-system helpers use the same bounded finite scanner for
input vectors, CSR data payloads, normal-equation matrices/RHS, PETSc/Scipy
solutions, movement Jacobian inputs, and contact-impedance updates.
Matrix-free GN and dual-mesh Jacobian helpers use bounded finite scans for
residual/parameter vectors, regularization CSR data, measurement-weight
diagonals, dense matrix actions, and projected action vectors.
Measurement-channel contracts use bounded finite scans for channel vectors,
Jacobian arrays, and diagonal/full measurement weights before applying bad
channel masking.
Measurement temporal filtering uses bounded finite scans for frame batches,
hook outputs, resumed filter state, and timestamp metadata.
The shared temporal frame validator now preserves `float32` input batches, and
measurement/inverse moving-average filters direct-fill output rows without full
denominator or sliced-difference temporaries; TV-PDHG postprocess work buffers
and sparse graph operators also follow the seed dtype on single-precision 3D
paths.
Preprojected RM batch/temporal online apply paths preserve `float32` voltage
frames through channel masking, diagonal weight scaling, metadata wrapping, and
`rm_matmul(dtype=float32)` outputs, avoiding a float64 round-trip before the RM
kernel.
GUI single-step sigma-floor limiting and constrained update paths likewise keep
single-precision background/update vectors through alpha limiting, floor clamp,
and display-delta generation.
Measurement-channel contracts preserve `float32` Jacobian, residual, and
diagonal-weight payloads through bad-channel zeroing and lightweight diagonal
transform application, so single-precision RM/GREIT/dynamic paths do not expand
their measurement-side arrays before downstream matmul or solve setup.
Difference projection helpers now preserve `float32` real voltage and Jacobian
inputs through raw/normalized vector, frame, and measurement-projection paths;
integer/default inputs still use float64 and complex inputs keep complex phase.
Hardware equipotential PyVista rendering keeps its planar surface point buffer
on the incoming display coordinate dtype instead of widening float32
reconstruction geometry solely for VTK/PyVista.
RM matmul CPU/GPU kernels use bounded finite scans for prepared RM matrices,
batched voltage deltas, and output payloads before returning results.
Inverse temporal/TV postprocess helpers use bounded finite scans for EMA initial
vectors, TV seeds, and graph-TV difference vectors.
Dynamic sequence and EIDORS noise data ingress use bounded finite scans and
min-based nonnegative checks for frame batches, timing, weights, frequencies,
and noise signals.
RtR and TV-IRLS prior helpers use bounded finite scans for prior outputs,
diagonal hints, dense/sparse payloads, graph gradients, IRLS weights,
difference data, state vectors, measurement vectors, frame batches, and initial
states.
Gauss-Newton regularization readiness uses bounded finite scans for RtRPrior
probes, diagonal hints, dense views, sparse data, LinearOperator probes, and
dense fallback matrices before optional torch transfer.
Gauss-Newton linear-system helpers use bounded finite scans for runtime arrays,
native complex deltas, matrix-free auto diagonals, custom/PMAT probes, sparse
PMAT data, dense PMATs, and fused reduced deltas.
Reduced-order snapshot ingestion uses bounded finite scans before optional
normalization and bank trimming.
Dual-mesh array validators use bounded finite scans and min-based sign checks
for cell coordinates, connectivity, voxel origins, and spacing before projection
or locator setup.
GUI reconstruction single-step sigma update guards and voxel bounds parsing use
bounded finite scans for sigma, delta, raw estimates, and bounds metadata.
Dynamic inverse helpers use bounded finite scans for sparse difference data,
timestamps, Jacobian stacks, spatial-prior data, and block-solver outputs.
GREIT desired-image, 3D distribution, finite-target, RM, rec-model, metric, and
noise-figure helpers use bounded finite scans across remaining large array
validators.
ADC/digit/holdout experiment helpers use bounded finite scans for voltage
vectors, EIT metric vectors/matrices, holdout fit matrices, and spline
predictions.
Factor/voltage sweep and dense bucket experiment builders use bounded finite
scans for input vectors, anomaly truth, reference voltage vectors, and dense
sensitivity matrices.
Unit consistency, cached 3D CEM measure validation, TV smoothness weighting,
and measurement projection guards use bounded finite scans for residual runtime
finite checks.
Channel, block-system, matrix-free GN, electrode-length, holdout-area, and
circle-bucket radius range checks use min/max/argmin reductions instead of
materializing full comparison masks.
GREIT range validators use min/max reductions for desired extents, inferred
spacing, adaptive bands, distribution bounds, image sizes, axis edges,
downsample settings, target radii, and desired steepness.
Geometry-exchange and graph-core index guards use min reductions, while the GUI
forward-result complex-measurement detector uses a bounded absolute-threshold
work buffer instead of materializing `np.abs(... ) > tol` masks.
Electrode measurement hashing computes positive/negative hit matrices once and
reuses them for argmax and row-mask checks.
Normalized-difference reference-floor clamping uses bounded chunk work buffers
for real sign and complex phase preservation, avoiding boolean subset copies
when reference channels are near zero.
Measurement-form one-step RM and GREIT scalar/vector measurement regularisation
or noise covariance stay as diagonal vectors until the dense system matrix is
available, then scaled diagonal terms are added in place.
GREIT default-radius nearest-distance filtering uses bounded finite-positive
`where` reductions instead of materializing a positive-distance subset.
Gauss-Newton fast linear-system setup detects dense diagonal regularisation via
bounded off-diagonal scans, and Woodbury jitter is added to the diagonal in
place instead of constructing a dense identity shift.
Matrix-free GN dense measurement weights use the same bounded off-diagonal scan
and diagonal-view extraction instead of building a second dense diagonal matrix.
Smoothness, Tikhonov, and TV regularization identity fallbacks build sparse
diagonal identities directly, avoiding dense `np.eye` payloads for large
parameter meshes.
PyVista 3D volume highlight selection reuses the existing anomaly boolean mask
for `extract_cells` first, keeping integer index construction as compatibility
fallback only.
Native complex GN normal solves add identity/vector-diagonal regularization
directly into `J^H J` and form prior terms as vector products, avoiding dense
diagonal `reg` and `lambda*reg` temporaries.
The GUI native-complex reconstruction controller now forwards missing
regularization as `None`, preserving that lazy identity path through dispatch.
Matrix-free dense PMAT setup preserves caller immutability with one copy, then
adds the stabilising diagonal shift in place instead of building an identity
shift matrix.
Sparse Bayesian CPU IRLS, multilevel coarse correction, and block refinement
copy/scale their dense Hessian payload once and add diagonal regularization
terms in place, avoiding `np.diag` / `np.eye` temporaries in refinement systems.
Dynamic Kalman filtering direct-fills default dense identity/covariance matrices,
reuses the RM-observation identity matrix across frames, and computes `I-KH`
once per frame before the Joseph covariance update.
TV nonlinear regularization keeps the dense ndarray return contract but fills the
scaled diagonal directly, avoiding a separate `np.diag(weights)` temporary.
Digit-metric surrogate and measurement-space RM ridge helpers add identity
regularization terms into the copied dense normal systems in place.
Measurement-channel dense compatibility conversion and dynamic vector covariance
conversion direct-fill dense diagonal matrices instead of going through
`np.diag`.
Gauss-Newton runtime initial conductivity and prior vectors use `reshape(-1)`
instead of `.flatten()` so already contiguous array inputs can stay view-backed
until a later runtime tensor or DOLFINx assignment requires a copy.
Dense diagonal extraction paths use ndarray `.diagonal()` before any required
dtype conversion, keeping extraction view-based for RtR, GN regularization, and
POD basis filtering helpers.
Gauss-Newton linear-system dense identity/diagonal compatibility paths direct-fill
their final matrices, including native-complex regularization fallbacks and the
Woodbury small-system identity seed.
GN difference benchmark/common runner applies measurement-space identity shifts
and strict dense regularization diagonals in place, and reads preconditioner
diagonals through matrix views, so cache warm/build scripts follow the same
low-peak dense-system style.
The 3D inverse-overview diagnostic renderer clips regular volumes with a 2D
XY mask applied in place, reuses voxel masks for alpha, and computes shape and
quality metrics through reductions, avoiding full masked coordinate/value
subsets during report generation.
Common GN absolute/difference script plotters share a direct Pearson
correlation reducer, avoiding `np.corrcoef`'s stacked 2-by-N temporary for long
boundary-voltage vectors.
Shared real-reconstruction gallery diagnostics reuse that correlation reducer
and compute ROI/background means with NumPy `where=` reductions, avoiding
masked truth/reconstruction subsets in large 3D report metrics.
Other diagnostic parity scripts use finite-pair Pearson reducers that keep the
original arrays in place and zero invalid centered entries, so finite masks no
longer require compacted `a[mask]` / `b[mask]` copies before correlation.
Small-domain and gallery diagnostic ROI/background means use the same common
masked-mean reducer, replacing `np.mean(arr[mask])` with `where=` reductions.
The difference-runtime benchmark now cleans measurement weights in place after
squaring, and applies floors with `out=`, avoiding an extra `np.where` weight
copy in benchmark warm/build paths.
Holdout fitting diagnostics compute raw/fitted voltage RMSE over holdout
indices with chunked `np.take(..., out=...)`, avoiding full selected-channel
copies for long holdout sweeps.
Dynamic validation benchmark fixtures direct-fill travelling-wave/plant truth
frames and synthetic measurement Jacobians row-by-row, avoiding row-list +
`vstack` duplication when fixture dimensions are scaled up.
Real reconstruction gallery slice samplers direct-fill interpolation query point
matrices, including constant plane coordinates, instead of `column_stack`/`full`
temporaries.
Tank realdata holdout comparison reuses the common script Pearson reducer,
chunks indexed holdout RMSE, direct-fills frame/output matrices, and streams
field max-abs scans instead of concatenating all reconstruction fields.
Small-domain 8e/16e and scaled-boundary grid samplers direct-fill interpolation
query points and apply circular NaN masks with row work buffers, avoiding
whole-grid column-stack and radius-mask temporaries.
Synthetic parity reports use common Pearson reduction, in-place measurement
weight cleaning, in-place dense diagonal system shifts, and direct-filled
forward CSV matrices instead of generic dense identity/diagonal/stack builders.
Difference runtime benchmarks share the same in-place weight cleaning and dense
diagonal-add pattern, so cache warm/solve benchmarks no longer allocate identity
or diagonal matrices just to shift systems.
Prior travelling-wave benchmarks direct-fill truth frame and synthetic Jacobian
rows, and compute masked peak-time errors with chunked reductions, avoiding row
lists and compact peak-index copies in dynamic fixture sweeps.
Dual-model RM benchmarks direct-fill synthetic fine-mesh centers and Jacobian
rows, and scale coarse Jacobian columns directly instead of materializing
inverse-count diagonal matrices before coarse-to-fine projection.
Common reconstruction method runners direct-fill small measurement frame stacks
for paired GN and sparse-Bayes dispatch, so batch CLI cases avoid generic
`vstack` frame matrices before alignment or dataset conversion.
GREIT EIDORS-parity benchmarks direct-fill batch sigma matrices, fallback
identity matrices, scalar-noise Gram diagonals, and synthetic measurement rows,
reducing cold benchmark memory without changing Woodbury parity contracts.
3D inverse overview rendering direct-fills cylinder wireframe and electrode
marker point matrices, avoiding generic column-stack/full-like temporaries in
diagnostic report generation.
Fair EIDORS/PyEIDORS diagnostic export direct-fills 3D boundary facets,
measurement start offsets, and measurement matrix blocks before MATLAB payload
serialization, preserving schema while trimming generic stack/concat builders.
The EIDORS forward parity gate reuses the direct-fill measurement-matrix stacker
for pattern-manager verification instead of rebuilding the concatenated matrix
with generic `vstack`.
Mesh IO format benchmarks direct-fill tag `(entity,value)` hash matrices before
streaming them through the cache hash helper, preserving byte-level equality
checks without `column_stack`.
All-mode bucket noise-sweep plots stream reconstruction and error value ranges
across methods/SNRs instead of concatenating every plotted field into one
temporary array before resolving color limits.
Small diagnostic plotting helpers direct-fill electrode point coordinates,
stream electrode-tag centroids, and direct-fill fair EIDORS color-limit vectors,
removing the last generic stack/concat builders from those report paths.
GN difference linearized LSMR fallback direct-fills augmented matvec outputs and
RHS vectors, so matrix-free fallback no longer concatenates measurement and
regularization blocks on each operator application.
GREIT center-cloud geometry spacing reuses the sorted output contract of
`np.unique`, avoiding redundant sort and finite-positive diff subset copies.
3D spatial anomaly filtering reuses the KDTree nearest-distance buffer by marking
invalid distances as NaN in-place before the median radius calculation.
2D truth/reconstruction conductivity images preserve `float32`/`complex64` result
payload dtype through display preparation, and 3D-to-2D projection keeps the
incoming floating `z` coordinate dtype instead of widening to `float64`.
Their square-axis bounds also scan finite x/y values in chunks, avoiding copied
`x[finite]` and `y[finite]` coordinate subsets while preserving NaN/Inf
exclusion semantics.
Tetrahedral 3D-to-2D projection direct-fills retained boundary-face triangles
and source-cell indices after internal-face filtering, avoiding list
comprehension staging before the final int32 arrays.
The shared boundary-triangle helper uses the same direct-fill pattern for tetra
meshes, so surface/fallback renderers avoid kept-list staging before final int32
triangle/source arrays.
The shared 3D boundary-face helper also direct-fills retained face and
source-cell outputs after shared-face filtering, avoiding another kept payload
list before surface-helper output.
For valid generated meshes, the valid-face helper reuses those face and
source-cell outputs directly and only builds filtered arrays after an
out-of-range face index is detected.
Boundary and highlight surface-helper vertices are filled as one contiguous
face-vertex array per collection, rather than allocating one small vertex array
per rendered face.
Anomaly highlight helper output also direct-fills highlight face vertices
and scalar values together, avoiding separate `highlight_faces` and
`highlight_values` staging lists before color mapping.
It consumes the anomaly bool mask directly and counts active cells for
preallocation, so large highlighted regions no longer allocate a full
`flatnonzero` cell-index array before filling highlight geometry.
Boundary-voltage reconstructed overlay curves likewise keep projected real
floating dtype through the final `setData` handoff instead of widening to
`float64`.
Shared 3D display helpers keep cell-centered `float32` sigma through
boundary-face values, anomaly masking, and highlight value helpers, so display
bookkeeping no longer widens a full cell vector just to draw.
The 3D conductivity widget no longer keeps Matplotlib facecolor caches; opacity
toggles mutate PyVista actors or offscreen actors.
For point-data surface helpers, boundary face values are computed by direct
per-index accumulation, avoiding a tiny index array and value subset allocation
for every rendered face before the `nanmean`.
Boundary and highlight face vertices use the same direct-fill pattern, creating
only the vertex array required by `Poly3DCollection` and skipping per-face
integer index arrays for coordinate gathers.
Hardware equipotential 3D rendering also preserves incoming floating coords and
conductivity dtype through widget entry and cell-to-node averaging; only the
PyVista-specific point buffer is widened locally when that backend requires it.
The paired hardware reconstruction image keeps the same dtype budget through
static-scene/cache prep and cell-to-node averaging, so `float32` reconstruction
payloads are not widened before interpolation.
Its interpolation refresh path also reuses grid-sized sample/interpolated/abs/
normalized buffers and an invalid mask, avoiding per-frame valid-row subset
copies and inverse-mask alpha writes.
Grid-cache preparation follows the same pattern: Delaunay simplex ids are
clamped into a safe index vector and final vertex/weight arrays are direct-filled
with invalid rows zeroed by `np.copyto`, avoiding valid-row subset staging.
Simulation metrics compute finite-sample L2/correlation/RMSE from chunked pair
statistics, avoiding copied `gt[finite]` / `rc[finite]` arrays while preserving
the same finite-only semantics.
For different-mesh metric comparisons, nearest-neighbor resampling now has an
all-finite geometry fast path: full source/target finite masks and geometry/value
subsets are allocated only after a non-finite row is found.
That same all-finite path direct-fills the final mapped vector with
`np.take(..., out=mapped)`, avoiding an additional full `mapped_values` array
before assigning the result into the metrics buffer.
If SciPy KDTree is unavailable and metrics fall back to brute-force nearest
search, the distance and work buffers preserve `float32` source/target geometry
instead of widening to `float64`; mixed or explicit `float64` geometry still uses
`float64` work buffers.
Batch reconstruction PNG and voltage-fit report generation follows the GUI
display dtype budget too, preserving `float32` result arrays and `int32`
connectivity instead of widening solely for offline images.
Single-result exports from the main window use the same display coercion helpers,
so manual conductivity/voltage PNG saves avoid redundant `float64`/platform-int
copies as well.
Live/manual hardware voltage plot updates and recording interop exports now use
the same real-display coercion policy, preserving `float32` measured/simulated
voltage arrays instead of widening them before pyqtgraph or snapshot assembly.
Mesh-to-node averaging and 3D display helpers also use `float32` as the
non-floating fallback, so integer/label-style conductivity payloads do not grow
to `float64` solely while preparing GUI node or cell display values.
Hardware equipotential camera alignment reuses the same display float helper for
coordinate axes, avoiding extra `float64` x/y copies during initial render and
camera reset.
When the GUI array-geometry cache misses or rejects an invalid entry, the 3D
viewer fallback cell-center computation now feeds the streaming center helper
with display-preserved coordinates, so `float32` result geometry is not widened
solely to rebuild centers.
Spatial anomaly highlighting also preserves the candidate score dtype during
component-mass ranking, avoiding a `float64` candidate-score copy for `float32`
3D display payloads.
The same spatial filter keeps candidate center coordinates at their incoming
floating dtype before KDTree construction and no longer wraps nearest-distance
views in an explicit `float64` conversion.
Candidate indices and candidate centers for that KDTree are direct-filled in one
mask pass, avoiding a `flatnonzero(mask)` plus advanced-indexing gather of
`centers[candidate_idx, :3]`.
Candidate-center finite validation uses the display scan buffer with `out=`,
avoiding a full `np.isfinite(candidate_centers)` matrix before KDTree setup.
The final coherent mask is also marked by looping over local keep flags, avoiding
a `candidate_idx[keep]` kept-index subset before assignment.
Component-mass ranking now reads scores directly from the original score array
via candidate indices, avoiding both a full `candidate_scores` vector and
per-component score subset arrays before ranking coherent blobs.
The 3D forward volume-fraction painting path keeps generated float32 mesh
coordinates as float32 through streaming sample and vertex work buffers, so
complex64/CUDA GUI runs do not duplicate node coordinates as float64 merely to
paint inhomogeneities.
Its sampled inside-count and fraction buffers also follow the coordinate/sample
dtype (`float32` for generated single-precision meshes), rather than allocating
float64 counters solely to blend inclusion volume fractions.
The legacy `cell_vertices` volume-fraction fallback follows the same dtype
budget: float32 vertex tensors stay float32 through deterministic interior
sample generation, and `_paint_shape()` no longer widens them before calling the
helper.
Centroid fallback painting applies the already-computed inclusion masks with
`np.copyto(..., where=mask)` instead of boolean-lhs assignment on `values`, so
non-volume-fraction 2D/3D painting also stays on controlled mask-write paths.
Dataset generation uses the same forward geometry extractor as GUI forward
solves, deriving centers, node coordinates, and connectivity in one topology
pass instead of calling `cell_midpoints` and then rebuilding connectivity from a
Python list of per-cell links.
The shared FEMx connectivity helpers now follow the same pattern for fallback
callers: cell/facet connectivity arrays are preallocated and filled row by row
instead of staging all `connectivity.links(...)` rows in a Python list first.
GREIT finite-target conductivity cold builds reuse one radius mask for all
targets and apply target contrast with `np.add(..., where=mask)`, avoiding both
per-target bool-mask allocation and masked conductivity subset copies.
Background conductivity positivity uses the same reduction style, so finite
target setup does not allocate a cell-count `background <= 0` bool vector before
training rows are built.
Their positivity validation uses a minimum reduction over the prepared sigma row
instead of allocating a full `sigma <= 0` boolean vector for every target.
Finite-target measurement-order validation uses min/max reductions for range
checks, avoiding `(order < 0) | (order >= n)` boolean vectors on long 3D
measurement protocols before the permutation uniqueness check.
The permutation/identity check itself then uses a compact bool `seen` vector
instead of `np.unique(order)` sorting/copying the integer order, and provided
identity orders avoid building a separate `np.arange` just for comparison.
Desired-image extent sampling detects active axes with per-axis reductions
instead of allocating an `n_cells x 3` `extents > eps` boolean matrix before
building Gauss/Sobol offsets.
Desired cell-extents validation also uses a minimum reduction for non-negative
checks, avoiding an `n_cells x 3` `cell_extents < 0` boolean matrix.
GREIT XYZ point validation scans finite status with a bounded reusable boolean
work buffer, so large model-node and target-center arrays do not allocate a
full `np.isfinite(points)` matrix before padding/truncation.
Finite-target forward measurement vectors and GREIT training response matrices
reuse that bounded scan for non-finite checks, avoiding full-size
`np.isfinite(vector/Y)` boolean payloads during training.
Imported GREIT RM artifact components and EIDORS noise-figure inputs reuse the
same scan for `Y`, desired image `D`, noise covariance, PJt cache, measurement
noise, `vh`, and volume weights.
GREIT distribution domain fallback applies bounding-box inclusion axis by axis
with reusable one-dimensional masks instead of forming an `n_centers x 3`
comparison matrix when Delaunay is unavailable.
GREIT `vh` normalization guards scan absolute values into a bounded work buffer,
so ratio-difference and EIDORS noise-figure paths do not allocate full
`abs(vh)` or `abs(vh) <= eps` arrays just to reject zero reference channels.
GREIT3D distribution cold builds also compact inside target centers through a
direct-fill helper instead of `candidate_centers[inside_mask]` boolean row
indexing, while retaining the full candidate-center artifact payload.
GREIT quality metrics use masked reductions and a small chunked weighted-sum
buffer for qmi/opposite regions, avoiding full `weights[mask]` and
`signed_image[mask]` subset copies while preserving AR/PE/RES/SD/RNG values.
Cell-volume positivity validation also uses a minimum reduction, so metric calls
with large 3D volume weights do not allocate a `cell_volumes <= 0` bool vector
before the masked reductions run.
The common default-target case (`target_values=None`) also computes target
integral and center directly from the bool target mask and cell weights, avoiding
a cell-count `float64` target array just to evaluate metrics.
When that target integral is positive, the metric code reuses the image array as
the signed image view; only negative-target metrics allocate a full negated image
buffer.
Sparse Bayesian MAP linear warm starts likewise use `np.divide(..., where=mask)`
for singular-value filtering, avoiding masked numerator/singular-value subset
arrays before the solver starts.
Gauss-Newton measurement-weight preparation sanitizes non-finite weights and
applies the floor/median normalization in-place, avoiding full `np.where` and
`np.maximum` replacement arrays plus verbose finite-subset copies.
Difference-mode weighting clamps the fresh difference buffer in-place, and
matrix-free preconditioner diagonals sanitize/floor a private copy in-place,
avoiding `np.where` replacement arrays on large measurement or parameter
vectors.
Gauss-Newton line search scans objective metrics once to select the best finite
trial and feed perturbation heuristics, avoiding finite-index arrays and
`mlist[valid_idx]` objective subsets during retry scheduling.
Its perturb-limit sanitation path also reuses one boolean work mask and fills
sign-constrained/non-finite alpha bounds via `np.copyto(..., where=mask)`, so
large parameter vectors avoid repeated boolean-index assignment paths before
trial evaluation.
Dynamic TV/Huber robust helpers now build absolute-value work buffers with
in-place square/sqrt and fill Huber weights/penalties through `out` operations,
avoiding `np.where` replacement arrays in frame-by-parameter sequences.
Temporal ROI handling in the same solver keeps default all-ROI paths on full
views and scans masked columns when needed, avoiding `weights[:, roi_mask]` and
`temporal_diffs[:, roi_mask]` submatrix copies.
ROI index-mask validation also uses min/max reductions, avoiding
`(indices < 0) | (indices >= n_parameters)` boolean vectors before the ROI
mask is materialized.
When restricting sparse temporal-difference rows to an ROI, CSR row checks scan
column indices against the ROI mask directly instead of repeatedly allocating
`roi_mask[cols]` subsets.
Dynamic frame-batch and initial-state validation scan finite status through a
bounded reusable boolean buffer, avoiding full frame-by-parameter
`np.isfinite(...)` matrices on long 3D temporal windows.
Dynamic RM, transition, initial-state, and covariance matrix validation reuse
the same finite scan, avoiding full boolean payloads for large observation and
state-space matrices before Kalman/GN setup.
Temporal weighted-normal assembly skips all-zero parameter columns via a max
reduction instead of allocating `column_weights > 0` boolean vectors for every
parameter column.
Total-variation regularization weight preparation reuses the gradient buffer for
square/sqrt/reciprocal and normalizes in-place, reducing startup allocations
when large 3D parameter meshes build regularization matrices.
Its non-finite fallback also computes the median via a finite mask and private
NaN-marked copy, avoiding `weights[np.isfinite(weights)]` subset copies on
edge-case TV weight vectors.
Graph-prior volume weighting reuses simplex vertex, basis, and Gram work buffers
instead of gathering `coords[cell]` and rebuilding small dense matrices for every
mesh cell.
Gauss-Newton finite-vector diagnostics now scan values directly for count,
min/max, and L2 norm, avoiding finite-subset copies while preserving complex
magnitude summaries.
Shared numeric finite summaries use the same scan pattern for `safe_dot`
preflight/result errors, so warning-free numeric guards no longer allocate
finite-value slices just to format min/max diagnostics.
Electrode measurement projection diagnostics also scan finite values directly,
preserving complex-magnitude summaries without allocating finite subsets during
pattern/projection error reporting.
Gauss-Newton regularization validation reports sparse/dense finite min/max via
the same scan style, so non-finite regularization errors do not copy large
finite matrix payloads.
3D point-cloud anomaly downsampling scans true-mask ranks directly, avoiding
per-chunk `np.flatnonzero(chunk)` allocations while preserving evenly spaced
sample selection.
PETSc CEM electrode-matrix assembly direct-fills non-zero coupling indices and
values together, avoiding `np.flatnonzero(c_i)` plus `c_i[nz]` value copies for
each electrode vector.
GREIT blob target generation applies the inside mask by in-place multiplication,
avoiding construction of `~mask` replacement slices while preserving zero
outside-radius values.
Measurement-channel contracts zero bad rows, vectors, and dense weight rows/
columns by scanning the bad-channel mask, avoiding boolean-index assignment
paths on large Jacobian/weight arrays.
RM frame-batch online contracts use the same scan-style bad-column zeroing for
frame-by-measurement payloads, avoiding boolean column assignment on long
temporal or 3D simulation batches.
Temporal RM online application now returns contract metadata from the same
frame-contract pass, so diagonal/identity weights avoid a separate dense
measurement-contract preparation used only for metadata.
Dynamic temporal ROI weighting zeros non-ROI parameter columns by scanning the
ROI mask, avoiding inverse-mask boolean column assignment on long
frame-by-parameter robust-weight arrays.
Diagonal and identity measurement contracts keep `weight_matrix` and
`weight_transform` in a lightweight diagonal array-like form, preserving
`np.asarray(...)` compatibility while avoiding dense O(n^2) diagonal storage
during large 3D Jacobian/RM preparation.
Full measurement-weight square-root transforms row-scale the eigenvector matrix
directly, avoiding an intermediate dense diagonal matrix before the final
transform payload.
RM frame-batch diagonal weighting square-roots the private masked weight copy
in-place before scaling frames, avoiding a per-apply sqrt vector allocation.
The 3D cell-scalar face-value helper gathers boundary face values into a
preallocated dtype-preserving output via `np.take(..., out=...)`, avoiding an
extra gather-result allocation on large boundary surfaces.
3D point-cloud display sampling similarly gathers centers and sigma into
preallocated contiguous outputs, avoiding advanced-index gather arrays during
large sampled point-cloud refreshes.
Point-cloud sample-index merging sorts the combined anomaly/background index
buffer in-place, avoiding a second sorted index array.
Native GREIT rec-model center-spacing inference also treats `np.unique` axis
coordinates as already sorted, scans adjacent positive spacings into one private
buffer, and computes the median in-place instead of re-sorting or copying a
positive-diff subset during cold desired-image extent setup.
GREIT3D distribution construction reuses the same `inside_mask` count for both
empty-volume validation and direct-fill target-center compaction, avoiding a
separate `any` pass before the helper sizes its output.
The 3D finite scan helper now reuses one chunk bool buffer and writes finite
status with `np.isfinite(..., out=...)`; full finite masks are still deferred
until the first non-finite chunk.
Spatial anomaly nearest-distance cleanup reuses the nearest buffer and writes
NaNs with `np.copyto(..., where=...)`, avoiding boolean-lhs assignment while
preserving radius selection.

The core phases split runtime-cache prep, solver imports, system configuration,
mesh + forward-model setup (including FFCx/DOLFINx work), conductivity painting,
target solve, homogeneous solve, and result packing. Use this data before
choosing whether to spend engineering effort on JIT prewarming, mesh/derived
cache changes, solver tuning, HDF5 transport, or visualization downsampling.
The `configure_system` aggregate also carries subphase entries
`configure.forward_config`, `configure.pattern`, `configure.runtime`, and
`configure.system_object`, which are useful after mesh/JIT caches are hot and
the remaining startup cost is mostly runtime policy or object construction.
During conductivity painting, GUI forward assembly extracts node coordinates,
cell connectivity, and cell centers from the DOLFINx topology in one streaming
pass instead of computing midpoints and then traversing connectivity again.

For interactive 3D simulation there are three prewarm depths:

- `EIT_APP_FORWARD_PREWARM_3D_MODE=worker` starts/reuses the selected persistent
  backend worker, primes imports, and warms the profile-local PETSc CUDA
  capability probe cache.
- `EIT_APP_FORWARD_PREWARM_3D_MODE=setup` is the default. It sends the current
  `ForwardSolverRequest` to the worker as `prime_forward_setup`, building the
  generated mesh, DOLFINx/CEM static setup, FFCx forms, and process-local static
  setup cache without running target or homogeneous forward solves.
- `EIT_APP_FORWARD_PREWARM_3D_MODE=solve` runs the full prewarm solve and can
  later serve a matching user click from the ready result.

The setup-prime mode is the default middle ground for first-load latency: it
warms the expensive setup/JIT cache path while keeping solve work out of idle
input edits. Set `EIT_APP_FORWARD_PREWARM_3D_MODE=worker` for the lighter
import-only warm path on memory-constrained machines.
GUI setup-prime de-duplication uses a stable forward-setup signature, not the
full simulation-input signature: changing inhomogeneities or noise does not
repeat static setup/JIT priming, while changing mesh refinement, electrode
layout, protocol, backend, or other setup-shaping fields does.
Setup-prime also runs under the same profile-scoped backend cache lock as
normal forward setup, so GUI prewarm, one-shot fallback, in-process backend
work, and `eit-cache warm --forward-request` do not compile against the same
FFCx cache concurrently.
The same path is available outside the GUI for reproducible diagnostics:

```bash
./eit-cache warm --profile cuda --forward-request path/to/forward_request.h5 --output setup-warm.json
```

The resulting JSON reports `warm_mode=forward_setup`, worker RSS, request
duration, `prime_command=prime_forward_setup`, and the returned
`forward_timing_ms` phase data.

For repeatable first-load measurements without hand-writing a request file, use:

```bash
python scripts/diagnostics/benchmark_gui_forward_first_load.py \
  --mode both --profile cuda --mesh-refinement 0.25 --pretty \
  --output reports/runtime_benchmarks/gui_forward_first_load.json
```

The benchmark emits schema `eit_gui_forward_first_load_benchmark_v1` and
normalizes setup-prime and full-solve timing into one JSON shape so successive
cache, JIT, solver, and visualization changes can be compared phase by phase.
Add `--prewarm-worker` to measure the GUI-like path where the persistent worker
has already run import/capability warm before setup-prime or solve:

```bash
python scripts/diagnostics/benchmark_gui_forward_first_load.py \
  --mode setup --prewarm-worker --profile cuda --mesh-refinement 0.25 --pretty \
  --output reports/runtime_benchmarks/gui_forward_setup_after_worker_warm.json
```

This output includes `worker_prewarm` and marks `prewarm_worker=true`; when
`--repair-jit` is also supplied, repair is performed during worker prewarm and
the following setup-prime skips duplicate repair.
Progress messages in the benchmark JSON are bounded: `messages` keeps only a
preview, while `message_count`, `message_limit`, and `messages_truncated`
preserve enough telemetry to tell whether a long 3D warm/solve was chatty.
Use `--progress-message-limit N` to adjust the solve preview in diagnostic
runs.

Backend worker profiles also enable a profile-local PETSc CUDA capability probe
cache under `XDG_CACHE_HOME/pyeidors-capabilities`. The cache is keyed by stable
runtime traits such as Python executable/version, PETSc module/runtime object
identity/version/scalar type, CUDA Mat/Vec symbols, and available PETSc solver
packages. This avoids repeating the expensive Mat/Vec probe setup in every
short-lived diagnostic or worker process while still invalidating when the
PETSc runtime changes.
`eit-cache doctor`, `eit-cache stats`, and `eit-cache warm` backend worker
summaries scan those `petsc_cuda_*.json` files per profile and report
`capability_probe_cache` count/bytes/latest key plus latest PETSc CUDA, Hypre,
and AmgX booleans. This is a disk-cache diagnostic only; it does not claim to
inspect another live GUI process' in-memory LRU.

## Artifact kinds

- `mesh_bundle`
- `mesh_derived`
- `pattern_bundle`
- `forward_factor`
- `jacobian`
- `single_step_operator`
- `sparse_basis`
- `measurement_projection`
- `rom_snapshot_bank`
- `rom_global_basis`
- `rom_adaptive_basis`
- `rom_reduced_operator_absolute`
- `rom_reduced_rm_diff`

Persistent disk-artifact manifest status is tracked separately in
`docs/code-fusion/T82_disk_artifact_manifest_schema_audit.md`. That audit
separates integrated manifest kinds (`hdf5-artifact`, `dolfinx-mesh-cache`)
from future-scope candidates such as ADIOS2-side artifacts and a possible
`MeshCacheLayer` protocol.

## Key design

Keys are SHA-256 hashes generated from:

- `cache_schema_version` (current: `2`)
- artifact kind
- semantic payload (mesh, pattern, drive mode/value, backend config, etc.)
- code fingerprint

Any relevant model/backend/physics change produces a new key, preventing stale reuse.

For EIDORS-style shorthand usage, PyEIDORS also supports semantic cache objects via
`CacheManager.get_or_compute_semantic(...)`, where keys are derived from normalized
dependency signatures (`cache_obj_signature`) rather than runtime object identity.

## Invalidation rules

Invalidate automatically by key mismatch when any of the following changes:

- mesh geometry / tags / association
- pattern config
- drive configuration and geometry scale
- contact impedance
- backend solver config
- code fingerprint / cache schema
- background conductivity (`sigma_hash`)
- Jacobian payload hash
- linear backend config changes (PETSc/SciPy solver options)

Invalidate manually by management API:

- `clear_name(name, namespace=None)`
- `clear_max(max_bytes)`
- `clear_old(timestamp)`
- `clear_new(timestamp)`

Manual invalidation:

```python
system.clear_cache(scope="both")
```

Additional EIDORS-like operations:

- `cache_manager.clear_name(name, namespace=None)`
- `cache_manager.clear_max(max_bytes)`
- `cache_manager.clear_old(timestamp)`
- `cache_manager.clear_new(timestamp)`
- `cache_manager.collect_recent(names=[...], limit_per_name=1, include_value=False)`
- `cache_manager.install_to_cache(snapshot, target_layers="both")`
- `cache_manager.status(name=None)` / `cache_manager.set_enabled(on, name=None)`
- `cache_manager.debug_status(name=None)` / `cache_manager.set_debug(on, name=None)`
- `cache_manager.boost_priority(delta)`

## Runtime API

`EITSystem` exposes:

- `get_cache_stats() -> dict`
- `clear_cache(scope="process"|"disk"|"both")`

Stats include hit/miss counters and process/disk footprint.
Stats also include artifact and namespace breakdown for each layer.
Stats also include global cache/debug status, disabled function names, and active priority boost.

## Corruption handling

If a disk payload is unreadable/corrupted:

1. the entry is removed automatically
2. computation falls back to recompute
3. workflow continues without hard failure

## Performance notes

- Forward solve caches matrix factors (`forward_factor`) for repeated same-sigma solves.
- Jacobian and sparse basis reuse are enabled through the same manager interface.
- Single-step difference reconstruction caches `J/Jᵀ/NOSER/A(LU)` via
  `single_step_operator` and reuses them across runs when background conductivity and
  model signatures are unchanged.
- Reduced-order 3D fast paths persist snapshot banks, global/adaptive bases, and reduced operators via
  `rom_snapshot_bank`, `rom_global_basis`, `rom_adaptive_basis`, `rom_reduced_operator_absolute`, and
  `rom_reduced_rm_diff`. These artifacts are now considered experimental accelerators rather than the primary 3D fast path.
- Mesh-derived geometry arrays are cached as HDF5 under
  `.pyeidors_cache/v2/mesh_derived/<signature>.h5`. The signature is based on
  mesh coordinate and cell-connectivity hashes, topology dimension, geometry
  dimension, dtype, and shape. The artifact stores `node_coords`,
  `cell_connectivity`, `cell_centers`, and `cell_measures`, so GUI rendering,
  geometry matching, and RM/GREIT display paths can reuse the same derived
  arrays instead of recalculating them per view/request.
- `EITMesh` also memoizes the same derived arrays in-process. Repeated
  `cells()`, `cell_centers()`, and `cell_measures()` calls on one mesh reuse the
  first extraction instead of repeatedly walking DOLFINx connectivity links.
  Artifact cold-builds likewise reuse the already extracted coordinates and
  connectivity for centers, measures, signature payload, and metadata instead
  of re-walking topology just to hash the same mesh.
  Cold `cell_measures` builds direct-fill the final measure vector for line,
  surface, and volume meshes instead of staging Python lists of every
  area/volume before array conversion.
  Tetrahedral volume measures use a dedicated index-based determinant formula
  so large 3D tetra meshes do not allocate a tiny gathered `coords[cell]` array
  for every cell during cold derived-cache construction.
  Axis-aligned hexahedral cells use a guarded extent-volume fast path and only
  fall back to generic polyhedron/ConvexHull volume when the eight vertices do
  not describe a rectangular box.
  Generic polygon and polyhedron fallback measures fill one reusable float64
  cell-vertex buffer, avoiding per-cell `coords[cell]` gather arrays while
  preserving the existing area/volume precision.
- GUI result arrays that are no longer attached to an `EITMesh` use a separate
  process-local NumPy cache in `eit_app.ui.array_geometry_cache`. The cache is
  content-addressed from `node_coords` + `cell_connectivity`, stores
  `cell_centers`, and is shared by the 3D conductivity widget and simulation
  metrics panel. It deliberately imports only NumPy so metrics/UI paths do not
  pull DOLFINx, HDF5, PyVista, or VTK into lightweight refreshes. Integer
  connectivity arrays keep their original dtype, such as `int32`, during
  signature and center derivation; the helper only promotes non-integer
  connectivity to an index-safe dtype. Cell-center derivation avoids
  materializing `coords[cells]`; it accumulates one vertex slice at a time into
  a reusable work buffer, so peak display memory stays close to the final
  `cell_centers` array instead of a full `(n_cells, verts_per_cell, dims)`
  temporary. The process LRU is capped by item count
  `EIT_APP_ARRAY_GEOMETRY_CACHE_ITEMS` (default `8`) and resident center bytes
  `EIT_APP_ARRAY_GEOMETRY_CACHE_MAX_BYTES` (default `64MiB`, `0`/`off`
  disables); oversize derived arrays are returned to the caller but not retained
  in the GUI process.
- Large 3D viewers may still use point-cloud mode even after avoiding VTK volume
  grids. To keep display memory bounded, PyVista embedded/offscreen and the
  Matplotlib fallback cap rendered cell centers by
  `EIT_APP_3D_POINT_CLOUD_MAX_POINTS` (default `60000`, `0`/`off` disables).
  The sampling is visualization-only and deterministic: anomaly/highlight
  candidates are retained first when they fit, then evenly spaced background
  points fill the remaining budget. The underlying solver/result arrays remain
  untouched. The full-data sampling pass uses only value-based O(n) candidate
  selection; spatial coherence filtering is deferred to the sampled display set
  used for highlight rendering, so large point clouds do not build a SciPy
  `cKDTree` over every original cell center before drawing.
- The 3D viewer entrypoint preserves display payload dtypes where possible:
  floating `node_coords` / conductivity arrays such as `float32`, and integer
  `cell_connectivity` such as `int32`, pass through dispatch unchanged. Backend-
  specific allocations, for example VTK's flattened `int64` cell buffer, are
  created only when that backend is actually selected.
- On WSLg/headless paths where embedded VTK is unavailable, very large
  auto-point-cloud payloads can also bypass PyVista offscreen entirely via
  `EIT_APP_3D_PYVISTA_OFFSCREEN_MAX_CELLS` (default `12000`, `0`/`off`
  disables). Above the threshold the viewer goes straight to the Matplotlib
  point-cloud fallback, avoiding the PyVista/VTK import and offscreen plotter
  allocation just to draw a sampled point cloud.
- If PyVista offscreen import, plotter creation, or the first screenshot render
  fails, the GUI records a process-local negative capability cache and later 3D
  refreshes go straight to the Matplotlib fallback instead of repeating the same
  slow VTK failure path. Set `EIT_APP_3D_PYVISTA_OFFSCREEN_NEGATIVE_CACHE=0` to
  retry on every render while debugging a graphics runtime.
- `eit-cache doctor` / `eit-cache stats` also report
  `gui_array_geometry_cache`, a process-local snapshot of the NumPy cache in
  the reporting process. A standalone `eit-cache` process normally reports its
  own empty cache; the same helper can be called from the GUI process to inspect
  real 3D widget / metrics-panel reuse.
- GUI RM hot-path artifacts use a small process LRU only when their resident
  arrays fit `rm_artifact_process_cache_max_bytes` (or
  `EIT_APP_RM_ARTIFACT_PROCESS_CACHE_MAX_BYTES`, default `512MiB`). Oversize RM
  HDF5 artifacts on CPU/auto routes use chunked streaming matmul instead of
  full RM materialization. The streaming reader opens the HDF5 file once per
  reconstruction apply and reads all row blocks from that handle, rather than
  reopening the artifact for each chunk, unless `rm_streaming_matmul` /
  `EIT_APP_RM_STREAMING_MATMUL` is `off` or CUDA is explicitly requested. The
  chunk budget is `rm_streaming_chunk_bytes` /
  `EIT_APP_RM_STREAMING_CHUNK_BYTES` (default `8MiB`). The optional persisted
  RM artifact writer stores the main `rm` dataset with
  `rm_hdf5_chunk_layout=row_full_width_v1`: chunks span all measurement columns
  and split only along reconstruction rows, matching the streaming apply
  pattern. The streaming reader also batches row reads to an integer multiple of
  the dataset chunk rows when that stays within the runtime byte budget. The
  RM writer uses HDF5 `lzf` compression by default, favoring fast row-stream
  decompression and cold-write latency over maximum disk compression; GREIT and
  other general large-cache writers keep their gzip defaults. The
  lightweight HDF5 loader keeps optional GREIT `Y/D` training matrices as lazy
  dataset handles; normal streaming `RM @ ΔV` reads only RM row chunks and reads
  `Y/D` only if the GREIT boundary-fit projection is actually requested. The
  same loader skips optional `rec_model` materialization whenever explicit
  `node_coords` + `cell_connectivity` geometry is already present, using
  `rec_model` only as a fallback geometry source. The
  optional persisted voltage-fit Jacobian has a separate restore budget
  `rm_fit_jacobian_max_bytes` (or `EIT_APP_RM_FIT_JACOBIAN_MAX_BYTES`, default
  `512MiB`); if the dataset is present but larger than the budget, the GUI
  skips the voltage-fit overlay instead of rebuilding the RM artifact or keeping
  the dense Jacobian resident. Auto-built RM artifacts also apply the same
  budget while writing: oversize dense fit Jacobians are omitted with
  `fit_jacobian_persist_skip_reason=too_large`, and warm runs reuse the RM
  artifact without rebuilding just to recover the optional voltage-fit overlay.
- HDF5 dataset checksums preserve the existing `dtype|shape|raw-bytes` digest
  contract, but numeric payloads are hashed through a byte `memoryview` instead
  of `arr.tobytes()`. Large RM/GREIT writes therefore avoid a second full-array
  byte copy while keeping old checksum semantics. The writer also reuses the
  same per-array digest for the artifact manifest and dataset `sha256`
  attributes, so each array is hashed once per write.
- Non-C-contiguous numeric arrays keep the same C-order digest bytes without
  first materializing one full contiguous copy; the digest helper streams
  leading-axis chunks through bounded contiguous work buffers. This keeps
  transposed/sliced 3D cache payloads from briefly doubling resident memory
  during artifact-key generation.
- The main delivery path remains `woodbury / pcg / cholmod-precond`, with fused fallback chain `fused -> current fast path -> strict` available only when the experimental knobs are enabled.
- When reduced artifacts improve Jacobian assembly but not end-to-end totals, treat them as stage-local research wins rather than delivery-path regressions.
- The current mac CPU封版 and the next-stage WSL2/CUDA migration plan are documented in `docs/WSL2_CUDA_HANDOFF.md`.

## EIDORS Mapping

| EIDORS command | PyEIDORS equivalent |
|---|---|
| `eidors_cache(@func, {args}, opt.cache_obj, opt.fstr)` | `get_or_compute_semantic(..., name=fstr, cache_obj=...)` |
| `clear_name` | `cache_manager.clear_name(name, namespace)` |
| `clear_max` | `cache_manager.clear_max(max_bytes)` |
| `clear_old` / `clear_new` | `cache_manager.clear_old(ts)` / `cache_manager.clear_new(ts)` |
| `collect_recent` | `cache_manager.collect_recent(names, ..., include_value=True)` |
| `install_to_cache` | `cache_manager.install_to_cache(snapshot, target_layers)` |
| `on/off` (global or per function) | `cache_manager.set_enabled(on, name=None|func_name)` |
| `debug_on/debug_off` | `cache_manager.set_debug(on, name=None|func_name)` |
| `boost_priority` | `cache_manager.boost_priority(delta)` |

For command-line operations, prefer the installable `eit-cache` entrypoint
or the repository-local `./eit-cache` wrapper:

- `eit-cache doctor`
- `eit-cache doctor --repair-jit`
- `eit-cache stats`
- `eit-cache gc --max-size 20GB`
- `eit-cache warm --profile complex64-cuda --repair-jit`
- `eit-cache off --name calc_jacobian`
- `eit-cache on --name calc_jacobian`
- `eit-cache clear-old --timestamp <epoch-seconds>`
- `eit-cache collect-recent --name inv_solve_diff_GN_one_step --with-values --output snapshot.json`
- `eit-cache install-to-cache --input snapshot.json --target-layers both`

`doctor` reports the persistent `CacheManager` status, profile-scoped GUI
backend worker caches, FFCx JIT lock health, lightweight public-import health,
process-local GUI array-geometry cache stats, and legacy `.npz/.npy` cache
artifacts under `.pyeidors_cache`. The import-health probe covers public package
entry points including `pyeidors.data`, `pyeidors.electrodes`,
`pyeidors.forward`, `pyeidors.geometry`, `pyeidors.perf`, `pyeidors.physics`,
`pyeidors.cache`, `pyeidors.visualization`, and inverse packages, and fails if
they eagerly load PETSc/DOLFINx/Torch/CUQI/MPI/gmsh, meshio, h5py, Matplotlib,
SciPy/GPU-kernel helpers, data sweep/report modules, visualization renderers, or
forward/inverse/HDF5/cache/physics/electrode implementation modules. Solver
subpackages are treated the same way: `pyeidors.inverse.solvers` may declare
GN/matrix-free/sparse Bayesian
exports, but should not load those implementations before a symbol or submodule
is requested. Regularization subpackages may declare DOLFINx-backed smoothness
classes but should not load them before class/submodule access. Prior
subpackages may declare Laplace/RtR/TV-IRLS helpers but should not load SciPy
prior implementations before function/class/submodule access. Postprocess
subpackages may declare temporal/TV helpers but should not load SciPy
postprocess implementations before function/class/submodule access. HDF5
helpers likewise keep `h5py` behind `pyeidors.io` symbol/submodule access.
Reduced, matrix-free, and workflow subpackages also expose public symbols
without importing SciPy-backed reduced/matrix-free operators or sparse workflow
engines until a symbol/submodule is requested. FEM and interop package entries
likewise declare DOLFINx/MAT exchange helpers without importing DOLFINx, UFL,
SciPy I/O, or mesh-exchange implementations until a helper/submodule is
requested. Cache and physics package entries keep NumPy-backed key/policy,
manager/store, signature, current-drive, and unit-consistency implementations
behind symbol/submodule access. Electrode package imports similarly keep pattern
generation behind `StimMeasPatternManager` access while allowing light layout
submodule access. Forward scalar-mode helpers also keep `petsc4py`
behind the first dtype query, so merely importing helper symbols does not
initialize PETSc.
`stats` includes the same probe so import regressions are visible without
running a GUI. `warm --profile <name>` starts or reuses the persistent GUI
backend worker without opening the GUI; with `--repair-jit`, warm first repairs
stale FFCx source / marker files in the project/profile cache and then reports
the selected profile cache plus the full backend-worker cache summary. The
repair path never touches the user's global `~/.cache/fenics`.
Warm reports keep a bounded `messages` preview and always report
`message_count`, `message_limit`, and `messages_truncated`; set
`PYEIDORS_CACHE_WARM_MESSAGE_LIMIT=0` to keep counts only, or raise it when
investigating a noisy worker.
Persistent GUI backend workers also report Linux RSS metadata after warm,
forward, and reconstruction routes. The environment variable
`EIT_APP_BACKEND_WORKER_MAX_RSS_MB` bounds the process-resident heap retained by
profile-scoped workers after a request; the default is `4096` MiB, and `0` /
`off` disables recycling. If a large 3D request finishes above the budget, the
parent keeps the written result artifact, records
`backend_worker_recycled_after_request=true` with
`backend_worker_recycle_reason=rss_budget_exceeded`, then stops that worker so
DOLFINx/PETSc/Python memory is returned to WSL before the next request starts a
fresh process.
Warm-only worker routes enforce the same RSS budget after optional runtime
prime. If warm imports alone exceed the budget, the worker is recycled before
any solve is queued. Forward and reconstruction result metadata also includes
`backend_worker_primed_runtime`, `backend_worker_prime_command`, and
`backend_worker_prime_duration_ms`, making 3D first-load reports easier to split
into worker/import warmup, JIT/cache repair, and actual solver time.
The `prime_runtime` metadata includes `petsc_cuda_probe`, so a worker-only GUI
prewarm or `eit-cache warm --profile cuda` can prove that capability detection
was moved into the background warm path before a later setup-prime or solve.
The GUI simulation warm report normalizes this into
`petsc_cuda_available`, `petsc_cuda_probe_cache_hit`,
`petsc_cuda_probe_cache_layer`, and `petsc_cuda_probe_status`; warm/setup done
status-bar messages include the same PETSc probe status.
The GUI simulation tab surfaces default 3D setup-prime prewarm through
status-bar messages and keeps a per-profile warm report in the window state, so
the first 3D click is no longer a silent transition from idle UI to backend
startup or setup/JIT compilation.
Traditional direct-Jacobian electrode identity drive matrices are direct-filled,
removing the last scanned dense helper from core `src` while preserving
forward-solve drive semantics.
Remaining GREIT parity / KSP / dynamic-validation script finite guards and the
3D spatial component mass check now use bounded scan helpers, avoiding full-size
bool masks on large benchmark or UI arrays.
Total-variation regularization's nonfinite median fallback also uses bounded
chunk masks and a single exact finite-count value buffer, rather than a
full-size finite mask plus NaN work copy.
Rectangle/cuboid conductivity painting over element centers similarly streams
axis comparisons through bounded work buffers, so fallback painting no longer
creates one full temporary array per coordinate axis before applying
`np.copyto(..., where=mask)`.
GN nonfinite diagnostics reuse the existing `_finite_summary` scalar scan for
complex arrays directly, avoiding an eager full-array `np.abs` magnitude copy
when a complex residual/Jacobian/linear-system guard fails.
GN difference-style measurement weighting reuses the real subtraction result as
the absolute-value buffer, while complex inputs still produce a real magnitude
reference.
GN line-search lower-alpha limiting now scans `abs(x)/abs(dx)` in bounded
chunks, so full line-search no longer keeps a complete lower-alpha array beside
large 3D parameter/update vectors.
The same line-search path scans positive/negative overflow upper limits in
bounded chunks instead of materializing full `au_pos` and `au_neg` arrays.
GN matrix-free preconditioner diagonal cleanup now scans clamp detection with a
bounded mask, avoiding a full bad-entry mask before the in-place nan/floor pass.
Residual comparison guards now share bounded comparison helpers or direct-filled
row masks, so perturbation-step checks, single-step floor flags, dynamic sweep
monotonicity, GREIT ratio zero guards, and measurement-current filtering no
longer allocate full comparison masks.
The remaining `np.all(np.isfinite(...))` script/report checks were moved to
`all_finite_values`, covering GN difference feasibility/CG status, tank holdout
spline prediction, and 3D runtime forward-output finite metadata.
Single-step sigma-floor alpha limiting now scans negative updates in chunks via
`min_alpha_for_value_floor`, avoiding full negative-update masks and copied
sigma/delta subsets on large 3D parameter fields.
Holdout-fit and bucket dense structure metrics now stream masked centroid,
covariance, area, energy, and peak reductions, avoiding ROI/outside subset
copies during larger report sweeps.
ROI-restricted TV-PDHG postprocess updates now use `copyto(..., where=roi)` and
chunked residual reductions, avoiding per-iteration ROI/non-ROI subset copies on
large 3D parameter fields.
Measurement pattern filtering now direct-fills kept rows instead of using
`meas_mat[mask]` advanced indexing during stimulation-current exclusion.
3D conductivity display payloads preserve float32/int32 zero-copy inputs but
downcast float64/complex128 display-only coordinates and conductivity values to
float32 before PyVista scene construction, halving the common first
3D view resident payload without changing solver-side arrays.
On WSLg/Wayland, embedded VTK remains disabled for native-window safety and the
viewer now tries PyVista offscreen for real 3D rendering. If PyVista offscreen
cannot render, the widget shows an explicit unavailable caption instead of a
Matplotlib 3D substitute. `EIT_APP_3D_WSLG_PYVISTA_OFFSCREEN` is no longer
required to enable the PyVista offscreen attempt path.
Forward-result geometry extraction still detaches arrays from the DOLFINx mesh
before sending them to the GUI/backend protocol, but cell-center computation now
uses one `n_cells` work vector instead of an `n_cells×dim` coordinate temporary.
When DOLFINx exposes flat topology `array` / `offsets`, the same extraction path
builds the detached GUI connectivity matrix with one reshape/copy instead of
per-cell `links()` calls; fake or irregular connectivity keeps the compatibility
fallback.
Forward and dataset-generation result packing now reuse solver measurement
arrays for the common `noise_level=0` case and only allocate a private voltage
copy when synthetic noise must be added. Downstream dtype conversion and HDF5
dataset writes remain responsible for their final owned output buffers.
EIDORS-style noise injection follows the same ownership rule: validated `v1`
and same-shape `v2` arrays remain read-only views until the final noisy output
is allocated, while broadcast expansion still gets its own buffer.
Sparse Bayesian workflow result metadata follows the same lightweight reference
contract for diagnostic arrays: `baseline_used` and `reference_measured` point
at existing workflow arrays instead of retaining full metadata snapshots.
Difference-mode `EITData` construction also avoids duplicating raw target and
reference voltage vectors at the workflow boundary; GN still takes its own
runtime snapshot when it needs mutable per-iteration state.
For single-frame normalized differences, the reference-floor copy is now lazy:
the vector path scans for near-zero reference values first, then reuses the
already allocated difference buffer for division and orientation negation.
Normalized Jacobian projection applies the same lazy floor rule for the
reference vector while still allocating a distinct projected Jacobian output.
The online RM frame-contract path now creates its mutable frame payload in one
owned C-order allocation, so non-contiguous frame batches do not pay an
`ascontiguousarray(...).copy()` double-materialization penalty.
Batch normalized-difference helpers also broadcast a single reference vector
across target frames instead of expanding it into an `n_frames×n_meas` matrix;
per-frame reference batches continue to use the existing shape-matched path.
Semantic object signatures now pass ndarray views directly to the shared
streaming `hash_array()` helper, preserving the same dtype/shape/C-order hash
contract without local contiguous materialization.
RM signatures follow the same bounded-memory contract: `_digest_value()` records
the original dtype/shape metadata and streams ndarray payload views through the
shared cache-key helper, so non-contiguous `coarse2fine`, mask, or covariance
inputs do not require a full local contiguous copy just to build a cache key.
GREIT finite-target training likewise treats no-op target-plane handling as a
view-preserving path: target centers are copied only when an offset mutation is
actually requested.
Process-local persistent GN Jacobian cache entries are stored as read-only
C-order arrays. Fast GN cache hits reuse those arrays directly instead of
allocating a full private dense-Jacobian copy; paths that need mutation still
materialize their own buffers.
The same process cache now has a resident-byte budget in addition to the item
LRU. Dense Jacobians larger than
`PYEIDORS_PROCESS_JACOBIAN_CACHE_MAX_BYTES` (or
`EIT_APP_PROCESS_JACOBIAN_CACHE_MAX_BYTES`, default 512MiB) are not retained,
and cumulative cached entries evict least-recently-used arrays by bytes. This
keeps repeated 3D GN runs from trading compute savings for unbounded worker
memory growth.
Generated/loaded mesh process caches use the same budgeted LRU pattern via
`PYEIDORS_PROCESS_MESH_CACHE_MAX_BYTES` (or
`EIT_APP_PROCESS_MESH_CACHE_MAX_BYTES`, default 512MiB). Geometry coordinates,
mesh tags, electrode vertices, and any memoized derived arrays count toward the
retention estimate, so sweeping several large 3D meshes does not keep every
DOLFINx mesh object alive merely because the item count is still below eight.
Forward static setup process caches also use a byte budget:
`PYEIDORS_PROCESS_FORWARD_SETUP_CACHE_MAX_BYTES` (or
`EIT_APP_PROCESS_FORWARD_SETUP_CACHE_MAX_BYTES`, default 512MiB). The retention
estimate includes electrode length arrays and the CEM electrode CSR matrix, so
static setup reuse remains warm for ordinary repeated requests without keeping
oversize 3D setup bundles alive across unrelated configurations.
GUI reconstruction runtime caches have the same resident-memory guard. The
full-GN `EITSystem` cache uses `EIT_APP_RECONSTRUCTION_SYSTEM_CACHE_MAX_BYTES`
and the single-step realtime context cache uses
`EIT_APP_SINGLE_STEP_CONTEXT_CACHE_MAX_BYTES` (both default 512MiB). The
estimator counts mesh/display arrays, forward-model pattern payloads,
dense/linearized operator arrays, and LU factors; entries larger than the
budget are used for the current request but not retained for later requests.
RM fit-Jacobian process reuse also treats `_RM_FIT_JACOBIAN_CACHE_MAX_BYTES`
as a total resident-byte budget. A single oversize fit Jacobian is rejected as
before, and multiple accepted fit Jacobians evict least-recently-used entries
once their cumulative array bytes exceed the budget.
The GN baseline path also avoids a redundant final-image allocation: after
clipping and difference-step scaling, the owned final sigma array is reused for
the last forward-fit solve instead of copied into a temporary `EITImage`.
Backend worker FFCx cache cleanup now builds a one-pass index of compiled
`libffcx_*.so` module stems before pruning stale `.c` / `.c.cached` lock files.
That keeps profile-cache startup proportional to one directory scan even after
many 3D forms have been compiled.
GUI array-geometry signatures likewise keep floating coordinate arrays and
integer connectivity arrays as views when possible. The streaming hash helper
preserves the existing C-order byte contract, while cell-center computation can
consume those views directly without a separate contiguous staging copy.
Mesh-derived HDF5 cache builds now prefer flat DOLFINx connectivity arrays
(`connectivity.array` + `offsets`) and reshape them once. Only irregular or
legacy connectivity falls back to per-cell `links(i)` calls, reducing Python
overhead on large 3D first-load cache builds.
CUDA structured sigma-state reuse keys keep already-float64 sigma payloads as
views and rely on the shared streaming cache-key helper for C-order bytes. The
legacy hash bytes are unchanged, but non-contiguous 3D sigma slices no longer
need a full contiguous staging allocation just to decide whether the Schur/PCG
state can be reused.
The same view-preserving rule applies to GN absolute-startup and direct
Jacobian semantic cache keys. Sigma arrays are still dtype-normalized before
hashing, and complex hashes still carry their dtype prefix, but already-typed
non-contiguous sigma views now reach the streaming payload helper without an
extra full contiguous copy.
Matrix-free GN sigma fingerprints and reduced-order snapshot de-duplication
follow that cache-key budget as well. Fingerprint bytes still match the legacy
C-order payload, and ROM duplicate columns are still filtered by payload hash,
but non-contiguous sigma vectors and matrix columns are no longer copied solely
for hashing.
GREIT artifact registry signatures also defer C-order byte handling to the
streaming hash helper. Large custom protocol, mask, or grid arrays keep their
original views while the signature still records dtype, shape, and the same
payload hash used by older artifacts.
Sparse-Bayesian Jacobian cache keys now do the same for homogeneous baseline
vectors: dtype-normalized float64 views are hashed directly, while the cached
baseline snapshot remains an owned copy for later equality checks.
GN fast linear-system cache signatures also avoid local contiguous staging for
CSR regularization arrays and ROM basis/Jacobian payloads. The shared hash
helper still emits the same C-order bytes, but large reduced-order cache keys
can be built from dtype-normalized views.
RtR prior signatures follow the same rule inside `_signature_for_payload()`:
dense/sparse prior storage still owns its required buffers, but signature
construction does not allocate another contiguous copy merely to hash them.
The 3D widget's spatial anomaly filter now sanitizes nearest-neighbour
distances in bounded chunks before `nanmedian`, excluding invalid and
nonpositive distances without allocating a full `nearest_valid` mask. This
keeps large point-cloud highlight refinement closer to one distance buffer plus
a small work mask.
Point-cloud background sampling also reuses its evenly spaced rank array as the
initial candidate sequence. Searchsorted adjustment still allocates only when
anomalies shift those ranks, avoiding an unconditional integer copy on the
common no-shift pass.
Boundary-voltage y-range updates now stream finite min/max over each plotted
series with a reusable chunk mask. Long voltage traces therefore avoid a full
series-length finite mask while preserving finite-value exclusion and the
existing display dtype policy.
The hardware equipotential 3D widget uses the same bounded work-buffer pattern
for warp-height finite min/max, so large node-value arrays no longer allocate a
fresh boolean mask for every scan chunk.

The older script path remains as a compatibility wrapper:

- `python scripts/cache/cache_ctl.py status`

## Lifecycle

- Default `cache_lifecycle="session"` maps disk artifacts into a per-terminal directory under `<cache-root>/.sessions/<session-id>`.
- In supported `nix develop` / `nix develop .#cuda` shells, the shell hook owns that session and clears it on `EXIT`, `HUP`, `INT`, `TERM`, or `deactivate`.
- Multiple terminals do not share runtime disk cache; each shell gets its own session directory and only cleans its own directory.
- `cache_lifecycle="persistent"` bypasses `.sessions/` and leaves the cache root untouched across terminal restarts.
