# T82 Disk Artifact Manifest Schema Audit

Date: 2026-04-27

Scope: T82 phase 4. This is a small architecture audit for the persistent
disk-artifact manifest layer introduced by T82 phases 1..3. It records which
artifact kinds are already wired into the shared schema, which paths only
reference the schema indirectly, and which candidates remain future scope.

## Canonical Schema

Every manifest lives under artifact metadata as `artifact_manifest`. The
canonical fields are:

| field | meaning | rule |
|---|---|---|
| `artifact_kind` | logical artifact family | Examples: `hdf5-artifact`, `dolfinx-mesh-cache`. |
| `artifact_key` | stable semantic digest for one artifact | Built from `artifact_kind`, `namespace`, `schema_version`, and `key_payload`. Output path, device, and storage backend must not enter this key. |
| `subkeys` | named shared semantic digests | Used when different artifact kinds share provenance, e.g. `mesh_provenance`. |
| `namespace` | logical key namespace | Default `pyeidors`. |
| `schema_version` | manifest key schema version | Currently `1`. |
| `manifest_version` | manifest payload version | Currently `1`. |
| `key_payload` | semantic payload used for `artifact_key` | Must contain math/provenance inputs, not file placement. |
| `files` | physical artifact file fingerprints | May include path, size, mtime, directory flag, and optional SHA256. Files are diagnostic, not semantic identity. |
| `metadata` | format-level manifest metadata | Small descriptors such as `artifact_format` / `cache_format`. |

## Integrated Artifact Kinds

| artifact kind | status | writer | reader/backfill | subkeys |
|---|---|---|---|---|
| `hdf5-artifact` | integrated | `pyeidors.io.hdf5_artifacts.write_hdf5_artifact`, `write_large_cache_hdf5_artifact` | `read_hdf5_artifact` backfills missing `artifact_key` / `artifact_manifest` in memory for legacy HDF5. | Optional `subkey_payloads`; `mesh_provenance` tested. |
| `dolfinx-mesh-cache` | integrated | `pyeidors.geometry.dolfinx_mesh_cache.write_dolfinx_mesh_cache` | `load_dolfinx_mesh_cache` backfills missing manifest in memory for legacy metadata JSON. | Automatic `mesh_provenance`. |

## Indirectly Covered Paths

These paths inherit `hdf5-artifact` because they call the shared HDF5 writer:

| path | status | note |
|---|---|---|
| GREIT RM HDF5 artifacts | covered-by-`hdf5-artifact` | Uses `GREITRM.save` -> `write_large_cache_hdf5_artifact`. |
| one-step RM / RtR / dataset / dynamic measurement HDF5 packages | covered-by-`hdf5-artifact` | Existing readers get in-memory manifest backfill through `read_hdf5_artifact`. |
| MATLAB mesh bridge HDF5 packages | covered-by-`hdf5-artifact` | Covered when using the shared writer/reader. |

## Future Scope

| candidate | status | gate before integration |
|---|---|---|
| `adios4dolfinx-checkpoint` | future scope | Add only if ADIOS4DOLFINx becomes an independent reload source. Today it is optional and is listed as a file inside the `dolfinx-mesh-cache` manifest. |
| `adios2-vtx-side-artifact` | future scope | Add only if the VTX/BP side artifact gains a supported reader. Today it remains optional write-side output. |
| `cache-manager-disk-object` | future scope | The `.pyeidors_cache/v2` sqlite/object store keeps its own runtime index. Bridge to `artifact_manifest` only if a durable export/import workflow needs it. |
| legacy `.npz` artifacts | future scope/read-only | Remain read-only compatibility inputs. New large 3D artifacts must use HDF5 per V65/V67. |
| `MeshCacheLayer` protocol | future scope | Do not introduce until XDMF/HDF5, optional ADIOS2, and HDF5 numeric packages show a real shared storage backend, not only a shared manifest. |

## Phase 4 Decision

T82 is not complete yet. Phases 1..3 established:

- shared manifest key builder,
- writer-side manifest embedding for HDF5 and DOLFINx mesh cache,
- reader-side in-memory backfill for legacy HDF5 and legacy mesh metadata,
- cross-layer `mesh_provenance` subkey.

Remaining work is governance and optional integration, not a blocker for current
HDF5/DOLFINx usage. Keep T82 status `~` until either:

1. this audit is accepted as the completion boundary and future candidates move
   to a new task, or
2. the future-scope candidates above are implemented and gated.
