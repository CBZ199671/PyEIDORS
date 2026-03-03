{
  description = "PyEIDORS development shell with FEniCSx (DOLFINx) via Nix + uv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      lib = nixpkgs.lib;
      systems = [
        "aarch64-darwin"
        "x86_64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
      forAllSystems = lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python3;
          py = python.pkgs;
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.uv
              python
              pkgs.openmpi
              pkgs.hdf5
              pkgs.gmsh
              pkgs.pkg-config
              pkgs.cmake
              pkgs.ninja

              py."fenics-dolfinx"
              py."fenics-basix"
              py."fenics-ffcx"
              py."fenics-ufl"
              py.mpi4py

              py.numpy
              py.scipy
              py.matplotlib
              py.pandas
              py.h5py
              py.pyyaml
              py.meshio
              py.gmsh
              py.pytest
              py."pytest-cov"
              py.black
              py.flake8
              pkgs.pre-commit
            ];

            shellHook = ''
              export UV_PYTHON="${python}/bin/python3"
              export UV_PYTHON_PREFERENCE=only-system
              export PYTHONNOUSERSITE=1
              export HDF5_DIR="${pkgs.hdf5}"

              if [ ! -d .venv ]; then
                echo "[nix+uv] Creating .venv with access to Nix site-packages..."
                uv venv --python "$UV_PYTHON" --system-site-packages
              fi

              source .venv/bin/activate

              echo "[nix+uv] Dev shell ready. First-time install:"
              echo "  uv pip install --python .venv/bin/python --no-deps -e ."
              echo "[nix+uv] Verify FEniCSx:"
              echo "  python -c \"import dolfinx, ufl, basix; print(dolfinx.__version__)\""
            '';
          };
        }
      );
    };
}
