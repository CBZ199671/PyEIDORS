from __future__ import annotations

import numpy as np
from petsc4py import PETSc

from pyeidors.perf.capabilities import probe_petsc_cuda_runtime


def main() -> None:
    print("scalar", np.dtype(PETSc.ScalarType))
    print("cuda_probe", probe_petsc_cuda_runtime())


if __name__ == "__main__":
    main()
