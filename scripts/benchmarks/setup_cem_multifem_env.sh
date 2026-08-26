#!/usr/bin/env bash
set -euo pipefail

# User-local, version-pinned runtime for the multi-FEM CEM accuracy study.
# No sudo, pip, uv, Nix-lock, or project dependency mutation is performed.

MFEM_VERSION="4.9"
MFEM_ARCHIVE_SHA256="ea3ac13e182c09f05b414b03a9bef7a4da99d45d67ee409112b8f11058447a7c"
MFEM_ARCHIVE_URL="https://github.com/mfem/mfem/archive/refs/tags/v${MFEM_VERSION}.tar.gz"
MFEM_BUILD_JOBS="${PYEIDORS_CEM_MFEM_BUILD_JOBS:-2}"

FREEFEM_PACKAGE="freefem++=4.9+dfsg1-2build1"
FREEFEM_LIBRARY_PACKAGE="libfreefem++=4.9+dfsg1-2build1"
GETFEM_PYTHON_PACKAGE="python3-getfem++=5.3+dfsg1-4ubuntu1"
DEB_PACKAGES=(
  "freeglut3=2.8.1-6"
  "${FREEFEM_PACKAGE}"
  "${FREEFEM_LIBRARY_PACKAGE}"
  "libmumps-seq-5.4=5.4.1-2"
  "libqhull8.0=2020.2-4"
  "libmetis5=5.1.0.dfsg-7build2"
  "libmetis-dev=5.1.0.dfsg-7build2"
  "libgetfem5++=5.3+dfsg1-4ubuntu1"
  "${GETFEM_PYTHON_PACKAGE}"
)

default_prefix() {
  if [[ -n "${PYEIDORS_CEM_MULTIFEM_PREFIX:-}" ]]; then
    printf '%s\n' "${PYEIDORS_CEM_MULTIFEM_PREFIX}"
    return
  fi
  local data_home="${XDG_DATA_HOME:-${HOME}/.local/share}"
  printf '%s\n' "${data_home}/pyeidors-cem-multifem"
}

PREFIX="$(default_prefix)"
DEB_ROOT="${PREFIX}/ubuntu-jammy"
DOWNLOAD_DIR="${PREFIX}/downloads"
MFEM_SOURCE_DIR="${PREFIX}/src/mfem-${MFEM_VERSION}"
MFEM_BUILD_DIR="${PREFIX}/build/mfem-${MFEM_VERSION}"
MFEM_INSTALL_DIR="${PREFIX}/mfem-${MFEM_VERSION}"

validate_build_jobs() {
  if [[ ! "${MFEM_BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'PYEIDORS_CEM_MFEM_BUILD_JOBS must be a positive integer\n' >&2
    exit 2
  fi
}

download_and_extract_debs() {
  mkdir -p "${DOWNLOAD_DIR}" "${DEB_ROOT}"
  local package_spec
  for package_spec in "${DEB_PACKAGES[@]}"; do
    (
      cd "${DOWNLOAD_DIR}"
      apt-get download "${package_spec}"
    )
  done

  local archive
  while IFS= read -r -d '' archive; do
    dpkg-deb --extract "${archive}" "${DEB_ROOT}"
  done < <(find "${DOWNLOAD_DIR}" -maxdepth 1 -type f -name '*.deb' -print0)
}

download_and_build_mfem() {
  mkdir -p "${PREFIX}/src" "${PREFIX}/build" "${MFEM_INSTALL_DIR}"
  local archive="${DOWNLOAD_DIR}/mfem-v${MFEM_VERSION}.tar.gz"
  if [[ ! -f "${archive}" ]]; then
    curl -fL --retry 2 -o "${archive}" "${MFEM_ARCHIVE_URL}"
  fi
  printf '%s  %s\n' "${MFEM_ARCHIVE_SHA256}" "${archive}" | sha256sum --check

  if [[ ! -f "${MFEM_SOURCE_DIR}/CMakeLists.txt" ]]; then
    mkdir -p "${MFEM_SOURCE_DIR}"
    tar -xzf "${archive}" --strip-components=1 -C "${MFEM_SOURCE_DIR}"
  fi

  cmake -S "${MFEM_SOURCE_DIR}" -B "${MFEM_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${MFEM_INSTALL_DIR}" \
    -DBUILD_SHARED_LIBS=ON \
    -DMFEM_USE_MPI=OFF \
    -DMFEM_USE_METIS=OFF \
    -DMFEM_USE_SUITESPARSE=ON \
    -DMFEM_USE_LAPACK=ON \
    -DMETIS_DIR="${DEB_ROOT}/usr" \
    -DMETIS_INCLUDE_DIRS="${DEB_ROOT}/usr/include" \
    -DMETIS_LIBRARIES="${DEB_ROOT}/usr/lib/x86_64-linux-gnu/libmetis.so" \
    -DMFEM_ENABLE_EXAMPLES=OFF \
    -DMFEM_ENABLE_MINIAPPS=OFF
  cmake --build "${MFEM_BUILD_DIR}" --parallel "${MFEM_BUILD_JOBS}"
  cmake --install "${MFEM_BUILD_DIR}"
}

write_environment_metadata() {
  mkdir -p "${PREFIX}"
  {
    printf 'schema\t%s\n' 'cem-multifem-environment-v1'
    printf 'ubuntu_release\t%s\n' '22.04'
    printf 'mfem\t%s\n' "${MFEM_VERSION}"
    printf 'mfem_source_url\t%s\n' "${MFEM_ARCHIVE_URL}"
    printf 'mfem_source_sha256\t%s\n' "${MFEM_ARCHIVE_SHA256}"
    printf 'freefem_ubuntu_package\t%s\n' "${FREEFEM_PACKAGE}"
    printf 'freefem_library_ubuntu_package\t%s\n' "${FREEFEM_LIBRARY_PACKAGE}"
    printf 'getfem_ubuntu_package\t%s\n' "${GETFEM_PYTHON_PACKAGE}"
    printf 'mfem_prefix\t%s\n' "${MFEM_INSTALL_DIR}"
    printf 'mfem_build_jobs\t%s\n' "${MFEM_BUILD_JOBS}"
    printf 'deb_root\t%s\n' "${DEB_ROOT}"
  } > "${PREFIX}/environment.tsv"
}

print_environment() {
  printf 'export PYEIDORS_CEM_MULTIFEM_PREFIX=%q\n' "${PREFIX}"
  printf 'export PATH=%q:$PATH\n' "${MFEM_INSTALL_DIR}/bin:${DEB_ROOT}/usr/bin"
  printf 'export PYTHONPATH=%q:${PYTHONPATH:-}\n' "${DEB_ROOT}/usr/lib/python3/dist-packages"
  printf 'export LD_LIBRARY_PATH=%q:%q:%q:${LD_LIBRARY_PATH:-}\n' \
    "${MFEM_INSTALL_DIR}/lib" \
    "${DEB_ROOT}/usr/lib/x86_64-linux-gnu" \
    "${DEB_ROOT}/usr/lib/freefem++"
  printf 'export FF_LOADPATH=%q:${FF_LOADPATH:-}\n' "${DEB_ROOT}/usr/lib/freefem++"
}

case "${1:-install}" in
  install)
    validate_build_jobs
    download_and_extract_debs
    download_and_build_mfem
    write_environment_metadata
    ;;
  print-env)
    print_environment
    ;;
  *)
    printf 'usage: %s [install|print-env]\n' "$0" >&2
    exit 2
    ;;
esac
