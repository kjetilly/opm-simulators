cmake_policy(VERSION 3.3)

set(AMGX_VALID_COMPONENTS
    C
    CXX)

find_package(CUDA)

SET(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_cusolver_LIBRARY})
# search AMGX header
find_path(AMGX_INCLUDE_DIR amgx_c.h amgx_capi.h
    HINTS "${AMGX_DIR}/include"
    DOC "Include directory of AMGX"
)

find_library(AMGX_LIBRARY
      NAMES "amgxsh" "amgx"
      HINTS "${AMGX_DIR}/lib64" "${AMGX_DIR}/lib"
)

SET(AMGX_INCLUDE_DIRS ${AMGX_INCLUDE_DIR})
SET(AMGX_LIBRARIES ${AMGX_LIBRARY})

#set HAVE_AMGXSOLVER for config.h
set(HAVE_AMGXSOLVER ${AMGXSOLVER_FOUND})
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (AmgX
  DEFAULT_MSG AMGX_LIBRARY AMGX_INCLUDE_DIR)
