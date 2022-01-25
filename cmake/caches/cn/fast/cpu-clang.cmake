set(CMAKE_C_COMPILER
    "clang"
    CACHE STRING "")
set(CMAKE_CXX_COMPILER
    "clang++"
    CACHE STRING "")
set(CMAKE_EXE_LINKER_FLAGS_INIT
    "-fuse-ld=lld"
    CACHE STRING "")
set(CMAKE_MODULE_LINKER_FLAGS_INIT
    "-fuse-ld=lld"
    CACHE STRING "")
set(CMAKE_SHARED_LINKER_FLAGS_INIT
    "-fuse-ld=lld"
    CACHE STRING "")
set(BUILD_SHARED_LIBS
    YES
    CACHE BOOL "")
set(CMAKE_LINK_DEPENDS_NO_SHARED
    YES
    CACHE BOOL "")
set(BUILD_CUDA
    NO
    CACHE BOOL "")
set(BUILD_TESTING
    YES
    CACHE BOOL "")
set(THIRD_PARTY_MIRROR
    aliyun
    CACHE STRING "")
set(PIP_INDEX_MIRROR
    "https://pypi.tuna.tsinghua.edu.cn/simple"
    CACHE STRING "")
set(CMAKE_BUILD_TYPE
    RelWithDebInfo
    CACHE STRING "")
set(CMAKE_GENERATOR
    Ninja
    CACHE STRING "")
set(CMAKE_C_COMPILER_LAUNCHER
    ccache
    CACHE STRING "")
set(CMAKE_CXX_COMPILER_LAUNCHER
    ccache
    CACHE STRING "")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION
    OFF
    CACHE BOOL "")
set(BUILD_HWLOC
    OFF
    CACHE BOOL "")
