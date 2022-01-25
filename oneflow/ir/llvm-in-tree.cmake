include(FetchContent)
message("-- LLVM_MONO_REPO_URL: " ${LLVM_MONO_REPO_URL})
message("-- LLVM_MONO_REPO_MD5: " ${LLVM_MONO_REPO_MD5})
FetchContent_Declare(llvm_monorepo)
FetchContent_GetProperties(llvm_monorepo)

set(LLVM_INSTALL_DIR ${THIRD_PARTY_DIR}/llvm)

if(NOT llvm_monorepo_POPULATED)
  FetchContent_Populate(llvm_monorepo URL ${LLVM_MONO_REPO_URL} URL_HASH MD5=${LLVM_MONO_REPO_MD5})
endif()
set(CMAKE_CXX_FLAGS "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_DEBUG "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "" CACHE STRING "" FORCE)

set(CMAKE_INSTALL_PREFIX ${LLVM_INSTALL_DIR} CACHE STRING "" FORCE)
set(LLVM_ENABLE_RTTI ON CACHE BOOL "turn this on to make it compatible with protobuf")
set(LLVM_ENABLE_EH ON CACHE BOOL "turn this on to make it compatible with half (the library)")
set(LLVM_BUILD_EXAMPLES OFF CACHE BOOL "")
set(LLVM_BUILD_TOOLS OFF CACHE BOOL "")
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
set(MLIR_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
set(LLVM_TARGETS_TO_BUILD host;NVPTX CACHE STRING "")
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
set(LLVM_ENABLE_PROJECTS mlir CACHE STRING "")
set(LLVM_APPEND_VC_REV OFF CACHE BOOL "")
set(LLVM_ENABLE_ZLIB OFF CACHE BOOL "")
set(LLVM_INSTALL_UTILS ON CACHE BOOL "")
set(LLVM_ENABLE_OCAMLDOC OFF CACHE BOOL "")
set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "")
set(LLVM_OPTIMIZED_TABLEGEN ON CACHE BOOL "" FORCE)
set(MLIR_ENABLE_CUDA_RUNNER ${WITH_MLIR_CUDA_CODEGEN} CACHE STRING "")
set(LLVM_MAIN_SRC_DIR ${llvm_monorepo_SOURCE_DIR}/llvm)
set(LLVM_BINARY_DIR ${llvm_monorepo_BINARY_DIR})
set(LLVM_TOOLS_BINARY_DIR ${llvm_monorepo_BINARY_DIR}/bin CACHE STRING "" FORCE)
set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
set(MLIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include)
set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")

set(llvm_monorepo_BINARY_DIR ${llvm_monorepo_BINARY_DIR})
install(TARGETS oneflow of_protoobj of_cfgobj of_functional_obj EXPORT oneflow DESTINATION lib)
install(EXPORT oneflow DESTINATION lib/oneflow)
add_subdirectory(${llvm_monorepo_SOURCE_DIR}/llvm ${llvm_monorepo_BINARY_DIR})
set(LLVM_INCLUDE_DIRS ${LLVM_MAIN_SRC_DIR}/include;${llvm_monorepo_BINARY_DIR}/include)
set(LLVM_EXTERNAL_LIT "${llvm_monorepo_BINARY_DIR}/bin/llvm-lit" CACHE STRING "" FORCE)
set(LTDL_SHLIB_EXT ${CMAKE_SHARED_LIBRARY_SUFFIX})
set(LLVM_LIBRARY_DIR "${llvm_monorepo_BINARY_DIR}/lib")
