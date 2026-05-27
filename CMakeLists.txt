cmake_minimum_required(VERSION 3.20)

project(sp_tiles LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(cuda_tile_source_dir "${CMAKE_SOURCE_DIR}/ext/cuda-tile")
set(cuda_tile_binary_dir "${CMAKE_BINARY_DIR}/cuda_tile-cmake")

# CUDA Tile options.
set(CUDA_TILE_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(CUDA_TILE_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "" FORCE)
set(CUDA_TILE_ENABLE_TOOLS ON CACHE BOOL "" FORCE)

# CUDA Tile searches MLIR here:
#   ${cuda_tile_binary_dir}/lib/cmake/mlir
#
# But in your embedded build MLIRConfig.cmake is generated here:
#   ${CMAKE_BINARY_DIR}/lib/cmake/mlir
#
# So we make CUDA Tile's expected path point to the real one.
set(real_mlir_cmake_dir "${CMAKE_BINARY_DIR}/lib/cmake/mlir")
set(cuda_tile_expected_mlir_cmake_dir "${cuda_tile_binary_dir}/lib/cmake/mlir")

file(MAKE_DIRECTORY "${cuda_tile_binary_dir}/lib/cmake")
file(REMOVE_RECURSE "${cuda_tile_expected_mlir_cmake_dir}")
file(CREATE_LINK
  "${real_mlir_cmake_dir}"
  "${cuda_tile_expected_mlir_cmake_dir}"
  SYMBOLIC
)

find_package(Vulkan REQUIRED)

add_subdirectory("${cuda_tile_source_dir}" "${cuda_tile_binary_dir}")

add_executable(sp_tiles_translate
  src/main.cpp
)

add_executable(sp_tiles_translate_test
  src/examples.cpp
)

target_link_libraries(sp_tiles_translate_test PRIVATE Vulkan::Vulkan)


if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(sp_tiles_translate
    PRIVATE
      -fno-rtti
      -fno-exceptions
  )
endif()

target_include_directories(sp_tiles_translate
  PRIVATE
    "${cuda_tile_source_dir}/include"
    "${cuda_tile_binary_dir}/include"

    "${cuda_tile_binary_dir}/llvm-project/llvm/include"
    "${cuda_tile_binary_dir}/llvm-project/mlir/include"
    "${cuda_tile_binary_dir}/llvm-project-build/include"
    "${cuda_tile_binary_dir}/llvm-project-build/tools/mlir/include"
)

target_link_libraries(sp_tiles_translate
  PRIVATE
    CudaTileBytecodeReader
    CudaTileDialect

    MLIRParser
    MLIRDialect
    MLIRIR
    MLIRSupport
    MLIRPass
    MLIRSPIRVDialect
    MLIRSPIRVConversion
    MLIRSPIRVSerialization
    MLIRSPIRVTransforms

    MLIRFunctionInterfaces
    MLIRControlFlowInterfaces
    MLIRInferTypeOpInterface
    MLIRSideEffectInterfaces
    MLIRViewLikeInterface

    LLVMSupport
)