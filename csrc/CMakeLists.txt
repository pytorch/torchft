cmake_minimum_required(VERSION 3.4...3.18)
project(_torchft_cpp)

# Python
find_package (Python3 COMPONENTS Interpreter Development)
set(PYTORCH_ROOT "${Python3_SITELIB}")
include_directories(BEFORE "${Python3_INCLUDE_DIRS}")

# CUDA
find_package(CUDA REQUIRED)
include_directories(BEFORE "${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_VERSION: ${CUDA_VERSION}")

set(TORCH_CUDA_ARCH_LIST "5.0;6.0;7.0;7.5;8.0;8.6;9.0")

# NCCL
include(ExternalProject)
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/External/nccl.cmake)

# torch
# if pytorch was installed in develop mode we need to resolve the egg-link
set(PYTORCH_EGG_LINK "${PYTORCH_ROOT}/torch.egg-link")
if (EXISTS "${PYTORCH_EGG_LINK}")
  file (STRINGS "${PYTORCH_EGG_LINK}" PYTORCH_ROOT LIMIT_COUNT 1)
endif()

message(STATUS "PYTORCH_ROOT: ${PYTORCH_ROOT}" )

include_directories(BEFORE "${PYTORCH_ROOT}/torch/include")
include_directories(BEFORE "${PYTORCH_ROOT}/torch/include/torch/csrc/api/include/")
LINK_DIRECTORIES("${PYTORCH_ROOT}/torch/lib")

#include_directories(BEFORE "${Python3_SITELIB}/triton/backends/nvidia/include/")
#include_directories(BEFORE "${Python3_SITELIB}/nvidia/cuda_runtime/include/")
#include_directories(BEFORE "${Python3_SITELIB}/nvidia/cusparse/include/")

add_definitions(-DUSE_C10D_NCCL)

add_library(${PROJECT_NAME} SHARED ProcessGroupNCCL.cpp init.cpp NCCLUtils.cpp cuda_utils.cpp)

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
target_link_libraries(${PROJECT_NAME} __caffe2_nccl)
target_link_libraries(${PROJECT_NAME} torch_cpu torch_cuda c10_cuda)

set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".${Python3_SOABI}.so"
)


install(
    TARGETS ${PROJECT_NAME}
    DESTINATION "${PROJECT_SOURCE_DIR}/../torchft"
)
