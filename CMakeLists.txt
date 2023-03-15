cmake_minimum_required(VERSION "3.0.0" FATAL_ERROR)
project(dockcpp)

find_program(CCACHE "ccache")
if (CCACHE)
    message(STATUS "ccache found at ${CCACHE}")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE})
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT")

# nvci does not support openmp, so we set nvcc flags in advance
set(NVCC_FLAGS_RELEASE " -std=c++11 -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -DNDEBUG -O3 -w")
set(CUDA_NVCC_FLAGS ${NVCC_FLAGS_RELEASE} ${CUDA_NVCC_FLAGS} -arch=sm_75 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60
        -gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=compute_75 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_86,code=sm_86; --use_fast_math)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -fno-strict-aliasing -Wno-deprecated-declarations -Wno-unused-local-typedefs -Wno-parentheses -Wno-catch-value")
set(CMAKE_CXX_FLAGS_DEBUG   " ${CMAKE_CXX_FLAGS} -DDEBUG -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE " ${CMAKE_CXX_FLAGS} -DNDEBUG -O3")

find_package(CUDA ${CUDA_VERSION} REQUIRED)
list(APPEND INCLUDE_PATHS ${CUDA_INCLUDE_DIRS})
list(APPEND LINK_LIBS 
    ${CUDA_LIBRARIES}
    #${CUDA_CUFFT_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    #${CUDA_cupti_LIBRARY}
    #${CUDA_curand_LIBRARY}
    #${CUDA_cusparse_LIBRARY}
    #${CUDA_npp_LIBRARY}
    #${CUDA_nppc_LIBRARY}
    #${CUDA_nppi_LIBRARY}
    #${CUDA_npps_LIBRARY}
    #${CUDA_nvcuvenc_LIBRARY}
    ${CUDA_nvcuvid_LIBRARY}
)

add_subdirectory(${PROJECT_SOURCE_DIR}/src)
add_subdirectory(${PROJECT_SOURCE_DIR}/test)


