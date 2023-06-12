cmake_minimum_required(VERSION "3.0.0" FATAL_ERROR)
project(dockcpp)

find_program(CCACHE "ccache")
if (CCACHE)
    message(STATUS "ccache found at ${CCACHE}")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE})
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++14 -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT")

# nvci does not support openmp, so we set nvcc flags in advance
# set(NVCC_FLAGS_RELEASE -std=c++14 -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -DNDEBUG -O3 -w)
set(NVCC_FLAGS_RELEASE -g -G -std=c++14 -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -DNDEBUG -O0 -w)
#set(CUDA_NVCC_FLAGS ${NVCC_FLAGS_RELEASE} ${CUDA_NVCC_FLAGS} -arch=sm_75 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 
#    -gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=compute_75 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_86,code=sm_86; 
#    --use_fast_math)
# set(CUDA_NVCC_FLAGS ${NVCC_FLAGS_RELEASE} ${CUDA_NVCC_FLAGS} -arch=sm_75 -gencode arch=compute_75,code=compute_75 -gencode arch=compute_86,code=sm_86; 
#    --use_fast_math)
set(CUDA_NVCC_FLAGS ${NVCC_FLAGS_RELEASE} ${CUDA_NVCC_FLAGS} -arch=sm_75 -gencode arch=compute_75,code=compute_75 -gencode arch=compute_86,code=sm_86; )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -fno-strict-aliasing -Wno-deprecated-declarations -Wno-unused-local-typedefs -Wno-parentheses -Wno-catch-value")
set(CMAKE_CXX_FLAGS_DEBUG   " ${CMAKE_CXX_FLAGS} -DDEBUG -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE " ${CMAKE_CXX_FLAGS} -DNDEBUG -O3")

set(THIRD_PARTY_DOWNLOAD_DIR ${CMAKE_SOURCE_DIR}/thirdparty_download)
set(THIRDPARTY_DIR ${CMAKE_SOURCE_DIR}/thirdparty)

set(LNK_LIBS ${LNK_LIBS} ${OpenCV_LIBS} )
set(INCLUDE_PATHS ${INCLUDE_PATHS} ${OpenCV_INCLUDE_DIRS} )
# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
include_directories(${INCLUDE_PATHS})
add_subdirectory(${PROJECT_SOURCE_DIR}/src)
add_subdirectory(${PROJECT_SOURCE_DIR}/test)


