
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

set(SRC ${PROJECT_SOURCE_DIR}/src)
list(APPEND SRC_FILES 
        ${SRC}/cu/dock.cu
        ${SRC}/dock.cpp
)
cuda_add_library(${PROJECT_NAME} STATIC ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} ${LINK_LIBS})   
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(PYDOCK_VERSION "0.1")

# Setup Python executable
if(NOT DEFINED Python3_FIND_REGISTRY)
    # Only consider PATH variable on Windows by default
    set(Python3_FIND_REGISTRY NEVER)
endif()
# Requires Python 3.6+
find_package(Python3 3.6 COMPONENTS Interpreter Development)
if(Python3_FOUND)
    # Setup PYTHON_EXECUTABLE for 3rdparty modules which still use the deprecated find_package(PythonInterp)
    set(PYTHON_EXECUTABLE
        ${Python3_EXECUTABLE}
        CACHE STRING "Deprecated path to the Python executable (for 3rdparty only)" FORCE)
else()
    message(FATAL_ERROR "BUILD_PYTHON_MODULE=ON requires Python >= 3.6. Please ensure it is in PATH.")
endif()

if(NOT Python3_EXECUTABLE)
    message(FATAL_ERROR "Python 3 not found in top level file")
endif()

# We need to get python version to configure some meta files
set(PYTHON_VERSION "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")
set(PYPI_PACKAGE_NAME "pydock")
set(Torch_DIR "${Python3_SITELIB}/torch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(INCDIRS "${TORCH_INCLUDE_DIRS}")
set(LNKLIBS "${TORCH_LIBRARIES}")
list(APPEND LNKLIBS "${Python3_SITELIB}/torch/lib/libtorch_python.so")

# pybind11
option(USE_SYSTEM_PYBIND11 OFF)
if(USE_SYSTEM_PYBIND11)
    find_package(pybind11)
endif()
if(NOT USE_SYSTEM_PYBIND11 OR NOT TARGET pybind11::module)
    set(USE_SYSTEM_PYBIND11 OFF)
    include(${THIRDPARTY_DIR}/pybind11/pybind11.cmake)
    # pybind11 will automatically become available.
endif()

# At `make`:  (or the equivalents) will be created at PYTHON_COMPILED_MODULE_DIR. The default location is
# `build/lib/${CMAKE_BUILD_TYPE}/`
set(PYTHON_COMPILED_MODULE_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/compiled")
message(STATUS "PYTHON_COMPILED_MODULE_DIR is ${PYTHON_COMPILED_MODULE_DIR}")
set(PYTHON_EXTRA_DIRS "")

pybind11_add_module(cudock)
set_target_properties(
    cudock
    PROPERTIES FOLDER "pydock"
               LIBRARY_OUTPUT_DIRECTORY "${PYTHON_COMPILED_MODULE_DIR}"
               ARCHIVE_OUTPUT_DIRECTORY "${PYTHON_COMPILED_MODULE_DIR}")

add_dependencies(cudock ${PROJECT_NAME})
list(APPEND PYSRC bind.cpp cuda_context.cpp)
target_sources(cudock PRIVATE ${PYSRC})

list(APPEND INCDIRS ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${INCDIRS})
target_link_libraries(cudock PRIVATE ${LNKLIBS} ${PROJECT_NAME})

set(PYTHON_PACKAGE_SRC_DIR "${PROJECT_SOURCE_DIR}/python")
set(PYTHON_PACKAGE_DST_DIR "${CMAKE_BINARY_DIR}/lib/python_package")
message(STATUS "PYPI_PACKAGE_NAME: ${PYPI_PACKAGE_NAME}")

set(PYDOCK_BUILD_TYPE ${CMAKE_BUILD_TYPE})
configure_file("_build_config.py.in" "${CMAKE_BINARY_DIR}/lib/_build_config.py.in")
file(
    GENERATE
    OUTPUT "${PYTHON_COMPILED_MODULE_DIR}/_build_config.py"
    INPUT "${CMAKE_BINARY_DIR}/lib/_build_config.py.in")

set(COMPILED_MODULE_PATH_LIST $<TARGET_FILE:cudock>)

add_custom_target(
    python-package
    COMMAND
        ${CMAKE_COMMAND} -DPYTHON_PACKAGE_SRC_DIR=${PYTHON_PACKAGE_SRC_DIR}
        -DPYTHON_PACKAGE_DST_DIR=${PYTHON_PACKAGE_DST_DIR} -DPYTHON_COMPILED_MODULE_DIR=${PYTHON_COMPILED_MODULE_DIR}
        -DPROJECT_ROOT_DIR=${CMAKE_SOURCE_DIR} -DPYTHON_VERSION=${PYTHON_VERSION}
        -DBUILD_PYDOCK_VERSION=${PYDOCK_VERSION} -DPYDOCK_TARGET=${PYDOCK_TARGET}
        -DCOMPILED_MODULE_PATH_LIST=${COMPILED_MODULE_PATH_LIST}
        -DPROJECT_VERSION=${PYDOCK_VERSION} -DPYPI_PACKAGE_NAME=${PYPI_PACKAGE_NAME} -P
        ${CMAKE_CURRENT_SOURCE_DIR}/make_python_package.cmake
    VERBATIM
    DEPENDS cudock)

# Use `make pip-package` to create the pip package in the build directory
add_custom_target(
    pip-package
    COMMAND ${Python3_EXECUTABLE} setup.py bdist_wheel --dist-dir pip_package
    COMMAND echo "pip wheel created at ${PYTHON_PACKAGE_DST_DIR}/pip_package"
    WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
    DEPENDS python-package)

# Use `make install-pip-package` to install pip wheel package to the current python environment.
add_custom_target(
    install-pip-package
    COMMAND ${CMAKE_COMMAND} -DPYTHON_PACKAGE_DST_DIR=${PYTHON_PACKAGE_DST_DIR}
            -DPython3_EXECUTABLE=${Python3_EXECUTABLE} -P ${CMAKE_CURRENT_SOURCE_DIR}/make_install_pip_package.cmake
    DEPENDS pip-package)

# FOR DEBUGGING ONLY Use `make install-python-package` to build and install python package in the current python
# environment. This is substantially faster than `make install-pip-package`. However this approach does not create wheel
# or egg files and does not take care of dependencies thus not suitable for deployment. Ref:
# https://stackoverflow.com/a/33791008/1255535
add_custom_target(
    install-python-package
    COMMAND ${Python3_EXECUTABLE} setup.py install --single-version-externally-managed --root=/
    WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
    DEPENDS python-package)
