set(SRC ${PROJECT_SOURCE_DIR}/src)
list(APPEND SRC_FILES 
        ${SRC}/cu/dock.cu
        ${SRC}/dock.cpp
)
cuda_add_library(${PROJECT_NAME} SHARED ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} ${LINK_LIBS})   
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)