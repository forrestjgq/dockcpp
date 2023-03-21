
include_directories(cases)
include_directories(${PROJECT_SOURCE_DIR}/src)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

FILE(GLOB_RECURSE TEST_SRC_FILES  *.c *.cc *.cpp)
# list(APPEND ${LNK_LIBS} pthread)

foreach(test_src ${TEST_SRC_FILES})
    GET_FILENAME_COMPONENT(ONE_TEST_TARGET ${test_src} NAME_WE)
    cuda_add_executable(${ONE_TEST_TARGET} ${test_src})
    TARGET_LINK_LIBRARIES(${ONE_TEST_TARGET} pthread ${PROJECT_NAME} ${LNK_LIBS} )
    install(TARGETS ${ONE_TEST_TARGET} RUNTIME DESTINATION bin/)
endforeach(test_src)