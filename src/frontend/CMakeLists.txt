add_library(mgnc_slam_frontend STATIC
    FeatureExtractor.cc
    OrbExtractor.cc
    XFeatModel.cc
    XFeatExtractor.cc
    ${PROJECT_SOURCE_DIR}/include/frontend/KeypointDetector.h
    ${PROJECT_SOURCE_DIR}/include/frontend/DescriptorGenerator.h
    ${PROJECT_SOURCE_DIR}/include/frontend/FeatureExtractor.h
    ${PROJECT_SOURCE_DIR}/include/frontend/OrbExtractor.h
    ${PROJECT_SOURCE_DIR}/include/frontend/XFeatModel.h
    ${PROJECT_SOURCE_DIR}/include/frontend/XFeatExtractor.h
)

add_library(mgnc::slam::frontend ALIAS mgnc_slam_frontend)

target_include_directories(mgnc_slam_frontend
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include/frontend
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend
)

find_package(OpenCV 4.4 REQUIRED)

set(CMAKE_PREFIX_PATH "/opt/libtorch")
find_package(Torch REQUIRED)


# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message("-- Torch Version: " ${Torch_VERSION} "; Torch flags: " ${TORCH_CXX_FLAGS} )

target_compile_options(mgnc_slam_frontend PRIVATE ${TORCH_CXX_FLAGS})


message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch lib dirs: ${TORCH_LIBRARIES}")

target_link_libraries(mgnc_slam_frontend
    PRIVATE opencv_core ${TORCH_LIBRARIES}
)

message(STATUS "Torch lib linked: ${Torch_VERSION}")