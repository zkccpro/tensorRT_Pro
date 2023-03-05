cmake_minimum_required(VERSION 3.17)
project(pro)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)

# 如果要支持python则设置python路径
set(HAS_PYTHON OFF)
set(PythonRoot "/opt/conda/")
set(PythonName "python3.8")

# 如果你是不同显卡，请设置为显卡对应的号码参考这里：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
set(CUDA_GEN_CODE "-gencode=arch=compute_86,code=sm_86")

# 如果你的opencv找不到，可以自己指定目录
set(OpenCV_DIR   "/env/opencv-4.5.4/build")
set(CUDA_TOOLKIT_ROOT_DIR     "/usr/local/cuda-11.1")
set(CUDNN_DIR    "/usr/local/cudnn8.0.4")
set(TENSORRT_DIR "/env/TensorRT-8.2.2.1")

set(MMDEPLOY_TRTOPS_DIR "third_party/mmdeploy/csrc/mmdeploy/backend_ops/tensorrt")
set(AMIRSTAN_OPS_DIR "third_party/amirstan_plugin")

# 因为protobuf，需要用特定版本，所以这里指定路径
# 我直接把protobuf-3.11.4安装到/usr/local/路径下了(protobuf源码安装时make install)
set(PROTOBUF_DIR "/usr/local")

find_package(CUDA REQUIRED)
find_package(OpenCV)

include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/app
    ${PROJECT_SOURCE_DIR}/src/tensorRT
    ${PROJECT_SOURCE_DIR}/src/tensorRT/common
    # ${AMIRSTAN_OPS_DIR}/include
    # ${AMIRSTAN_OPS_DIR}/src/plugin/common
    # ${MMDEPLOY_TRTOPS_DIR}
    # ${MMDEPLOY_TRTOPS_DIR}/../common
    # ${MMDEPLOY_TRTOPS_DIR}/common
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${PROTOBUF_DIR}/include
    ${TENSORRT_DIR}/include
    ${CUDNN_DIR}/include
)

if ("${OPS_LIB}" MATCHES "amirstan")
    include_directories(
        ${AMIRSTAN_OPS_DIR}/include
        ${AMIRSTAN_OPS_DIR}/src/plugin/common
    )
endif()

if ("${OPS_LIB}" MATCHES "mmdeploy")
    include_directories(
        ${MMDEPLOY_TRTOPS_DIR}
        ${MMDEPLOY_TRTOPS_DIR}/../common
        ${MMDEPLOY_TRTOPS_DIR}/common
    )
endif()

# 切记，protobuf的lib目录一定要比tensorRT目录前面，因为tensorRTlib下带有protobuf的so文件
# 这可能带来错误
link_directories(
    ${PROTOBUF_DIR}/lib
    ${TENSORRT_DIR}/lib
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${CUDNN_DIR}/lib
)

if("${HAS_PYTHON}" STREQUAL "ON")
    message("Usage Python ${PythonRoot}")
    include_directories(${PythonRoot}/include/${PythonName})
    link_directories(${PythonRoot}/lib)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAS_PYTHON")
endif()

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O0 -Wfatal-errors -pthread -w -g")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")

if ("${OPS_LIB}" MATCHES "amirstan")
    file(GLOB_RECURSE cpp_srcs ${PROJECT_SOURCE_DIR}/src/*.cpp ${AMIRSTAN_OPS_DIR}/*.cpp)
    file(GLOB_RECURSE cuda_srcs ${PROJECT_SOURCE_DIR}/src/*.cu ${AMIRSTAN_OPS_DIR}/*.cu)

endif()

if ("${OPS_LIB}" MATCHES "mmdeploy")
    file(GLOB_RECURSE cpp_srcs ${PROJECT_SOURCE_DIR}/src/*.cpp ${MMDEPLOY_TRTOPS_DIR}/*.cpp)
    file(GLOB_RECURSE cuda_srcs ${PROJECT_SOURCE_DIR}/src/*.cu ${MMDEPLOY_TRTOPS_DIR}/*.cu)
endif()

cuda_add_library(plugin_list SHARED ${cuda_srcs})
target_link_libraries(plugin_list nvinfer nvinfer_plugin)
target_link_libraries(plugin_list cuda cublas cudart cudnn)
target_link_libraries(plugin_list protobuf pthread)
target_link_libraries(plugin_list ${OpenCV_LIBS})

# add_executable(pro ${cpp_srcs}) # 直接作为工程编译
# 编译单测
file(GLOB_RECURSE cpp_tests ${PROJECT_SOURCE_DIR}/test/*.cpp)
add_executable(pro ${cpp_srcs} ${cpp_tests}) # 编译单测

# 如果提示插件找不到，请使用dlopen(xxx.so, NOW)的方式手动加载可以解决插件找不到问题
target_link_libraries(pro nvinfer nvinfer_plugin)
target_link_libraries(pro cuda cublas cudart cudnn)
target_link_libraries(pro protobuf pthread plugin_list)
target_link_libraries(pro ${OpenCV_LIBS})
target_link_libraries(pro gtest)

if("${HAS_PYTHON}" STREQUAL "ON")
    set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/example-python/pytrt)
    add_library(pytrtc SHARED ${cpp_srcs})
    target_link_libraries(pytrtc nvinfer nvinfer_plugin)
    target_link_libraries(pytrtc cuda cublas cudart cudnn)
    target_link_libraries(pytrtc protobuf pthread plugin_list)
    target_link_libraries(pytrtc ${OpenCV_LIBS})
    target_link_libraries(pytrtc "${PythonName}")
    target_link_libraries(pro "${PythonName}")
endif()

add_subdirectory(third_party/googletest) # 编译单测
