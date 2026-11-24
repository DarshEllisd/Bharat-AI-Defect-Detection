# vision_server (ADR-022): ExecuTorch QNN (NPU) + XNNPACK (CPU), models resident, Unix socket.
# Built by build_android.sh against the ExecuTorch install in $WORK/et-android-vision.
cmake_minimum_required(VERSION 3.24)
project(vision_server CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(executorch CONFIG REQUIRED)

add_executable(vision_server vision_server.cpp)
target_include_directories(vision_server PRIVATE ${EXECUTORCH_INCLUDE_DIRS})

# Backends and kernel libraries register themselves from static initialisers. ExecuTorch's exported
# targets already carry their own --whole-archive / --no-as-needed link options, so they are linked
# plainly (adding whole-archive again duplicates every symbol). ${EXECUTORCH_LIBRARIES} is NOT used: the
# config appends every optional library it finds, i.e. several kernel registration libraries at once,
# which aborts at startup ("Re-registering aten::..."). Exactly one CPU kernel library is linked
# (optimized_native_cpu_ops_lib = optimized + portable fallbacks) plus quantized_ops_lib (different ops:
# quantize/dequantize), as qnn_executor_runner does.
target_link_libraries(
  vision_server PRIVATE
  qnn_executorch_backend
  xnnpack_backend
  optimized_native_cpu_ops_lib
  quantized_ops_lib
  extension_module_static
  extension_tensor
  extension_data_loader
  executorch
  executorch_core
  log
)
# find libqnn_executorch_backend.so next to the binary or on LD_LIBRARY_PATH
set_target_properties(vision_server PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'")
