# 指定所需的最低CMake版本
cmake_minimum_required(VERSION 3.10)

message(STATUS "CMAKE version ${CMAKE_VERSION}")
# 定义项目名称和C++标准
project(HMM)

# 设置使用的C++标准
set(CMAKE_CXX_STANDARD 11)  # 或者更高的标准，比如 C++14, C++17

# 设置编译选项
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif()

# 添加编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
set(CMAKE_CXX_FLAGS "-std=c++14 -g3 -rdynamic")
set(CMAKE_CXX_FLAGS_DEBUG "-O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")


#设置库文件输出目录
if (NOT LIBRARY_OUTPUT_PATH) 
    set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/libs)
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
# find_package(Ceres REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})


# # set the default modules to be compiled
# set(compile_dirs common)
# # compile only the necessary modules
# # list(APPEND compile_dirs civmap modules)


# foreach(subdir ${compile_dirs})
#   if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/${subdir}")
#     add_subdirectory(${subdir})
#   endif()
# endforeach()


add_subdirectory(common)
add_subdirectory(modules)
add_subdirectory(civmap)

# 添加源文件
set(SOURCES main.cpp)

# 指定生成的可执行文件
add_executable(HMM main.cpp)

# 如果需要链接其他库，例如，pthread，OpenCV等，可以在这里加
target_link_libraries(HMM PRIVATE common)
