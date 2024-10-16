# 创建 common 动态库 (SHARED 表示动态库)
add_library(common SHARED
    common_source.cpp
)

# 将主目录作为头文件的包含路径，从而可以使用 "common/type_define.h" 形式的路径
target_include_directories(common PUBLIC ${CMAKE_SOURCE_DIR})
