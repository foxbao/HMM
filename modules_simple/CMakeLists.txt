# 创建 modules 动态库
add_library(modules SHARED
    a.cpp
)

# 链接 common 动态库到 modules 动态库
target_link_libraries(modules PRIVATE common)

# 设置 include 目录，使得 modules 可以引用 common 的头文件
target_include_directories(modules PUBLIC ${CMAKE_SOURCE_DIR})
