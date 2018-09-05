cmake_minimum_required(VERSION 3.5)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
message(STATUS "compile name ${CMAKE_CXX_COMPILER_ID}")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
message(STATUS "compile use clang")
if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../build-android/arm64-v8a/install/libstatic)
else("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a")
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../build-android/armeabi-v7a/install/libstatic)
endif ()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
message(STATUS "compile use gcc")
if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../build-android/arm64-v8a-gcc/install/libstatic)
else("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a")
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../build-android/armeabi-v7a-gcc/install/libstatic)
endif ()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
# using Intel C++
message(STATUS "compile use intel")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
# using Visual Studio C++
message(STATUS "compile use msvc")
endif()

if( CMAKE_SYSTEM_PROCESSOR MATCHES armv7-a )
add_definitions(-mfpu=neon-fp16)
endif()

#add_definitions(-std=c++11)

if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")

add_executable(tinySgemmConv_test64 main.c)
target_link_libraries(tinySgemmConv_test64 tinySgemmConv)

else("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a")

add_executable(tinySgemmConv_test32 main.c)
target_link_libraries(tinySgemmConv_test32 tinySgemmConv)

endif ()
