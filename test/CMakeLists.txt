cmake_minimum_required(VERSION 3.5)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
message(STATUS "compile name ${CMAKE_CXX_COMPILER_ID}")

include(FindOpenMP)
if(OPENMP_FOUND)
	#if(OpenMP_CXX_FOUND OR OPENMP_FOUND)
	#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	message(STATUS ${OpenMP_C_FLAGS})
	message(STATUS ${OpenMP_CXX_FLAGS})
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
message(STATUS "OpenMP flags ${CMAKE_CXX_FLAGS}")


if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

message(STATUS "compile use clang")
if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../build-android/arm64-v8a/install/libstatic)
else("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a")
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../build-android/armeabi-v7a/install/libstatic)
endif()

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

message(STATUS "compile use gcc")
if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../build-android/arm64-v8a-gcc/install/libstatic)
else("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a")
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../build-android/armeabi-v7a-gcc/install/libstatic)
endif()

endif()

if( CMAKE_SYSTEM_PROCESSOR MATCHES armv7-a )
add_definitions(-mfpu=neon-fp16)
endif()

add_definitions(-std=gnu99)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")

add_executable(tinySgemmConv_test64 convolution_3x3s1.c main.c)
target_link_libraries(tinySgemmConv_test64 tinySgemmConv)

else("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a")

add_executable(tinySgemmConv_test32 convolution_3x3s1.c main.c)
target_link_libraries(tinySgemmConv_test32 tinySgemmConv)

endif ()

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")

add_executable(tinySgemmConv_test64gcc convolution_3x3s1.c main.c)
target_link_libraries(tinySgemmConv_test64gcc tinySgemmConv)

else("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a")

add_executable(tinySgemmConv_test32gcc convolution_3x3s1.c main.c)
target_link_libraries(tinySgemmConv_test32gcc tinySgemmConv)

endif ()

endif ()
