cmake_minimum_required(VERSION 3.5)
PROJECT(libtinySgemmConv)

message(CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR})
message(CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR})
message(PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR})

option(ENABLE_OPENMP "openmp support" ON)

ENABLE_LANGUAGE(ASM)

file(GLOB LIB_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
file(GLOB LIB_CSRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
file(GLOB LIB_SSRC ${CMAKE_CURRENT_SOURCE_DIR}/src/arm64/*.S)
message(STATUS "ARCH: ARM64")
else("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a")
file(GLOB LIB_SSRC ${CMAKE_CURRENT_SOURCE_DIR}/src/arm32/*.S)
message(STATUS "ARCH: ARM32")
endif()

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wno-unused-function -Wno-unused-parameter")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden -frtti")
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
message(STATUS "compile use clang")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
message(STATUS "compile use gcc")

endif ()

if( CMAKE_SYSTEM_PROCESSOR MATCHES armv7-a )
add_definitions(-mfpu=neon-fp16)
endif()
message(STATUS "C flags: ${CMAKE_C_FLAGS}")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

ADD_LIBRARY(tinySgemmConv SHARED ${LIB_CSRC} ${LIB_SSRC})
ADD_LIBRARY(tinySgemmConv_static STATIC ${LIB_CSRC} ${LIB_SSRC})

SET_TARGET_PROPERTIES(tinySgemmConv PROPERTIES VERSION 1.2 SOVERSION 1)
SET_TARGET_PROPERTIES(tinySgemmConv_static PROPERTIES OUTPUT_NAME "tinySgemmConv")

set(SGEMM_INSTALL_DIR "${PROJECT_BINARY_DIR}/install")

message(CMAKE_INSTALL_PREFIX: ${SGEMM_INSTALL_DIR})
message(Library headers: ${LIB_HEADERS})

install(TARGETS tinySgemmConv tinySgemmConv_static 
LIBRARY DESTINATION "${SGEMM_INSTALL_DIR}/lib"
ARCHIVE DESTINATION "${SGEMM_INSTALL_DIR}/libstatic"
)
install(FILES ${LIB_HEADERS} DESTINATION "${SGEMM_INSTALL_DIR}/include")
