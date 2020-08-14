cmake_minimum_required( VERSION 3.3 )
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
project( rgbdrec )

option( WITH_K4A "Enable K4A (Azure Kinect); either this or FREENECT2 needed for demo" ON )
option( WITH_FREENECT2 "Enable Freenect2 (Kinect v2); either this or K4A needed for demo" ON )

set( INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include" )
set( SRC_DIR "${PROJECT_SOURCE_DIR}/src" )

find_package( Threads REQUIRED )
find_package( OpenCV REQUIRED )
find_package( ZLIB REQUIRED )
find_package( Eigen3 REQUIRED )
set( Boost_USE_STATIC_LIBS ON )
set( Boost_USE_STATIC ON )
find_package( Boost REQUIRED COMPONENTS filesystem program_options )

find_package( meshview )
if (NOT meshview_FOUND)
    message( WARNING "meshview not found, 3D visualization is disabled. Please install from https://github.com/sxyu/meshview" )
endif()

set(
  DEPENDENCIES
  ${CMAKE_THREAD_LIBS_INIT}
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
  ZLIB::ZLIB
  Eigen3::Eigen
)

# require K4A?
set( k4a_FOUND OFF )
if ( ${WITH_K4A} )
    find_package( k4a )
    if( k4a_FOUND)
      message( STATUS "Found k4a (Azure Kinect SDK)" )
    else()
      message( STATUS "k4a (Azure Kinect SDK) not found, live demo, etc disabled" )
    endif( k4a_FOUND )
    set( _FOUND_CAMERA_LIB ON )
else()
    message( STATUS "K4A (Azure Kinect SDK) disabled, configure with -DWITH_K4A=ON to enable" )
endif ()

set( freenect2_FOUND OFF )
if ( ${WITH_FREENECT2} )
    find_package( freenect2 )
    if( freenect2_FOUND )
      message( STATUS "Found libfreenect2" )
    else()
      message( STATUS "libfreenect2 not found" )
    endif( freenect2_FOUND )
    set( _FOUND_CAMERA_LIB ON )
else()
    message( STATUS "Freenect2 disabled, configure with -DWITH_FREENECT2=ON to enable" )
endif ()

#  Set up configs
set (FOUND_CAMERA_LIB FALSE)
if( NOT k4a_FOUND )
  set( _AZURE_KINECT_SDK_ "//" )
else()
  set( _AZURE_KINECT_SDK_ "" )
  set( SOURCES ${SOURCES} AzureKinectCamera.cpp )
  set( HEADERS ${HEADERS} ${INCLUDE_DIR}/AzureKinectCamera.h )
  list(APPEND DEPENDENCIES k4a::k4a)
  if (NOT FOUND_CAMERA_LIB)
      set(FOUND_CAMERA_LIB TRUE)
      set( _PREFERRED_CAMERA_ "AzureKinectCamera" )
  endif()
endif( NOT k4a_FOUND )

if( NOT freenect2_FOUND )
  set( _FREENECT2_ "//" )
else()
  set( _FREENECT2_ "" )
  set( SOURCES ${SOURCES} Freenect2Camera.cpp )
  set( HEADERS ${HEADERS} ${INCLUDE_DIR}/Freenect2Camera.h )
  list(APPEND DEPENDENCIES freenect2)
  if (NOT FOUND_CAMERA_LIB)
      set(FOUND_CAMERA_LIB TRUE)
      set( _PREFERRED_CAMERA_ "Freenect2Camera" )
  endif()
endif( NOT freenect2_FOUND )

if (NOT FOUND_CAMERA_LIB)
    message (WARNING No camera lib found, live demo and data recoding will not be built)
endif()

configure_file( Version.h.in "${PROJECT_BINARY_DIR}/include/Version.h" )

include_directories (
    ${INCLUDE_DIR}
    ${PROJECT_BINARY_DIR}/include
    ${Boost_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

file(GLOB_RECURSE SOURCES ${SRC_DIR}/*.cpp)
file(GLOB_RECURSE HEADERS ${INCLUDE_DIR}/*.h)

add_executable(rgbdrec ${SOURCES} rgbdrec.cpp)
target_link_libraries( rgbdrec ${DEPENDENCIES} )
install(TARGETS rgbdrec DESTINATION bin)

add_executable(rgbdpose-colmap ${SOURCES} rgbdpose-colmap.cpp)
target_link_libraries( rgbdpose-colmap ${DEPENDENCIES} )
install(TARGETS rgbdpose-colmap DESTINATION bin)

if (meshview_FOUND)
    add_executable(rgbdpose-viewer ${SOURCES} rgbdpose-viewer.cpp)
    target_link_libraries( rgbdpose-viewer ${DEPENDENCIES} meshview::meshview )
    install(TARGETS rgbdpose-viewer DESTINATION bin)
endif()
