cmake_minimum_required(VERSION 2.8.0)

# project( boost.numpy )

# put our local cmake find scripts at the beginning of the cmake
# module search path
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/libs/numpy/cmake ${CMAKE_MODULE_PATH})

# configure output folders so artifacts are built into a single set of
# top-level folders rather than the default cmake build structure that
# matches the source tree.

# find required python packages
find_package(PythonLibs 2.7 REQUIRED)
find_package(NumPy REQUIRED)

# find boost
# 
# set(Boost_USE_STATIC_LIBS ON)
# set(Boost_USE_MULTITHREADED ON) 
# set(Boost_USE_STATIC_RUNTIME ON)
FIND_PACKAGE(Boost COMPONENTS python REQUIRED)

message( STATUS "found boost:"
  "\nINCLUDE: ${Boost_INCLUDE_DIRS}"
  "\nLIB: ${Boost_LIBRARIES}"
)

# add_definitions( -DBOOST_ALL_NO_LIB )
# In some cases, you may need to explicitly specify that a dynamic Boost is used; if so use:
# add_definitions( -DBOOST_ALL_DYN_LINK )

# variable controlling whether the boost_numpy is a shared or static library
if (WIN32)
  set(LIBRARY_TYPE STATIC CACHE STRING "type of library to make for boost_numpy")
else()
  set(LIBRARY_TYPE SHARED CACHE STRING "type of library to make for boost_numpy")
endif()

# variable controlling building of documentation
set(BUILD_DOCS OFF CACHE BOOL "Build Boost.NumPy Documentation")

# logic for configuring documentation generation
if(BUILD_DOCS)
  # find sphinx
  EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} -c "import sphinx; print sphinx.__version__"
    RESULT_VARIABLE SPHINX_PROCESS
    OUTPUT_VARIABLE SPHINX_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  
  set(HAVE_SPHINX 0)
  if(SPHINX_PROCESS EQUAL 0)
    FIND_PROGRAM(SPHINX_BUILD sphinx-build)
    if(SPHINX_BUILD)
      set(HAVE_SPHINX 1)
      message(STATUS "    Found Sphinx ${SPHINX_VERSION}: ${SPHINX_BUILD}")
    else()
      # sphinx is required to generate documention, so it is an error
      # if we cannot find it
      MESSAGE(SEND_ERROR "must be able to find sphinx-build when BUILD_DOCS is enabled")
    endif()
  endif()
  
  # find pdflatex, which is only required for doc-pdf
  FIND_PACKAGE(LATEX)
  if (PDFLATEX_COMPILER)
    message( STATUS "Found PDFLATEX_COMPILER=${PDFLATEX_COMPILER}" )
  else()
    message( STATUS "Found PDFLATEX_COMPILER NOT found" )
  endif()
endif()

# compiler definitions for non-windows builds
if (NOT WIN32)
  add_definitions(-fPIC)
endif()

# enable ctest targets
ENABLE_TESTING()

# turn on visual studio solution folders
SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

# global settings for include paths
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PYTHON_INCLUDE_DIRS}
  ${NUMPY_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS} 
  )

# install headers
install(DIRECTORY boost
    DESTINATION "include"
    FILES_MATCHING
    PATTERN "*.hpp"
    ${INSTALL_PERMISSIONS_SRC}
  )

# add submodules
ADD_SUBDIRECTORY(libs/numpy/src)
# ADD_SUBDIRECTORY(libs/numpy/example)
# ADD_SUBDIRECTORY(libs/numpy/test)

if (BUILD_DOCS)
  ADD_SUBDIRECTORY(libs/numpy/doc)
endif()


