# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/dongnan/SLF/SLFEIC/build/3rdparty/pybind11/pybind11-src")
  file(MAKE_DIRECTORY "/home/dongnan/SLF/SLFEIC/build/3rdparty/pybind11/pybind11-src")
endif()
file(MAKE_DIRECTORY
  "/home/dongnan/SLF/SLFEIC/build/3rdparty/pybind11/pybind11-build"
  "/home/dongnan/SLF/SLFEIC/build/3rdparty/pybind11/pybind11-download/pybind11-prefix"
  "/home/dongnan/SLF/SLFEIC/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/tmp"
  "/home/dongnan/SLF/SLFEIC/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/src/pybind11-stamp"
  "/home/dongnan/SLF/SLFEIC/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/src"
  "/home/dongnan/SLF/SLFEIC/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/src/pybind11-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/dongnan/SLF/SLFEIC/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/src/pybind11-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/dongnan/SLF/SLFEIC/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/src/pybind11-stamp${cfgdir}") # cfgdir has leading slash
endif()
