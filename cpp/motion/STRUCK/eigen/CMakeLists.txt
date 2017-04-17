project(Eigen)
cmake_minimum_required(VERSION 2.6.2)

#set(INCLUDE_INSTALL_DIR
#    "${CMAKE_INSTALL_PREFIX}/include/eigen2"
#    CACHE PATH
#    "The directory where we install the header files"
#    FORCE)

set(EIGEN_VERSION_NUMBER "2.0.17")
set(EIGEN_VERSION "${EIGEN_VERSION_NUMBER}")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_INCLUDE_CURRENT_DIR ON)


set(EIGEN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)


#############################################################################
# find how to link to the standard libraries                                #
#############################################################################

# find_package(StandardMathLibrary)

# set(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO "")

# if(NOT STANDARD_MATH_LIBRARY_FOUND)

#   message(FATAL_ERROR
#     "Can't link to the standard math library. Please report to the Eigen developers, telling them about your platform.")

# else()

#   if(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO)
#     set(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO "${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO} ${STANDARD_MATH_LIBRARY}")
#   else()
#     set(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO "${STANDARD_MATH_LIBRARY}")
#   endif()

# endif()



# if(CMAKE_COMPILER_IS_GNUCXX)
#   if(CMAKE_SYSTEM_NAME MATCHES Linux)
#     include(CheckCXXCompilerFlag)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnon-virtual-dtor -Wno-long-long -ansi -Wundef -Wcast-align -Wchar-subscripts -Wall -W -Wpointer-arith -Wwrite-strings -Wformat-security -fno-exceptions -fno-check-new -fno-common -fstrict-aliasing")
#     check_cxx_compiler_flag("-Wextra" has_wextra)
#     if(has_wextra)
#       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
#     endif()
#     if(NOT EIGEN_TEST_LIB)
#       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")
#     endif(NOT EIGEN_TEST_LIB)
#   endif(CMAKE_SYSTEM_NAME MATCHES Linux)
# endif(CMAKE_COMPILER_IS_GNUCXX)


#include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})


#configure_file(eigen2.pc.in eigen2.pc) # uses INCLUDE_INSTALL_DIR
#install(FILES ${CMAKE_CURRENT_BINARY_DIR}/eigen2.pc DESTINATION share/pkgconfig)


#add_subdirectory(Eigen)
#add_subdirectory(unsupported)
