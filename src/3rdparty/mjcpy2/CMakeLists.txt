include_directories(SYSTEM "${CMAKE_SOURCE_DIR}")
include_directories(SYSTEM "${CMAKE_SOURCE_DIR}/Boost.NumPy")

include_directories(SYSTEM 
    "${CMAKE_SOURCE_DIR}/mjpro"
)
# boost_python_module(mjcpy2 mjcpy2.cpp mjosgviewer.cpp)
# target_link_libraries(mjcpy mjc boost_numpy)

add_definitions("-DmjUSEDOUBLE")


# if (APPLE)
if(APPLE)
	include_directories(SYSTEM
		"/Users/xyz/Src/osg/include"
		"/Users/xyz/build/osg/include"
		)
	link_directories("/Users/xyz/build/osg/lib")
	set(OSG_LIBRARIES 
		osg
		osgViewer
		OpenThreads
		osgGA
		osgUtil
		osgText
		osgDB
		)
else()
	find_package( osg )
	find_package( osgViewer )
	find_package( OpenThreads )
	find_package(osgGA)       
	# link_directories(${OSG})
	set(OSG_LIBRARIES ${OSG_LIBRARY} ${OSGVIEWER_LIBRARY} ${OPENTHREADS_LIBRARY} ${OSGGA_LIBRARY} )       
endif()


include_directories(SYSTEM ${OSG_INCLUDE_DIR})
message("osg includes: ${OSG_INCLUDE_DIR}")
# add_definitions("-DOPENTHREADS_ATOMIC_USE_MUTEX")
# add_executable(test_mujoco_osg test_mujoco_osg.cpp mujoco_osg_viewer.cpp)
# target_link_libraries(test_mujoco_osg mujoco ${OSG_LIBRARIES} stdc++ pthread)


include_directories(SYSTEM "${CMAKE_SOURCE_DIR}")
include_directories(SYSTEM "${CMAKE_SOURCE_DIR}/Boost.NumPy")
boost_python_module(mjcpy mjcpy2.cpp  mujoco_osg_viewer.cpp)
target_link_libraries(mjcpy "${CMAKE_SOURCE_DIR}/mjpro/libmujoco.so" ${OSG_LIBRARIES} boost_numpy)
