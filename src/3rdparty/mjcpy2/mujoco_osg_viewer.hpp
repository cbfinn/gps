#pragma once
#include <osg/Geometry>
#include <osgViewer/Viewer>
#include <osg/ref_ptr>
#include "mujoco.h"
//#include "mj_engine.h"


class MujocoOSGViewer {
public:
	MujocoOSGViewer();
	MujocoOSGViewer(osg::Vec3, osg::Vec3);
	void Idle(); // Block and draw in a loop until the 'p' key is pressed
	void RenderOnce();
	void HandleInput();
	void StartAsyncRendering();
	void StopAsyncRendering(); // 
	void SetModel(const mjModel*);
	void SetData(const mjData*);
	void _UpdateTransforms();
	void SetCamera(float x, float y, float z, float px, float py, float pz);

  // osg::ref_ptr<EventHandler> m_handler;
  mjData* m_data;  
  const mjModel* m_model;
  osg::ref_ptr<osg::Group> m_root, m_robot;
  osg::ref_ptr<osg::Image> m_image;
  osgViewer::Viewer m_viewer;
  std::vector<osg::MatrixTransform*> m_geomTFs, m_axTFs;
  osg::ref_ptr<osgGA::GUIEventHandler> m_handler;
};


void NewModelFromXML(const char* filename,mjModel*&);
