#pragma once
#include <osg/Geometry>
#include <osgViewer/Viewer>
#include <osg/ref_ptr>
#include "mj_engine.h"


class MujocoOSGViewer {
public:
	MujocoOSGViewer();
	void Idle(); // Block and draw in a loop until the 'p' key is pressed
	void RenderOnce();
	void HandleInput();
	void StartAsyncRendering();
	void StopAsyncRendering(); // 
	void SetModel(const mjModel*);
	void SetData(const mjData*);
	void _UpdateTransforms();  

  // osg::ref_ptr<EventHandler> m_handler;
  mjOption* m_option;
  mjData* m_data;  
  const mjModel* m_model;
  osg::ref_ptr<osg::Group> m_root, m_robot;
  osgViewer::Viewer m_viewer;
  std::vector<osg::MatrixTransform*> m_geomTFs, m_axTFs;
  osg::ref_ptr<osgGA::GUIEventHandler> m_handler;
};


void NewModelFromXML(const char* filename,mjModel*&,mjOption*&);
