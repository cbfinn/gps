#include "assert.h"
#include "mujoco_osg_viewer.hpp"
#include <string>
#include "mj_engine.h"
#include "mj_user.h"
#include "mj_xml.h"
#include "unistd.h"
#include <iostream>
#include <osg/io_utils>
#include <sstream>
#include <vector>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osg/Material>
#include <osg/Texture2D>
#include <osg/Array>
#include <osg/TexEnv>
#include <osgGA/NodeTrackerManipulator>
#include <osgGA/TrackballManipulator>
#include <osgGA/KeySwitchMatrixManipulator>

#include "macros.h"

#define PLOT_JOINTS 0

struct EventHandler : public osgGA::GUIEventHandler
{
    EventHandler( MujocoOSGViewer* parent ) : 
    m_parent(parent),
    m_idling(false) {}
    bool handle( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa );
    MujocoOSGViewer* m_parent;
    bool m_idling;
};

bool EventHandler::handle(const osgGA::GUIEventAdapter& ea,osgGA::GUIActionAdapter& aa ) {
    osgGA::GUIEventAdapter::EventType t = ea.getEventType();
    if (t == osgGA::GUIEventAdapter::KEYDOWN) {
      int key = ea.getKey();
      switch (key) {
        case 'h':
            printf(
                "=== Mujoco OSG Viewer ===\n"
                "'h': Print help\n"
                "'p': Unpause\n"
                "'1': Set camera mode to trackball manipulator\n"
                "'2': Set camera mode to node tracker\n"
            );
            return true;
        case 'p':
            if (m_idling) 
            m_idling = false;
            return true;
        }
    }
    return osgGA::GUIEventHandler::handle(ea,aa);
}

// GROUND PLANE CODE RIPPED OFF FROM https://github.com/naderman/orca-robotics/blob/master/src/libs/orcaqgui3dfactory/gridelement.cpp
void makeCheckImage64x64x3( GLubyte img[64][64][3],
                            int numSquaresPerEdge,
                            int lowVal,
                            int highVal )
{


    const int widthInPixels=64;
    const int heightInPixels=64;
    assert( lowVal >= 0 && lowVal <= 255 );
    assert( highVal >= 0 && highVal <= 255 );

    int wOn=0;
    int hOn=0;
    for (int i = 0; i < widthInPixels; i++)
    {
        if ( (i % (widthInPixels/numSquaresPerEdge)) == 0 )
            wOn = wOn ? 0 : 1;

        for (int j = 0; j < heightInPixels; j++)
        {
            if ( (j % (heightInPixels/numSquaresPerEdge)) == 0 )
                hOn = hOn ? 0 : 1;

            int c = (wOn^hOn);
            if ( c==0 ) c = lowVal;
            else c = highVal;
            // cout<<"TRACE(glutil.cpp): hOn: " << hOn << ", wOn: " << wOn << ", c: " << c << endl;
            img[i][j][0] = (GLubyte) c;
            img[i][j][1] = (GLubyte) c;
            img[i][j][2] = (GLubyte) c;
        }
    }
}

osg::Image *createCheckImage()
{
    osg::Image *img = new osg::Image;
    img->allocateImage( 64, 64, 3, GL_RGB, GL_UNSIGNED_BYTE );

    GLubyte checkImage[64][64][3];

    // Draw the chess-board in memory
    makeCheckImage64x64x3( checkImage, 2, 12, 200 );

    // copy to the image
    int n=0;
    for ( uint i=0; i < 64; i++ )
        for ( uint j=0; j < 64; j++ )
            for ( uint k=0; k < 3; k++ )
                img->data()[n++] = checkImage[i][j][k];

    return img;
}




osg::Node* createGroundPlane(){
    //
    // Create the geode
    //
    osg::Geode* groundPlaneGeode_ = new osg::Geode;

    //
    // Create the texture
    //
    osg::ref_ptr<osg::Image> checkImage = createCheckImage();

    osg::ref_ptr<osg::Texture2D> checkTexture = new osg::Texture2D;
    // protect from being optimized away as static state:
    checkTexture->setDataVariance(osg::Object::DYNAMIC); 
    checkTexture->setImage( checkImage.get() );

    // Tell the texture to repeat
    checkTexture->setWrap( osg::Texture::WRAP_S, osg::Texture::REPEAT );
    checkTexture->setWrap( osg::Texture::WRAP_T, osg::Texture::REPEAT );

    // Create a new StateSet with default settings: 
    osg::ref_ptr<osg::StateSet> groundPlaneStateSet = new osg::StateSet();

    // Assign texture unit 0 of our new StateSet to the texture 
    // we just created and enable the texture.
    groundPlaneStateSet->setTextureAttributeAndModes(0,checkTexture.get(),osg::StateAttribute::ON);

    // Texture mode
    osg::TexEnv* texEnv = new osg::TexEnv;
    texEnv->setMode(osg::TexEnv::DECAL); // (osg::TexEnv::MODULATE);
    groundPlaneStateSet->setTextureAttribute(0,texEnv);

    // Associate this state set with our Geode
    groundPlaneGeode_->setStateSet(groundPlaneStateSet.get());

    //
    // Create the ground plane
    //
    osg::ref_ptr<osg::Geometry> groundPlaneGeometry = new osg::Geometry();
    double groundPlaneSquareSpacing_ = 1;

    const double infty=1000;

    const double metresPerTile=2*groundPlaneSquareSpacing_;
    const double texCoordExtreme=2*infty/metresPerTile;

    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array;
    groundPlaneGeometry->setVertexArray( vertices.get() );
    
    osg::ref_ptr<osg::Vec2Array> texCoords = new osg::Vec2Array;
    groundPlaneGeometry->setTexCoordArray( 0, texCoords.get() );

    vertices->push_back( osg::Vec3d( -infty, -infty, 0 ) );
    texCoords->push_back( osg::Vec2( 0, 0 ) );
    vertices->push_back( osg::Vec3d(  infty, -infty, 0 ) );
    texCoords->push_back( osg::Vec2( texCoordExtreme, 0 ) );
    vertices->push_back( osg::Vec3d(  infty,  infty, 0 ) );
    texCoords->push_back( osg::Vec2( texCoordExtreme, texCoordExtreme ) );
    vertices->push_back( osg::Vec3d( -infty,  infty, 0 ) );
    texCoords->push_back( osg::Vec2( 0, texCoordExtreme ) );

//     osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array;
//     colors->push_back(osg::Vec4(0.2, 0.2, 0.2, 1.0) );
//     // colors->push_back(osg::Vec4(0.0, 0.5, 0.0, 1.0) );
//     groundPlaneGeometry->setColorArray(colors.get());
//     groundPlaneGeometry->setColorBinding(osg::Geometry::BIND_PER_PRIMITIVE_SET);

    osg::ref_ptr<osg::DrawElementsUInt> quad = 
        new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS);
    for ( uint i=0; i < vertices->size(); i++ )
        quad->push_back( i );

    groundPlaneGeometry->addPrimitiveSet( quad.get() );

    groundPlaneGeode_->addDrawable( groundPlaneGeometry.get() );
    return groundPlaneGeode_;
}

osg::Node* createOSGNode(const mjModel* model, int i_geom) {
    int geom = model->geom_type[i_geom];
    mjtNum* size = model->geom_size + 3*i_geom;
    osg::Shape* shape = NULL;
    switch (geom) {
        case mjGEOM_PLANE: {
            return createGroundPlane();
        }
        // case mjGEOM_HFIELD:
        //     break;
        case mjGEOM_SPHERE: {
            shape = new osg::Sphere();
            ((osg::Sphere*)shape)->setRadius(size[0]);
            break;           
        } 
        case mjGEOM_CAPSULE: {
            shape = new osg::Capsule();
            ((osg::Capsule*)shape)->setRadius(size[0]);
            ((osg::Capsule*)shape)->setHeight(2*size[1]);
            break;            
        }
        #if 0
        case mjGEOM_ELLIPSOID: {
            osg::MatrixTransform* mt = new osg::MatrixTransform(osg::Matrix::scale(size[0], size[1], size[2]));
            osg::Sphere* sphere = new osg::Sphere();
            sphere->setRadius(1);
            osg::Geode* geode1 = new osg::Geode;
            geode1->addDrawable(new osg::ShapeDrawable(sphere));
            mt->addChild(geode1);
            geode->addChild(mt);
            break;
        }
        #endif
        case mjGEOM_CYLINDER: {            
            shape = new osg::Cylinder();
            ((osg::Cylinder*)shape)->setRadius(size[0]);
            ((osg::Cylinder*)shape)->setHeight(size[1]*2);
            break;
        }
            
        case mjGEOM_BOX: {
            shape = new osg::Box();
            ((osg::Box*)shape)->setHalfLengths(osg::Vec3(size[0],size[1],size[2]));
            break;
        }
        case mjGEOM_MESH: {
            int id = model->geom_dataid[i_geom];
            shape = new osg::TriangleMesh;
            osg::Vec3Array* vertices = new osg::Vec3Array;
            osg::IntArray* indices = new osg::IntArray;
            for (int j=0; j < model->mesh_vertnum[id]; ++j) {
                float* p = model->mesh_vert + 3*model->mesh_vertadr[id] + 3*j;
                vertices->push_back(osg::Vec3f(p[0], p[1], p[2]));
            }
            for (int j=0; j < model->mesh_facenum[id]; ++j) {
                int* p = model->mesh_face + 3*model->mesh_faceadr[id] + 3*j;
                indices->push_back( p[0] );
                indices->push_back( p[1] );
                indices->push_back( p[2] );

            }

            ((osg::TriangleMesh*)shape)->setIndices(indices);
            ((osg::TriangleMesh*)shape)->setVertices(vertices);
            break;
        }
        default:
            printf("unimplemented geom type: %i\n",geom);
            break;
    }
    osg::Geode* geode = new osg::Geode;    
    osg::ShapeDrawable* drawable = new osg::ShapeDrawable(shape);
    float* p = model->geom_rgba + i_geom*4;
    drawable->setColor(osg::Vec4(p[0],p[1],p[2],p[3]));
    geode->addDrawable(drawable);

    return geode;
}



MujocoOSGViewer::MujocoOSGViewer() 
: m_option(NULL), m_data(NULL), m_model(NULL)
{
    m_root = new osg::Group;
    m_robot = new osg::Group;
    m_root->addChild(m_robot);
    m_viewer.setSceneData(m_root.get());
    m_viewer.setUpViewInWindow(0, 0, 640, 480);
    m_viewer.realize();

    osg::ref_ptr<osgGA::TrackballManipulator> man = new osgGA::TrackballManipulator;
    man->setHomePosition(osg::Vec3(2, 3, 2), osg::Vec3(0, 0, 0), osg::Vec3(-1, -1.5, 2));
    m_viewer.setCameraManipulator(man);

    m_handler = new EventHandler(this);
    m_viewer.addEventHandler(m_handler.get());
}

void MujocoOSGViewer::Idle() {    
    EventHandler* handler = dynamic_cast<EventHandler*>(m_handler.get());
    handler->m_idling = true;
    mj_kinematics(m_model, m_option, m_data);    
    _UpdateTransforms();
    while (handler->m_idling && !m_viewer.done()) {
        m_viewer.frame();
        OpenThreads::Thread::microSleep(30000);   
    }
}

void MujocoOSGViewer::RenderOnce() {
    mj_kinematics(m_model,m_option,m_data);    
    _UpdateTransforms();
    m_viewer.frame();
}

void MujocoOSGViewer::_UpdateTransforms() {
    for (int i=0; i < m_model->ngeom; ++i) {
        mjtNum* tptr = m_data->geom_xpos + 3*i,
              * rptr = m_data->geom_xmat + 9*i;
        osg::Matrix mat(rptr[0],rptr[3],rptr[6],0,rptr[1],rptr[4],rptr[7],0,rptr[2],rptr[5],rptr[8],0,tptr[0],tptr[1],tptr[2],1);
        m_geomTFs[i]->setMatrix(mat);
    }
#if PLOT_JOINTS
    for (int i=0; i < m_model->njnt; ++i) {
        if (m_model->jnt_type[i] == mjJNT_HINGE) {        
            mjtNum* panchor = m_data->xanchor + 3*i,
                  * pax = m_data->xaxis + 3*i;
            osg::Vec3 anchor(panchor[0], panchor[1], panchor[2]);
            osg::Quat q;
            q.makeRotate(osg::Vec3(0,0,1), osg::Vec3(pax[0], pax[1], pax[2]));
            osg::Matrix mat;
            mat.setTrans(anchor);
            mat.setRotate(q);
            m_axTFs[i]->setMatrix(mat);
        }
    }
#endif
}

void MujocoOSGViewer::HandleInput() {
    // TODO
}

void MujocoOSGViewer::StartAsyncRendering() {
    
}

void MujocoOSGViewer::StopAsyncRendering() {
    // TODO
}

void MujocoOSGViewer::SetModel(const mjModel* m) {
    m_model = m;
    if (m_option!=NULL) mj_deleteOption(m_option);
    m_option = mj_makeOption();
    if (m_data!=NULL) mj_deleteData(m_data);
    m_data = mj_makeData(m_model,m_option);

    m_geomTFs.clear();
    for (int i=0; i < m->ngeom; ++i) {
        osg::MatrixTransform* geom_tf = new osg::MatrixTransform;
        m_geomTFs.push_back(geom_tf);
        if (i==0) {m_root->addChild(geom_tf);}
        else {m_robot->addChild(geom_tf);}
        osg::Node* node = createOSGNode(m,i);
       // FAIL_IF_FALSE(!!node);
        if (!!node) {
            geom_tf->addChild(node);
        }
        else printf("SKIPPING\n");

    }

#if PLOT_JOINTS
    m_axTFs.resize(m->njnt);
    for (int i=0; i < m->njnt; ++i) {
        if (m_model->jnt_type[i] == mjJNT_HINGE) {
            printf("jnt type %i %i\n",i,m_model->jnt_type[i]);
            osg::Cylinder* shape = new osg::Cylinder();
            shape->setRadius(0.01);
            shape->setHeight(0.3);
            osg::ShapeDrawable* drawable = new osg::ShapeDrawable(shape);
            drawable->setColor(osg::Vec4(1,1,0,1));
            osg::Geode* geode = new osg::Geode;    
            geode->addDrawable(drawable);
            osg::MatrixTransform* tf = new osg::MatrixTransform;
            tf->addChild(geode);
            m_root->addChild(tf);
            m_axTFs[i] = tf;
        }
    }
#endif
}

void MujocoOSGViewer::SetData(const mjData* d) {
    mju_copy(m_data->qpos, d->qpos, m_model->nq);
}


void NewModelFromXML(const char* filename,mjModel*& model, mjOption*& option) {
    char errmsg[100];
    mjCModel* usermodel = mjParseXML(filename, errmsg);
    if( !usermodel ) {
        printf("%s\n",errmsg);
    }
    else {
        model = usermodel->Compile();
        if( !model ) {
            strcpy(errmsg, usermodel->GetError().message);
            printf("%s\n",errmsg);
        }
        option = mj_makeOption();
        *option = usermodel->option;
        delete usermodel;
    }   
}

