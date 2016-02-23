#include "mujoco_osg_viewer.hpp"
#include "macros.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("usage: test_mujoco_osg /path/to/file.xml\n");
        exit(1);
    }
    const char* fname = argv[1];

	mjOption* option;
	mjModel* model;
    NewModelFromXML(fname, model, option);
    mjData* data = mj_makeData(model, option);
    if (!model) PRINT_AND_THROW("couldn't load model: " + std::string(fname));
    MujocoOSGViewer viewer;
    viewer.SetModel(model);
    viewer.SetData(data);
    viewer.Idle();

    while (true) {
        mj_step(model, option, data);
        viewer.SetData(data);
        viewer.RenderOnce();
        OpenThreads::Thread::microSleep(30000);   
    }

}
