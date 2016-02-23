#include <boost/numpy.hpp>
#include <cmath>
#include "macros.h"
#include <iostream>
#include <boost/python/slice.hpp>
#include "mujoco_osg_viewer.hpp"

namespace bp = boost::python;
namespace bn = boost::numpy;




namespace {

bp::object main_namespace;

template<typename T>
bn::ndarray toNdarray1(const T* data, long dim0) {
  long dims[1] = {dim0};
  bn::ndarray out = bn::empty(1, dims, bn::dtype::get_builtin<T>());
  memcpy(out.get_data(), data, dim0*sizeof(T));
  return out;
}
template<typename T>
bn::ndarray toNdarray2(const T* data, long dim0, long dim1) {
  long dims[2] = {dim0,dim1};
  bn::ndarray out = bn::empty(2, dims, bn::dtype::get_builtin<T>());
  memcpy(out.get_data(), data, dim0*dim1*sizeof(T));
  return out;
}
template<typename T>
bn::ndarray toNdarray3(const T* data, long dim0, long dim1, long dim2) {
  long dims[3] = {dim0,dim1,dim2};
  bn::ndarray out = bn::empty(3, dims, bn::dtype::get_builtin<T>());
  memcpy(out.get_data(), data, dim0*dim1*dim2*sizeof(T));
  return out;
}


bool endswith(const std::string& fullString, const std::string& ending)
{
	return (fullString.length() >= ending.length()) && 
		(0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
}




class PyMJCWorld2 {


public:

    PyMJCWorld2(const std::string& loadfile);
    bp::object Step(const bn::ndarray& x, const bn::ndarray& u);
    bp::object StepMulti(const bn::ndarray& x, const bn::ndarray& u);
    void Plot(const bn::ndarray& x);    
    void Idle(const bn::ndarray& x);
    bn::ndarray GetCOMMulti(const bn::ndarray& x);
    bn::ndarray GetJacSite(int site);
    void Kinematics();
    bp::dict GetModel();
    void SetModel(bp::dict d);
    bp::dict GetOption();
    void SetOption(bp::dict d);
    bp::dict GetData();
    void SetData(bp::dict d);
    bn::ndarray GetContacts(const bn::ndarray& x);
    bp::object GetFeatDesc();
    void SetFeats(const bp::list& feats);
    bn::ndarray GetImage(const bn::ndarray& x);
    void SetNumSteps(int n) {m_numSteps=n;}

    ~PyMJCWorld2();
private:
    // PyMJCWorld(const PyMJCWorld&) {}

    void _PlotInit();
    int _CopyFeat(mjtNum* ptr);
    int _FeatSize();

    // void _SetState(const mjtNum* xdata) {mju_copy(m_data->qpos, (xdata), NQ); mju_copy(m_data->qvel, (xdata)+NQ, NV); }
    // void _SetControl(const mjtNum* udata) {for (int i=0; i < m_actuatedDims.size(); ++i) m_u[m_actuatedDims[i]] = (udata)[i];}


    mjModel* m_model;
    mjOption* m_option;
    mjData* m_data;
    MujocoOSGViewer* m_viewer;
    int m_numSteps;
    int m_featmask;

};

PyMJCWorld2::PyMJCWorld2(const std::string& loadfile) {
  	if (endswith(loadfile, "xml")) {
    		NewModelFromXML(loadfile.c_str(), m_model, m_option);		
  	}
  	else {
  	    NOTIMPLEMENTED;
  	}	
    if (!m_model) PRINT_AND_THROW("couldn't load model: " + std::string(loadfile));
    FAIL_IF_FALSE(!!m_option);
    m_option->disableflags |= mjDSBL_WARMSTART;
    m_option->disableflags |= mjDSBL_PERTURB;
    m_option->disableflags |= mjDSBL_ENERGY;
    m_option->disableflags |= mjDSBL_CLEARCTRL;
    m_option->disableflags |= mjDSBL_CBCTRL;
    m_option->disableflags |= mjDSBL_CBEND;
    m_option->disableflags |= mjDSBL_CLAMPVEL;
    m_option->integrator = mjINT_RK4;
    m_data = mj_makeData(m_model,m_option);
    FAIL_IF_FALSE(!!m_data);
    m_viewer = NULL;
    m_numSteps = 1;
    m_featmask = 0;
}


PyMJCWorld2::~PyMJCWorld2() {
	if (m_viewer) {
		delete m_viewer;
	}
	mj_deleteData(m_data);
	mj_deleteOption(m_option);
	mj_deleteModel(m_model);
}

void _GetMinContactDist(mjModel* m, mjData* d, mjtNum* mindist) {
    for (int i=0; i < m->ngeom; ++i) {
        mindist[i] = m->geom_mindist[i];
    }
    for (int i=0; i < d->nc; ++i) {
        const mjContact& c = d->contact[i];
        // float contactdist = (c.dist != 0) ? c.dist : c.mindist;
        // mindist[c.geom1] = fmin(mindist[c.geom1],c.dist);
        mindist[c.geom2] = fmin(mindist[c.geom2],c.dist);
    }
}


enum _FEAT {
    feat_cdof                    = 1<<0,
    feat_cinert                  = 1<<1,
    feat_cvel                    = 1<<2,
    feat_cacc                    = 1<<3,
    feat_qfrc_bias               = 1<<4,
    feat_qfrc_passive            = 1<<5,
    feat_qfrc_actuation          = 1<<6,
    feat_qfrc_impulse            = 1<<7,
    feat_qfrc_constraint         = 1<<8,
    feat_cfrc_ext                = 1<<9,
    feat_contactdists            = 1<<10
};
#define size_cdof       6*m_model->nv
#define size_cinert          10*m_model->nbody
#define size_cvel        6*m_model->nbody
#define size_cacc        6*m_model->nbody
#define size_qfrc_bias       m_model->nv
#define size_qfrc_passive        m_model->nv
#define size_qfrc_actuation          m_model->nv
#define size_qfrc_impulse        m_model->nv
#define size_qfrc_constraint        m_model->nv
#define size_cfrc_ext       6*m_model->nbody
#define size_contactdists   m_model->ngeom


#define ADDFEAT(featname) featmap[#featname] = feat_##featname
void PyMJCWorld2::SetFeats(const bp::list&  feats) {    
    std::map<std::string, int> featmap;

    ADDFEAT(cdof);
    ADDFEAT(cinert);
    ADDFEAT(cvel);
    ADDFEAT(cacc);
    ADDFEAT(qfrc_bias);
    ADDFEAT(qfrc_passive);
    ADDFEAT(qfrc_actuation);
    ADDFEAT(qfrc_impulse);
    ADDFEAT(qfrc_constraint);
    ADDFEAT(cfrc_ext);
    ADDFEAT(contactdists);

    m_featmask = 0;
    bp::ssize_t n = bp::len(feats);
    for(bp::ssize_t i=0;i<n;i++) {
        std::string s = bp::extract<std::string>(feats[i]);
        FAIL_IF_FALSE(featmap.find(s) != featmap.end());
        m_featmask |= featmap[s];
    }
}
#undef ADDFEAT


#define ADDFEAT(featname)                          \
    if (m_featmask & feat_##featname) {                 \
        mju_copy(ptr, m_data->featname, size_##featname);  \
        ptr += (size_##featname);   \
    }    
int PyMJCWorld2::_CopyFeat(mjtNum* ptrstart) {
    mjtNum* ptr = ptrstart;
    ADDFEAT(cdof);
    ADDFEAT(cinert);
    ADDFEAT(cvel);
    ADDFEAT(cacc);
    ADDFEAT(qfrc_bias);
    ADDFEAT(qfrc_passive);
    ADDFEAT(qfrc_actuation);
    ADDFEAT(qfrc_impulse);
    ADDFEAT(qfrc_constraint);
    ADDFEAT(cfrc_ext);
    if (m_featmask & feat_contactdists) {        
        _GetMinContactDist(m_model,m_data,ptr);
        ptr += size_contactdists;
    }
    return ptr-ptrstart;
}
#undef ADDFEAT


#define ADDFEAT(featname)                       \
    if (m_featmask & feat_##featname) {              \
        out.append(bp::make_tuple(#featname,size_##featname));        \
    }
bp::object PyMJCWorld2::GetFeatDesc() {
    bp::list out;
    ADDFEAT(cdof);
    ADDFEAT(cinert);
    ADDFEAT(cvel);
    ADDFEAT(cacc);
    ADDFEAT(qfrc_bias);
    ADDFEAT(qfrc_passive);
    ADDFEAT(qfrc_actuation);
    ADDFEAT(qfrc_impulse);
    ADDFEAT(qfrc_constraint);
    ADDFEAT(cfrc_ext);
    ADDFEAT(contactdists);
    return out;
}
#undef ADDFEAT

#define ADDFEAT(featname) if (m_featmask & feat_##featname) size += size_##featname
int PyMJCWorld2::_FeatSize() {
    int size=0;
    ADDFEAT(cdof);
    ADDFEAT(cinert);
    ADDFEAT(cvel);
    ADDFEAT(cacc);
    ADDFEAT(qfrc_bias);
    ADDFEAT(qfrc_passive);
    ADDFEAT(qfrc_actuation);
    ADDFEAT(qfrc_impulse);
    ADDFEAT(qfrc_constraint);
    ADDFEAT(cfrc_ext);
    ADDFEAT(contactdists);
    return size;
}
#undef ADDFEAT

int StateSize(mjModel* m) {
    return m->nq + m->nv;
}
void GetState(mjtNum* ptr, const mjModel* m, const mjData* d) {
    mju_copy(ptr, d->qpos, m->nq);
    ptr += m->nq;
    mju_copy(ptr, d->qvel, m->nv);
}
void SetState(const mjtNum* ptr, const mjModel* m, mjData* d) {
    mju_copy(d->qpos, ptr, m->nq);
    ptr += m->nq;
    mju_copy(d->qvel, ptr, m->nv);
}
inline void SetCtrl(const mjtNum* ptr, const mjModel* m, mjData* d) {
    mju_copy(d->ctrl, ptr, m->nu);
}

#define MJTNUM_DTYPE bn::dtype::get_builtin<mjtNum>()

bp::object PyMJCWorld2::Step(const bn::ndarray& x, const bn::ndarray& u) {
    FAIL_IF_FALSE(x.get_dtype() == MJTNUM_DTYPE && x.get_nd() == 1 && x.get_flags() & bn::ndarray::C_CONTIGUOUS && x.shape(0) == m_model->nq+m_model->nv);
    FAIL_IF_FALSE(u.get_dtype() == MJTNUM_DTYPE && u.get_nd() == 1 && u.get_flags() & bn::ndarray::C_CONTIGUOUS && u.shape(0) == m_model->nu);

    SetState(reinterpret_cast<const mjtNum*>(x.get_data()), m_model, m_data);

    mj_step1(m_model,m_option,m_data);
    SetCtrl(reinterpret_cast<const mjtNum*>(u.get_data()), m_model, m_data);
    mj_step2(m_model,m_option,m_data);

    // mj_kinematics(m_model, m_option, m_data);
    // printf("before: %f\n", m_data->com[0]);
/*
    for (int i=0; i < m_numSteps; ++i) mj_step(m_model,m_option,m_data);
    if (m_featmask & feat_cfrc_ext) mj_rnePost(m_model, m_option, m_data);
*/
    // printf("after step: %f\n", m_data->com[0]);
    // mj_kinematics(m_model, m_option, m_data);
    // printf("after step+kin: %f\n", m_data->com[0]);

    long xdims[1] = {StateSize(m_model)};
    long odims[1] = {_FeatSize()};
    bn::ndarray xout = bn::empty(1, xdims, bn::dtype::get_builtin<mjtNum>());
    bn::ndarray oout = bn::empty(1, odims, bn::dtype::get_builtin<mjtNum>());

    GetState((mjtNum*)xout.get_data(), m_model, m_data);
    _CopyFeat((mjtNum*)oout.get_data());

	return bp::make_tuple(xout, oout);
}



bp::object PyMJCWorld2::StepMulti(const bn::ndarray& x, const bn::ndarray& u) {
    int state_size = StateSize(m_model);
    FAIL_IF_FALSE(x.get_dtype() == MJTNUM_DTYPE && x.get_nd() == 2 && x.get_flags() & bn::ndarray::C_CONTIGUOUS && x.shape(1) == state_size);
    FAIL_IF_FALSE(u.get_dtype() == MJTNUM_DTYPE && u.get_nd() == 2 && u.get_flags() & bn::ndarray::C_CONTIGUOUS && u.shape(1) == m_model->nu);
    FAIL_IF_FALSE(x.shape(0) == u.shape(0));

    int N = x.shape(0);
    int feat_size = _FeatSize();

    long xdims[2] = {N, state_size};
    long odims[2] = {N, feat_size};
    bn::ndarray xout = bn::empty(2, xdims, bn::dtype::get_builtin<mjtNum>());
    bn::ndarray fout = bn::empty(2, odims, bn::dtype::get_builtin<mjtNum>());

    mjtNum* xinptr = reinterpret_cast<mjtNum*>(x.get_data());
    mjtNum* uinptr = reinterpret_cast<mjtNum*>(u.get_data());
    mjtNum* xoutptr = reinterpret_cast<mjtNum*>(xout.get_data());
    mjtNum* foutptr = reinterpret_cast<mjtNum*>(fout.get_data());

    for (int n=0; n < N; ++n) {
        SetState(xinptr, m_model, m_data);
        xinptr += state_size;
        SetCtrl(uinptr, m_model, m_data);
        uinptr += m_model->nu;
        for (int i=0; i < m_numSteps; ++i) mj_step(m_model,m_option,m_data);
        if (m_featmask & feat_cfrc_ext) mj_rnePost(m_model, m_option, m_data);
        GetState(xoutptr, m_model, m_data);
        xoutptr += state_size;
        _CopyFeat(foutptr);
        foutptr += feat_size;
    }
    return bp::make_tuple(xout, fout);

}

void GetCOM(const mjModel* m, const mjData* d, mjtNum* com) {
    // see mj_com in engine_core.c
    mjtNum tot=0;
    com[0] = com[1] = com[2] = 0;
    for(int i=1; i<m->nbody; i++ ) {
        com[0] += d->xipos[3*i+0]*m->body_mass[i];
        com[1] += d->xipos[3*i+1]*m->body_mass[i];
        com[2] += d->xipos[3*i+2]*m->body_mass[i];
        tot += m->body_mass[i];
    }
    // compute com
    com[0] /= tot;
    com[1] /= tot;
    com[2] /= tot;
}

bn::ndarray PyMJCWorld2::GetCOMMulti(const bn::ndarray& x) {
    int state_size = StateSize(m_model);
    FAIL_IF_FALSE(x.get_dtype() == MJTNUM_DTYPE && x.get_nd() == 2 && x.get_flags() & bn::ndarray::C_CONTIGUOUS && x.shape(1) == state_size);
    int N = x.shape(0);
    long outdims[2] = {N,3};
    bn::ndarray out = bn::empty(2, outdims, bn::dtype::get_builtin<mjtNum>());
    mjtNum* ptr = (mjtNum*)out.get_data();
    for (int n=0; n < N; ++n) {
        SetState(reinterpret_cast<const mjtNum*>(x.get_data()), m_model, m_data);
        mj_kinematics(m_model, m_option, m_data);
        GetCOM(m_model, m_data, ptr);
        ptr += 3;
    }
    return out;
}

bn::ndarray PyMJCWorld2::GetJacSite(int site) {
    bn::ndarray out = bn::zeros(bp::make_tuple(3,m_model->nv), bn::dtype::get_builtin<mjtNum>());
    mjtNum* ptr = (mjtNum*)out.get_data();
    mj_jacSite(m_model, m_option, m_data, ptr, 0, site);
    return out;
}

void PyMJCWorld2::Kinematics() {
    mj_kinematics(m_model, m_option, m_data);
    mj_com(m_model, m_option, m_data);
    mj_tendon(m_model, m_option, m_data);
    mj_transmission(m_model, m_option, m_data);
}

void PyMJCWorld2::_PlotInit() {
    if (m_viewer == NULL) {
        m_viewer = new MujocoOSGViewer();
        m_viewer->SetModel(m_model);
    }
}

void PyMJCWorld2::Plot(const bn::ndarray& x) {
    FAIL_IF_FALSE(x.get_dtype() == MJTNUM_DTYPE && x.get_nd() == 1 && x.get_flags() & bn::ndarray::C_CONTIGUOUS);
    _PlotInit();
    SetState(reinterpret_cast<const mjtNum*>(x.get_data()),m_model,m_data);
	m_viewer->SetData(m_data);
	m_viewer->RenderOnce();
}

void PyMJCWorld2::Idle(const bn::ndarray& x) {
    FAIL_IF_FALSE(x.get_dtype() == MJTNUM_DTYPE && x.get_nd() == 1 && x.get_flags() & bn::ndarray::C_CONTIGUOUS);
    _PlotInit();
    SetState(reinterpret_cast<const mjtNum*>(x.get_data()),m_model,m_data);
    m_viewer->SetData(m_data);
    m_viewer->Idle();
}


int _ndarraysize(const bn::ndarray& arr) {
  int prod = 1;
  for (int i=0; i < arr.get_nd(); ++i) {
    prod *= arr.shape(i);
  }
  return prod;
}
template<typename T>
void _copyscalardata(const bp::object& from, T& to) {
  to = bp::extract<T>(from);
}
template <typename T>
void _copyarraydata(const bn::ndarray& from, T* to) {
  FAIL_IF_FALSE(from.get_dtype() == bn::dtype::get_builtin<T>() && from.get_flags() & bn::ndarray::C_CONTIGUOUS);
  memcpy(to, from.get_data(), _ndarraysize(from)*sizeof(T));
}
template<typename T>
void _csdihk(bp::dict d, const char* key, T& to) {
  // copy scalar data if has_key
  if (d.has_key(key)) _copyscalardata(d[key], to);
}
template<typename T>
void _cadihk(bp::dict d, const char* key, T* to) {
  // copy array data if has_key
  if (d.has_key(key)) {
    bn::ndarray arr = bp::extract<bn::ndarray>(d[key]);
    _copyarraydata<T>(arr, to);
  }
}

bn::ndarray _GetContacts(mjData* d) {
    bn::ndarray contacts = bn::zeros(bp::make_tuple(d->nc), bp::extract<bn::dtype>(main_namespace["contact_dtype"]));
    memcpy(contacts.get_data(), d->contact, d->nc*sizeof(mjContact));
    return contacts;    
}

bp::dict PyMJCWorld2::GetModel() {
    bp::dict out;
    #include "mjcpy_getmodel_autogen.i"
    return out;
}
void PyMJCWorld2::SetModel(bp::dict d) {
    #include "mjcpy_setmodel_autogen.i"
}
bp::dict PyMJCWorld2::GetOption() {
    bp::dict out;
    #include "mjcpy_getoption_autogen.i"
    return out;
}
void PyMJCWorld2::SetOption(bp::dict d) {
    #include "mjcpy_setoption_autogen.i"
}
bp::dict PyMJCWorld2::GetData() {
    bp::dict out;
    #include "mjcpy_getdata_autogen.i"
    
    out["contacts"] = _GetContacts(m_data);

    return out;
}
void PyMJCWorld2::SetData(bp::dict d) {
    #include "mjcpy_setdata_autogen.i"
}

bn::ndarray PyMJCWorld2::GetContacts(const bn::ndarray& x) {
    FAIL_IF_FALSE(x.get_dtype() == MJTNUM_DTYPE && x.get_nd() == 1 && x.get_flags() & bn::ndarray::C_CONTIGUOUS);
    SetState(reinterpret_cast<const mjtNum*>(x.get_data()), m_model, m_data);
    mj_solvePosition(m_model, m_option, m_data);
    return _GetContacts(m_data);
}


}


BOOST_PYTHON_MODULE(mjcpy) {
    bn::initialize();

    bp::enum_<mjtDisableBit>("disable_bit")
    .value("impulse", mjDSBL_IMPULSE)
    .value("friction", mjDSBL_JNTFRICTION)
    .value("limit", mjDSBL_LIMIT)
    .value("contact", mjDSBL_CONTACT)
    .value("constraint", mjDSBL_CONSTRAINT)
    .value("passive", mjDSBL_PASSIVE)
    .value("gravity", mjDSBL_GRAVITY)
    .value("bias", mjDSBL_BIAS)
    .value("anticipate", mjDSBL_ANTICIPATE)
    .value("clampvel", mjDSBL_CLAMPVEL)
    .value("warmstart", mjDSBL_WARMSTART)
    .value("perturb", mjDSBL_PERTURB)
    .value("energy", mjDSBL_ENERGY)
    .value("filterparent", mjDSBL_FILTERPARENT)
    .value("actuation", mjDSBL_ACTUATION)
    .value("cbctrl", mjDSBL_CBCTRL)
    .value("cbend", mjDSBL_CBEND)
    .value("clearctrl", mjDSBL_CLEARCTRL);

    bp::class_<PyMJCWorld2,boost::noncopyable>("MJCWorld","docstring here", bp::init<const std::string&>())

        .def("step",&PyMJCWorld2::Step)
        .def("step_multi",&PyMJCWorld2::StepMulti)
        // .def("StepMulti2",&PyMJCWorld::StepMulti2)
        // .def("StepJacobian", &PyMJCWorld::StepJacobian)
        // .def("Plot",&PyMJCWorld::Plot)
        // .def("SetActuatedDims",&PyMJCWorld::SetActuatedDims)
        // .def("ComputeContacts", &PyMJCWorld::ComputeContacts)
        // .def("SetTimestep",&PyMJCWorld::SetTimestep)
        // .def("SetContactType",&PyMJCWorld::SetContactType)
        .def("get_model",&PyMJCWorld2::GetModel)
        .def("set_model",&PyMJCWorld2::SetModel)
        .def("get_option",&PyMJCWorld2::GetOption)
        .def("set_option", &PyMJCWorld2::SetOption)
        .def("get_data",&PyMJCWorld2::GetData)
        .def("set_data",&PyMJCWorld2::SetData)
        .def("plot",&PyMJCWorld2::Plot)
        .def("idle",&PyMJCWorld2::Idle)
        .def("get_feat_desc",&PyMJCWorld2::GetFeatDesc)
        .def("get_COM_multi",&PyMJCWorld2::GetCOMMulti)
        .def("get_jac_site",&PyMJCWorld2::GetJacSite)
        .def("kinematics",&PyMJCWorld2::Kinematics)
        .def("get_contacts",&PyMJCWorld2::GetContacts)
        .def("set_feats",&PyMJCWorld2::SetFeats)
        // .def("SetModel",&PyMJCWorld::SetModel)
        // .def("GetImage",&PyMJCWorld::GetImage)
        .def("set_num_steps",&PyMJCWorld2::SetNumSteps)
        ;


    bp::object main = bp::import("__main__");
    main_namespace = main.attr("__dict__");    
    bp::exec(
        "import numpy as np\n"
        "contact_dtype = np.dtype([('dim','i'), ('geom1','i'), ('geom2','i'),('flc_address','i'),('compliance','f8'),('timeconst','f8'),('dist','f8'),('mindist','f8'),('pos','f8',3),('frame','f8',9),('friction','f8',5)])\n"
        , main_namespace
    );


}
