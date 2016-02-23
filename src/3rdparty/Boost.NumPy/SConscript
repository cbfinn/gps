# -*- python -*-

# Copyright Jim Bosch 2010-2012.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

# Big thanks to Mike Jarvis for help with the configuration prescriptions below.

import os
import sys
import subprocess
from SCons.SConf import CheckContext

def setupPaths(env, prefix, include, lib):
    if prefix is not None:
        if include is None:
            include = os.path.join(prefix, "include")
        if lib is None:
            lib = os.path.join(prefix, "lib")
    if include:
        env.PrependUnique(CPPPATH=[include])
    if lib:
        env.PrependUnique(LIBPATH=[lib])
AddMethod(Environment, setupPaths)

def checkLibs(context, try_libs, source_file):
    init_libs = context.env.get('LIBS', [])
    context.env.PrependUnique(LIBS=[try_libs])
    result = context.TryLink(source_file, '.cpp')
    if not result :
        context.env.Replace(LIBS=init_libs)
    return result
AddMethod(CheckContext, checkLibs)

def CheckPython(context):
    python_source_file = """
#include "Python.h"
int main()
{
  Py_Initialize();
  Py_Finalize();
  return 0;
}
"""
    context.Message('Checking if we can build against Python... ')
    try:
        import distutils.sysconfig
    except ImportError:
        context.Result(0)
        print 'Failed to import distutils.sysconfig.'
        return False
    context.env.AppendUnique(CPPPATH=[distutils.sysconfig.get_python_inc()])
    libDir = distutils.sysconfig.get_config_var("LIBDIR")
    context.env.AppendUnique(LIBPATH=[libDir])
    libfile = distutils.sysconfig.get_config_var("LIBRARY")
    import re
    match = re.search("(python.*)\.(a|so|dylib)", libfile)
    if match:
        context.env.AppendUnique(LIBS=[match.group(1)])
        if match.group(2) == 'a':
            flags = distutils.sysconfig.get_config_var('LINKFORSHARED')
            if flags is not None:
                context.env.AppendUnique(LINKFLAGS=flags.split())
    flags = [f for f in " ".join(distutils.sysconfig.get_config_vars("MODLIBS", "SHLIBS")).split()
             if f != "-L"]
    context.env.MergeFlags(" ".join(flags))
    result, output = context.TryRun(python_source_file,'.cpp')
    if not result and context.env["PLATFORM"] == 'darwin':
        # Sometimes we need some extra stuff on Mac OS
        frameworkDir = libDir       # search up the libDir tree for the proper home for frameworks
        while frameworkDir and frameworkDir != "/":
            frameworkDir, d2 = os.path.split(frameworkDir)
            if d2 == "Python.framework":
                if not "Python" in os.listdir(os.path.join(frameworkDir, d2)):
                    context.Result(0)
                    print (
                        "Expected to find Python in framework directory %s, but it isn't there"
                        % frameworkDir)
                    return False
                break
        context.env.AppendUnique(LINKFLAGS="-F%s" % frameworkDir)
        result, output = context.TryRun(python_source_file,'.cpp')
    if not result:
        context.Result(0)
        print "Cannot run program built with Python."
        return False
    if context.env["PLATFORM"] == "darwin":
        context.env["LDMODULESUFFIX"] = ".so"
    context.Result(1)
    return True

def CheckNumPy(context):
    numpy_source_file = """
#include "Python.h"
#include "numpy/arrayobject.h"
void doImport() {
  import_array();
}
int main()
{
  int result = 0;
  Py_Initialize();
  doImport();
  if (PyErr_Occurred()) {
    result = 1;
  } else {
    npy_intp dims = 2;
    PyObject * a = PyArray_SimpleNew(1, &dims, NPY_INT);
    if (!a) result = 1;
    Py_DECREF(a);
  }
  Py_Finalize();
  return result;
}
"""
    context.Message('Checking if we can build against NumPy... ')
    try:
        import numpy
    except ImportError:
        context.Result(0)
        print 'Failed to import numpy.'
        print 'Things to try:'
        print '1) Check that the command line python (with which you probably installed numpy):'
        print '   ',
        sys.stdout.flush()
        subprocess.call('which python',shell=True)
        print '  is the same as the one used by SCons:'
        print '  ',sys.executable
        print '   If not, then you probably need to reinstall numpy with %s.' % sys.executable
        print '   Alternatively, you can reinstall SCons with your preferred python.'
        print '2) Check that if you open a python session from the command line,'
        print '   import numpy is successful there.'
        return False
    context.env.Append(CPPPATH=numpy.get_include())
    result = context.checkLibs([''],numpy_source_file)
    if not result:
        context.Result(0)
        print "Cannot build against NumPy."
        return False
    result, output = context.TryRun(numpy_source_file,'.cpp')
    if not result:
        context.Result(0)
        print "Cannot run program built with NumPy."
        return False
    context.Result(1)
    return True

def CheckBoostPython(context):
    bp_source_file = """
#include "boost/python.hpp"
class Foo { public: Foo() {} };
int main()
{
  Py_Initialize();
  boost::python::object obj;
  boost::python::class_< Foo >("Foo", boost::python::init<>());
  Py_Finalize();
  return 0;
}
"""
    context.Message('Checking if we can build against Boost.Python... ')
    context.env.setupPaths(
        prefix = GetOption("boost_prefix"),
        include = GetOption("boost_include"),
        lib = GetOption("boost_lib")
        )
    result = (
        context.checkLibs([''], bp_source_file) or
        context.checkLibs(['boost_python'], bp_source_file) or
        context.checkLibs(['boost_python-mt'], bp_source_file)
        )
    if not result:
        context.Result(0)
        print "Cannot build against Boost.Python."
        return False
    result, output = context.TryRun(bp_source_file, '.cpp')
    if not result:
        context.Result(0)
        print "Cannot run program built against Boost.Python."
        return False
    context.Result(1)
    return True

# Setup command-line options
def setupOptions():
    AddOption("--prefix", dest="prefix", type="string", nargs=1, action="store",
              metavar="DIR", default="/usr/local", help="installation prefix")
    AddOption("--install-headers", dest="install_headers", type="string", nargs=1, action="store",
              metavar="DIR", help="location to install header files (overrides --prefix for headers)")
    AddOption("--install-lib", dest="install_lib", type="string", nargs=1, action="store",
              metavar="DIR", help="location to install libraries (overrides --prefix for libraries)")
    AddOption("--with-boost", dest="boost_prefix", type="string", nargs=1, action="store",
              metavar="DIR", default=os.environ.get("BOOST_DIR"),
              help="prefix for Boost libraries; should have 'include' and 'lib' subdirectories")
    AddOption("--with-boost-include", dest="boost_include", type="string", nargs=1, action="store",
              metavar="DIR", help="location of Boost header files")
    AddOption("--with-boost-lib", dest="boost_lib", type="string", nargs=1, action="store",
              metavar="DIR", help="location of Boost libraries")
    AddOption("--rpath", dest="custom_rpath", type="string", action="append",
              help="runtime link paths to add to libraries and executables; may be passed more than once")
    variables = Variables()
    variables.Add("CCFLAGS", default=os.environ.get("CCFLAGS", "-O2 -g"), help="compiler flags")
    return variables

def makeEnvironment(variables):
    shellEnv = {}
    for key in ("PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PYTHONPATH"):
        if key in os.environ:
            shellEnv[key] = os.environ[key]
    env = Environment(variables=variables, ENV=shellEnv)
    if os.environ.has_key("CCFLAGS"):
        env.AppendUnique(CCFLAGS = os.environ["CCFLAGS"])
    custom_rpath = GetOption("custom_rpath")
    if custom_rpath is not None:
        env.AppendUnique(RPATH=custom_rpath)
    boost_lib = GetOption ('boost_lib')
    if boost_lib is not None:
        env.PrependUnique(LIBPATH=boost_lib)
    return env

def setupTargets(env, root="."):
    lib = SConscript(os.path.join(root, "libs", "numpy", "src", "SConscript"), exports='env')
    example = SConscript(os.path.join(root, "libs", "numpy", "example", "SConscript"), exports='env')
    test = SConscript(os.path.join(root, "libs", "numpy", "test", "SConscript"), exports='env')
    prefix = Dir(GetOption("prefix")).abspath
    install_headers = GetOption('install_headers')
    install_lib = GetOption('install_lib')
    if not install_headers:
        install_headers = os.path.join(prefix, "include")
    if not install_lib:
        install_lib = os.path.join(prefix, "lib")
    env.Alias("install", env.Install(install_lib, lib))
    for header in ("dtype.hpp", "invoke_matching.hpp", "matrix.hpp", 
                   "ndarray.hpp", "numpy_object_mgr_traits.hpp",
                   "scalars.hpp", "ufunc.hpp",):
        env.Alias("install", env.Install(os.path.join(install_headers, "boost", "numpy"),
                                           os.path.join(root, "boost", "numpy", header)))
    env.Alias("install", env.Install(os.path.join(install_headers, "boost"),
                                     os.path.join(root, "boost", "numpy.hpp")))

checks = {"CheckPython": CheckPython, "CheckNumPy": CheckNumPy, "CheckBoostPython": CheckBoostPython}

Return("setupOptions", "makeEnvironment", "setupTargets", "checks")
