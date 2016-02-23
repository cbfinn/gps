# -*- python -*-

# Copyright Jim Bosch 2010-2012.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

setupOptions, makeEnvironment, setupTargets, checks = SConscript("SConscript")

variables = setupOptions()

env = makeEnvironment(variables)
env.AppendUnique(CPPPATH="#.")

if not GetOption("help") and not GetOption("clean"):
    config = env.Configure(custom_tests=checks)
    if not (config.CheckPython() and config.CheckNumPy() and config.CheckBoostPython()):
       Exit(1)
    env = config.Finish()

setupTargets(env)
