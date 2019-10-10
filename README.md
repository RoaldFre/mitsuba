Mitsuba — Forward Scattering Dipole Implementation
==================================================

**NOTE: If you stumbled upon this repo looking for an implementation of 
Adaptive Lightslice for Virtual Ray Lights, checkout the ALVRL branch!**


## About the Forward Scattering Dipole

This repository implements the Forward Scattering Dipole model within the Mitsuba renderer. More information about the model can be found in the SIGGRAPH 2017 paper 'A Forward Scattering Dipole Model from a Functional Integral Approximation'

Project page: http://graphics.cs.kuleuven.be/publications/FD17FSDM/

Note: the importance sampling routines in this repository have been improved considerably compared to the version that was published in the original paper.


## Dependencies

All the usual dependencies of Mitsuba with an additional dependency on GSL (the Gnu Scientific Library). See the [official Mitsuba documentation](http://mitsuba-renderer.org/docs.html) for more information.


## How to Build

Get the repository:

    $ git clone https://github.com/roaldfre/mitsuba-ALVRL-fwddip

The forward scattering dipole code lives in the fwddip branch.

    $ git checkout fwddip

Select a build configuration (currently, only the gcc profile for double 
precision is fully supported)
    
    $ cp build/config-linux-gcc-double.py config.py

Alternatively, if you want spectral rendering (e.g. to render a realistic skin 
material), you can choose a spectral build profile:

    $ cp build/config-linux-gcc-spectral-double.py config.py

Once you have chosen a build profile, start the build process (e.g. assuming 
four cores)

    $ scons -j 4

Add the proper directories to the `$PATH` (e.g. assuming sh/bash shell)

    $ . setpath.sh

You should now be able to run the 'mitsuba' and 'mtsgui' commands (and several 
others).


## Using the Forward Scattering Dipole Model

You can find documented example scenes on the project page: 
http://graphics.cs.kuleuven.be/publications/FD17FSDM/

