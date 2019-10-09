Mitsuba — Adaptive Lightslice for Virtual Ray Lights (ALVRL) Branch
===================================================================

**NOTE: If you stumbled upon this repo looking for an implementation of the 
Forward Scattering Dipole Model, checkout the fwddip branch!**

## About

This ALVRL branch holds the Mitsuba code for Adaptive Lightslice for 
Virtual Ray Lights, as introduced in the paper:

> Frederickx Roald, Pieterjan Bartels, and Philip Dutré.  
> "Adaptive LightSlice for Virtual Ray Lights."  
> EUROGRAPHICS 2015-Short Papers (2015): 61-64.  

NOTE1: This is (CPU-only) 'research code', so it isn't very polished or 
optimized ;-).

NOTE2: The VRL integration sampling is not phase-function aware. 
Non-isotropic phase functions will need higher 'volVolSamples' and 
'volSurfSamples' (and are not fully tested).

NOTE3: The VRL renderer only handles the multiple scattering contribution 
and needs to be augmented with direct transmission and single scattering to 
obtain the full light transport.


## Documentation

See the Adaptive Lightslice paper for more information on the algorithm.
For compilation, usage, and a full plugin reference of Mitsuba, please see 
the [official Mitsuba documentation](http://mitsuba-renderer.org/docs.html).


## Example scenes

Example scenes are available on the
[project website](http://graphics.cs.kuleuven.be/publications/FBD15ALVRL/).

