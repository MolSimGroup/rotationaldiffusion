.. |ss| raw:: html
   <strike>
.. |se| raw:: html
   </strike>

README
======

This package was written to extract the principal axes and the rotational
diffusion tensor of a molecule (typically a protein in solution) from
Molecular Dynamics Simulations.

The method is based on the theoretical description of Brownian rotational
dynamics of a rigid body developed by Favro (1960). The method has been
implemented before (pydiffusion by Max Linke). My implementation adds the ability to compute what I call the "instantaneous"
diffusion tensor and principal axis frame, as well as the ability to fit a
semi-isotropic or isotropic model.

I am currently rewriting the code, mainly to include extensive documentation
with sphinx and testing with pytest, aiming towards code that is more
user-friendly, readable, and robust. Check-out rotationaldiffusion/mdaapi.py
to get a flavour of the new code.

Planned features
================

*May or may not be implemented*

- [X] use git for version control
- [X] align using MDAnalysis
- [ ] better testing (work in progress)
- [ ] continuous integration tests
- [ ] add proper logging, warnings, and errors
- [ ] make some functions private
- [ ] implement consistent variable naming
- [ ] command-line interface
- [ ] tutorial
- [ ] better documentation (work in progress)
- [X] automated documentation
- [ ] online documentation
- [ ] add doctests
- [ ] custom sphinx template
- [X] use codecov to check how much code is tested => Pycharm IDE uses coverage
- [ ] optimize project tree for publishing
- [ ] publish to PyPI
- [ ] publish to conda-forge (or similar)
- [̶ ]̶ ̶a̶d̶d̶ ̶p̶a̶r̶a̶l̶l̶e̶l̶i̶z̶a̶t̶i̶o̶n̶
- [ ] add acknowledgement / funding (RESOLV)


Author
======
This package was written by Simon Holtbruegge. Feel free to contact me in case
you experience any issues or need assistance with using the code
([simon.holtbruegge@rub.de](mailto:simon.holtbruegge@rub.de)).
