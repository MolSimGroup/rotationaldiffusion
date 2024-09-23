.. |ss| raw:: html
   <strike>
.. |se| raw:: html
   </strike>

Purpose
=======

Extract the principal axes and the rotational diffusion tensor from the
orientations of a rigid body, typically a protein in solution.

The orientations are extracted from Molecular Dynamics trajectories.
Alternatively, the orientations can be passed directly as rotational
matrices or quaternions.

Installation instructions
=========================
First, clone this repository using ::

    git clone https://github.com/MolSimGroup/rotationaldiffusion.git

Then, cd into the cloned directory and  install the package using pip ::

     pip install .

It is suggested to use the package by importing it as ::

    import rotationaldiffusion as rd

Basic usage
============
**Note:** a tutorial and more extensive documentation may follow soon.

For analyzing a single trajectory, run ::

    import rotationaldiffusion as rd
    import MDAnalysis as mda

    # Load the trajectory into a MDAnalysis universe.
    u = mda.Universe(TOPOLOGY, TRAJECTORY)

    # Compute the orientations.
    # Using 'name CA or name N or name C' typically leads to a similar
    # selection as GROMACS 'backbone' selection.
    orientation_analysis = rd.orientations.Orientations(u, select=SELECTION_STRING)
    orientation_analysis.run()
    orientations = orientation_analysis.results.orientations

    # Alternatively, the orientations can be loaded from GROMACS
    # 'gmx rotmat' output files.
    orientations, time = rd.load_orientations(XVG-FILE)

    # Convert to quaternions.
    quats = rd.quaternions.rotmat2quat(orientations)

    # Compute rotational correlation matrix Q.
    correlations = rd.extract_Q_data(quats)

    # Get array of correlation times.
    time = rd.arange_lag_times(correlations, TIME_STEP)

    # Compute time-dependent diffusion coefficients and principal axes.
    # Note: visualize these functions to decide on a model
    # (anisotropic, semi-isotropic, isotropic) and a fit window.
    D, PAF = rd.instantaneous_tensors(time, correlations)

    # Fit.
    fit = rd.least_squares_fit(time, correlations, model=MODEL_STRING)
    D = fit.D
    PAF = fit.rotation_axes

Notes
=====
- The API is still unstable and may change between versions.
- Docstrings are available, the documentation can be generated using
sphinx by running ``make html`` in the *docs* subdirectory.
- A short user guide and/or tutorial are planned, as is a publication
on this topic.
- The method is based on the theoretical description of Brownian rotational
dynamics of a rigid body developed by Favro (1960). The method has been
implemented before (pydiffusion by Max Linke).

Planned features
================

*May or may not be implemented, let's see ;)*

- [X] use git for version control
- [X] align using MDAnalysis
- [ ] better testing (work in progress)
- [ ] continuous integration tests
- [ ] add proper logging, warnings, and errors
- [ ] command-line interface
- [ ] tutorial
- [ ] better documentation (work in progress)
- [X] automated documentation
- [ ] online documentation
- [ ] add doctests
- [X] use codecov to check how much code is tested => Pycharm IDE uses coverage
- [ ] publish to PyPI
- [ ] publish to conda-forge (or similar)
- [ ] add parallelization (e.g. using dask)
- [ ] add acknowledgement / funding (RESOLV)

Author
======
This package was written by Simon Holtbruegge. Feel free to contact me in case
you experience any issues or need assistance with using the code
([simon.holtbruegge@rub.de](mailto:simon.holtbruegge@rub.de)).

© Simon Holtbrügge, Lars Schäfer, 2024.
