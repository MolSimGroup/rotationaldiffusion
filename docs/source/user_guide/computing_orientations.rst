Computing Orientations
======================

Why extract Orientations?
-------------------------
Rotational diffusion denotes the random *reorientation* of
an object in space. Hopefully, it is self-evident that we need to know
the *orientation* of the object before we can study its reorientation.
Therefore, we use the `MDAnalysis <https://www.mdanalysis.org>`_\ -based
:class:`Orientations <RotationalDiffusion.orientations.Orientations>`
class. By default, it first removes center-of-mass translation. Then,
for each trajectory frame, a rotation matrix is determined that
optimally aligns the frame to a reference structure by minimizing the
RMSD, that is, the root-mean-square deviation of atomic positions.


Usage
-----
First, we need to import the required modules.

>>> import RotationalDiffusion as rd
>>> import MDAnalysis as mda
>>> import numpy as np

Also, we need an exemplary trajectory that we load into an
MDAnalysis :class:`Universe <MDAnalysis.core.universe.Universe>` object.
Therefore, we use files from the `MDAnalysisTests` module, which contain
a solvated protein.

>>> from MDAnalysisTests.datafiles import TPR, GRO
>>> u = mda.Universe(TPR, GRO)

Now, as usual in MDAnalysis, we create an instance of the analysis class
and execute it by calling its
:meth:`run <RotationalDiffusion.orientations.Orientations.run>` method.
Thereby, we limit the analysis to the protein using standard
`MDAnalysis selection language
<https://userguide.mdanalysis.org/stable/selections.html>`_\ .

>>> ana = rd.Orientations(mobile=u, reference=u, select='protein').run()

The analysis yields a trajectory of orientations (i.e., rotational
matrices), which are stored in the ``results.orientations``
attribute of the
:class:`Orientations <RotationalDiffusion.orientations.Orientations>`
class instance. To conclude this example, we have a look at the
obtained orientations.

>>> np.shape(ana.results.orientations)
(1, 3, 3)
>>> np.allclose(ana.results.orientations, np.eye(3))
True

Evidently, we obtained a trajectory of orientations with a single frame,
which contains the identity matrix. Great! Also our input trajectory
contained a single frame, and since we used this trajectory for both
``mobile`` and ``reference``, we already knew that the optimal rotation
is no rotation. If the trajectories are longer, then the analysis loops
over the trajectory in ``mobile`` and fits every frame to the current,
fixed frame in ``reference``. Alternatively, if ``reference`` is not
explicitly specified, the current frame in ``mobile`` is used as the
reference.

Please make sure to checkout the API reference at
:class:`Orientations <RotationalDiffusion.orientations.Orientations>`\ ,
to see all supported options and features.


Performance Considerations
--------------------------
There are two ways to significantly speed up the calculations. First,
automatic unwrapping of the trajectory may be disabled by setting the
``unwrap`` argument to :any:`False` when instantiating the analysis
class. In that case, make sure that you provide a trajectory that is
already unwrapped, i.e., in which the molecular structure of interest is
not broken over the periodic boundary conditions! Second, the analysis
may be parallelized trivially over multiple CPU cores by specifying
a supported parallel ``backend`` (currently multiprocessing or dask)
and the number of workers (``n_workers``) in the call to the
:meth:`run <RotationalDiffusion.orientations.Orientations.run>`
method.


Alternative: Importing from GROMACS
-----------------------------------
The `GROMACS <https://www.gromacs.org>`_ molecular simulation suite also
provides a way for computing orientations. Support for that is *COMING
SOON*.