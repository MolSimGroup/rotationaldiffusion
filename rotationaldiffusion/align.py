import logging
import numpy as np

import MDAnalysis as mda
import MDAnalysis.analysis.rms as rms
from MDAnalysis.lib.util import get_weights
from MDAnalysis.lib.log import ProgressBar

from MDAnalysis.analysis.align import AverageStructure

logger = logging.getLogger('MDAnalysis.analysis.align')


def iterative_average(
    mobile, reference=None, select='all', weights=None, niter=100,
    eps=1e-6, verbose=False, **kwargs
):
    """Iteratively calculate an optimal reference that is also the average
    structure after an RMSD alignment.

    The optimal reference is defined as average
    structure of a trajectory, with the optimal reference used as input.
    This function computes the optimal reference by using a starting
    reference for the average structure, which is used as the reference
    to calculate the average structure again. This is repeated until the
    reference structure has converged. :footcite:p:`Linke2018`

    Parameters
    ----------
    mobile : mda.Universe
        Universe containing trajectory to be fitted to reference.
    reference : mda.Universe (optional)
        Universe containing the initial reference structure.
    select : str or tuple or dict (optional)
        Atom selection for fitting a substructue. Default is set to all.
        Can be tuple or dict to define different selection strings for
        mobile and target.
    weights : str, array_like (optional)
        Weights that can be used. If `None` use equal weights, if `'mass'`
        use masses of ref as weights or give an array of arbitrary weights.
    niter : int (optional)
        Maximum number of iterations.
    eps : float (optional)
        RMSD distance at which reference and average are assumed to be
        equal.
    verbose : bool (optional)
        Verbosity.
    **kwargs : dict (optional)
        AverageStructure kwargs.

    Returns
    -------
    avg_struc : AverageStructure
        AverageStructure result from the last iteration.

    Example
    -------
    `iterative_average` can be used to obtain a :class:`MDAnalysis.Universe`
    with the optimal reference structure.

    ::

        import MDAnalysis as mda
        from MDAnalysis.analysis import align
        from MDAnalysisTests.datafiles import PSF, DCD

        u = mda.Universe(PSF, DCD)
        av = align.iterative_average(u, u, verbose=True)

        averaged_universe = av.results.universe

    References
    ----------

    .. footbibliography::

    .. versionadded:: 2.8.0
    """
    if not reference:
        reference = mobile

    select = rms.process_selection(select)
    ref = mda.Merge(reference.select_atoms(*select['reference']))
    sel_mobile = select['mobile'][0]

    weights = get_weights(ref.atoms, weights)

    drmsd = np.inf
    for i in ProgressBar(range(niter)):
        # found a converged structure
        if drmsd < eps:
            break

        avg_struc = AverageStructure(
            mobile, reference=ref, select={
                'mobile': sel_mobile, 'reference': 'all'
                },
            weights=weights, **kwargs
        ).run()
        drmsd = rms.rmsd(ref.atoms.positions, avg_struc.results.positions,
                         weights=weights)
        ref = avg_struc.results.universe

        if verbose:
            logger.debug(
                f"iterative_average(): i = {i}, "
                f"rmsd-change = {drmsd:.5f}, "
                f"ave-rmsd = {avg_struc.results.rmsd:.5f}"
            )

    else:
        logger.warning(
            "iterative_average(): Did NOT converge in "
            f"{niter} iterations to DRMSD < {eps}. "
            f"Last DRMSD = {drmsd:.5f}."
            f"Final average RMSD = {avg_struc.results.rmsd:.5f}"
        )

        return avg_struc

    logger.info(
        f"iterative_average(): Converged to DRMSD < {eps}. "
        f"Final average RMSD = {avg_struc.results.rmsd:.5f}"
    )

    return avg_struc