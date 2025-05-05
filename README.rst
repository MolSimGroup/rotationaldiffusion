.. |ss| raw:: html
   <strike>
.. |se| raw:: html
   </strike>

RotationalDiffusion
===================
|docs| |License| |mdanalysis|

A Python package for analyzing rotational diffusion from molecular
dynamics simulations.


Features
--------

- Determine the orientation of a molecule along an MD trajectory
- Compute rotational correlation functions
- Analyze time-dependent rotational diffusion behaviour
- Fit a Brownian rotational diffusion model to the rotational correlation functions
- Estimate uncertainties of the obtained principal axes and diffusion coefficients

Installation
------------
First, clone this repository using ::

    git clone https://github.com/MolSimGroup/rotationaldiffusion.git

Then, cd into the cloned directory and  install the package using pip ::

     pip install .

We suggest to use the package by importing it as ::

    import RotationalDiffusion as rd


Documentation and Tutorial
--------------------------
The full documentation is available at:
https://rotationaldiffusion.readthedocs.io

A comprehensive tutorial demonstrating the main functionalities of the
package is included in the documentation.

Author
------
This package was developed by Simon Holtbrügge. Contact:
simon.holtbruegge@rub.de.

Citation
--------
If you use this package in your research, please cite

    \S. Holtbrügge and L. Schäfer *(in preparation)*

Acknowledgement
---------------
This package extends prior work on rotational diffusion by Max Linke
*et al.*:

    M. Linke, J. Köfinger, and G. Hummer; **(2018)**,
    *J. Phys. Chem. B*, 122(21), 5630-5639.
    `DOI: 10.1021/acs.jpcb.7b11988 <https://doi.org/10.1021/acs.jpcb.7b11988>`_

License
-------
This project is licensed under the GNU General Public License v3.0 - see
the LICENSE file for details.

© Simon Holtbrügge, Lars Schäfer, 2024.

  .. |docs| image:: https://readthedocs.org/projects/rotationaldiffusion/badge/?version=latest
    :alt: Documentation Status
    :target: https://rotationaldiffusion.readthedocs.io

  .. |License| image:: https://img.shields.io/badge/License-GPLv3-blue.svg
    :alt: License: GPL v3
    :target: https://www.gnu.org/licenses/gpl-3.0

  .. |mdanalysis| image:: https://img.shields.io/badge/powered%20by-MDAnalysis-orange.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA
    :alt: Powered by MDAnalysis
    :target: https://www.mdanalysis.org
