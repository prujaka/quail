# ------------------------------------------------------------------------ #
#
#       quail: A lightweight discontinuous Galerkin code for
#              teaching and prototyping
#		<https://github.com/IhmeGroup/quail>
#
#		Copyright (C) 2020-2021
#
#       This program is distributed under the terms of the GNU
#		General Public License v3.0. You should have received a copy
#       of the GNU General Public License along with this program.
#		If not, see <https://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------ #

# ------------------------------------------------------------------------ #
#
#       File : src/physics/shalowwater/functions.py
#
#       Contains definitions of Functions, boundary conditions, and source
#       terms for the shallow water equations.
#
#       By Sergey Tkachenko (https://github.com/prujaka)
#
# ------------------------------------------------------------------------ #

from enum import Enum, auto
import numpy as np
from scipy.optimize import fsolve, root

import errors
import general

from physics.base.data import (FcnBase, BCWeakRiemann, BCWeakPrescribed,
                               SourceBase, ConvNumFluxBase)


class FcnType(Enum):
    pass


class BCType(Enum):
    pass


class SourceType(Enum):
    pass


'''
---------------
State functions
---------------
These classes inherit from the FcnBase class. See FcnBase for detailed
comments of attributes and methods. Information specific to the
corresponding child classes can be found below. These classes should
correspond to the FcnType enum members above.
'''


class SteadyState(FcnBase):  # test case
    pass


'''
-------------------
Boundary conditions
-------------------
These classes inherit from either the BCWeakRiemann or BCWeakPrescribed
classes. See those parent classes for detailed comments of attributes
and methods. Information specific to the corresponding child classes can be
found below. These classes should correspond to the BCType enum members
above.
'''


class SlipWall(BCWeakPrescribed):
    pass


'''
------------------------
Numerical flux functions
------------------------
These classes inherit from the ConvNumFluxBase or DiffNumFluxBase class. 
See ConvNumFluxBase/DiffNumFluxBase for detailed comments of attributes 
and methods. Information specific to the corresponding child classes can 
be found below. These classes should correspond to the ConvNumFluxType 
or DiffNumFluxType enum members above.
'''


class LaxFriedrichs1D(ConvNumFluxBase):
    pass


class LaxFriedrichs2D(ConvNumFluxBase):
    pass