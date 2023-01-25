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
#       File : src/physics/shallowwater/functions.py
#
#       Contains definitions of Functions, boundary conditions, and source
#       terms for the shallow water equations.
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
    '''
    Enum class that stores the types of analytical functions for initial
    conditions, exact solutions, and/or boundary conditions. These
    functions are specific to the available shallow water equation sets.
    '''
    DepthWave = auto()
    SteadyState = auto()


class BCType(Enum):
    '''
    Enum class that stores the types of boundary conditions. These boundary
    conditions are specific to the available shallow water equation sets.
    '''
    SlipWall = auto()


class SourceType(Enum):
    '''
    Enum class that stores the types of source terms. These
    source terms are specific to the available shallow water equation sets.
    '''
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


class SteadyState(FcnBase):
    '''
    Simple smooth density wave.

    Attributes:
    -----------


    '''

    def __init__(self):
        '''
        This method initializes the attributes.

        Inputs:
        -------


        Outputs:
        --------
            self: attributes initialized
        '''
        pass

    def get_state(self, physics, x, t):
        sh, shu = physics.get_state_slices()

        Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

        h = 1.0 * x
        hu = 0. * h

        Uq[:, :, sh] = h
        Uq[:, :, shu] = hu

        return Uq  # [ne, nq, ns]


class DepthWave(FcnBase):
    '''
    Simple smooth density wave.

    Attributes:
    -----------


    '''

    def __init__(self):
        '''
        This method initializes the attributes.

        Inputs:
        -------


        Outputs:
        --------
            self: attributes initialized
        '''
        pass

    def get_state(self, physics, x, t):
        sh, shu = physics.get_state_slices()

        Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])

        h = 1.0 + 0.1 * np.sin(2. * np.pi * x)
        hu = h * 1.0

        Uq[:, :, sh] = h
        Uq[:, :, shu] = hu

        return Uq  # [ne, nq, ns]


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
    '''
    This class corresponds to a slip wall. See documentation for more
    details.
    '''

    def get_boundary_state(self, physics, UqI, normals, x, t):
        smom = physics.get_momentum_slice()

        # Unit normals
        n_hat = normals / np.linalg.norm(normals, axis=2, keepdims=True)

        # Remove momentum contribution in normal direction from boundary
        # state
        hveln = np.sum(UqI[:, :, smom] * n_hat, axis=2, keepdims=True)
        UqB = UqI.copy()
        UqB[:, :, smom] -= hveln * n_hat

        return UqB


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
    '''
    This class corresponds to the local Lax-Friedrichs flux function for the
    ShallowWater1D class. This replaces the generalized, less efficient version
    of the Lax-Friedrichs flux found in base.
    '''

    def compute_flux(self, physics, UqL, UqR, normals):
        # Normalize the normal vectors
        n_mag = np.linalg.norm(normals, axis=2, keepdims=True)
        n_hat = normals / n_mag

        # Left flux
        FqL, (u2L, hL, pL) = physics.get_conv_flux_projected(UqL, n_hat)

        # Right flux
        FqR, (u2R, hR, pR) = physics.get_conv_flux_projected(UqR, n_hat)

        # Jump
        dUq = UqR - UqL

        # Max wave speeds at each point
        aL = np.empty(pL.shape + (1,))
        aR = np.empty(pR.shape + (1,))
        aL[:, :, 0] = np.sqrt(u2L) + np.sqrt(physics.g * hL)
        aR[:, :, 0] = np.sqrt(u2R) + np.sqrt(physics.g * hR)
        idx = aR > aL
        aL[idx] = aR[idx]

        # Put together
        return 0.5 * n_mag * (FqL + FqR - aL * dUq)


class LaxFriedrichs2D(ConvNumFluxBase):
    '''
    This class corresponds to the local Lax-Friedrichs flux function for the
    ShallowWater2D class. This replaces the generalized, less efficient version
    of the Lax-Friedrichs flux found in base.
    '''

    def compute_flux(self, physics, UqL, UqR, normals):
        # Normalize the normal vectors
        n_mag = np.linalg.norm(normals, axis=2, keepdims=True)
        n_hat = normals / n_mag

        # Left flux
        FqL, (u2L, v2L, hL, pL) = physics.get_conv_flux_projected(UqL, n_hat)

        # Right flux
        FqR, (u2R, v2R, hR, pR) = physics.get_conv_flux_projected(UqR, n_hat)

        # Jump
        dUq = UqR - UqL

        # Max wave speeds at each point
        aL = np.empty(pL.shape + (1,))
        aR = np.empty(pR.shape + (1,))
        aL[:, :, 0] = np.sqrt(u2L + v2L) + np.sqrt(physics.g * hL)
        aR[:, :, 0] = np.sqrt(u2R + v2R) + np.sqrt(physics.g * hL)
        idx = aR > aL
        aL[idx] = aR[idx]

        # Put together
        return 0.5 * n_mag * (FqL + FqR - aL * dUq)
