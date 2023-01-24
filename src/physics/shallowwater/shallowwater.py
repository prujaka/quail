# ------------------------------------------------------------------------ #
#
#       quail: A lightweight discontinuous Galerkin code for
#              teaching and prototyping
#       <https://github.com/IhmeGroup/quail>
#
#       Copyright (C) 2020-2021
#
#       This program is distributed under the terms of the GNU
#       General Public License v3.0. You should have received a copy
#       of the GNU General Public License along with this program.
#       If not, see <https://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------ #

# ------------------------------------------------------------------------ #
#
#       File : src/physics/shallowwater/shallowwater.py
#
#       Contains class definitions for 1D and 2D hyperbolic
#       shallow water equations.
#
#       By Sergey Tkachenko (https://github.com/prujaka).
#
# ------------------------------------------------------------------------ #

from enum import Enum
import numpy as np

import errors
import general

import physics.base.base as base
import physics.base.functions as base_fcns
from physics.base.functions import BCType as base_BC_type
from physics.base.functions import ConvNumFluxType as base_conv_num_flux_type
from physics.base.functions import FcnType as base_fcn_type

import physics.shallowwater.functions as shallowwater_fcns
from physics.shallowwater.functions import BCType as shallowwater_BC_type
from physics.shallowwater.functions import ConvNumFluxType as \
    shallowwater_conv_num_flux_type
from physics.shallowwater.functions import FcnType as shallowwater_fcn_type
from physics.shallowwater.functions import SourceType as \
    shallowwater_source_type


# TODO: define all the mentioned functions for SW
# TODO: add g parameter to Shallowwater(base.PhysicsBase) like in Euler gamma
# TODO: pay attention to get_conv_flux_interior, it uses stuff from
#  get_conv_flux_projected in Lax-Friedrichs flux in functions.py

class ShallowWater(base.PhysicsBase):
    '''
    This class corresponds to the classic shallow water equations.
    It inherits attributes and methods from the PhysicsBase class.
    See PhysicsBase for detailed comments of attributes and methods.
    This class should not be instantiated directly. Instead,
    the 1D and 2D variants, which inherit from this class (see below),
    should be instantiated.

    Additional methods and attributes are commented below.

    Attributes:
    -----------
    g: float
        gravitational acceleration
    '''
    PHYSICS_TYPE = general.PhysicsType.ShallowWater

    def __init__(self):
        super().__init__()
        self.g = 0.

    def set_maps(self):
        super().set_maps()

        self.BC_map.update({
            base_BC_type.StateAll: base_fcns.StateAll,
            base_BC_type.Extrapolate: base_fcns.Extrapolate,
            shallowwater_BC_type.SlipWall: shallowwater_fcns.SlipWall,
            shallowwater_BC_type.PressureOutlet:
                shallowwater_fcns.PressureOutlet,
        })

    def set_physical_params(self, GravitationalAcceleration=9.81):
        '''
        This method sets physical parameters.

        Inputs:
        -------
            GravitationalAcceleration: free-fall gravitational acceleration

        Outputs:
        --------
            self: physical parameters set
        '''
        self.g = GravitationalAcceleration

    class AdditionalVariables(Enum):
        Pressure = "p"
        # InternalEnergy = "\\rho e"
        SoundSpeed = "c"
        MaxWaveSpeed = "\\lambda"
        Velocity = "|u|"
        XVelocity = "u"
        YVelocity = "v"

    def compute_additional_variable(self, var_name, Uq, flag_non_physical):
        ''' Extract state variables '''
        sh = self.get_state_slice("Depth")
        # shE = self.get_state_slice("Energy")
        smom = self.get_momentum_slice()
        h = Uq[:, :, sh]
        # hE = Uq[:, :, shE]
        mom = Uq[:, :, smom]

        ''' Unpack '''
        g = self.g

        ''' Flag non-physical state '''
        if flag_non_physical:
            if np.any(h < 0.):
                raise errors.NotPhysicalError

        ''' Nested functions for common quantities '''

        def get_pressure():
            varq = 0.5 * g * h ** 2
            if flag_non_physical:
                if np.any(varq < 0.):
                    raise errors.NotPhysicalError
            return varq

        ''' Compute '''
        vname = self.AdditionalVariables[var_name].name

        if vname is self.AdditionalVariables["Pressure"].name:
            varq = get_pressure()
        elif vname is self.AdditionalVariables["SoundSpeed"].name:
            varq = np.sqrt(g * h)
        elif vname is self.AdditionalVariables["MaxWaveSpeed"].name:
            # |u| + c
            varq = np.linalg.norm(mom, axis=2, keepdims=True) / h + np.sqrt(
                g * h)
        elif vname is self.AdditionalVariables["Velocity"].name:
            varq = np.linalg.norm(mom, axis=2, keepdims=True) / h
        elif vname is self.AdditionalVariables["XVelocity"].name:
            varq = mom[:, :, [0]] / h
        elif vname is self.AdditionalVariables["YVelocity"].name:
            varq = mom[:, :, [1]] / h
        else:
            raise NotImplementedError

        return varq

    def compute_pressure_gradient(self, Uq, grad_Uq):
        '''
        Compute the gradient of pressure with respect to physical space. This is
        needed for pressure-based shock sensors.

        Inputs:
        -------
            Uq: solution in each element evaluated at quadrature points
            [ne, nq, ns]
            grad_Uq: gradient of solution in each element evaluted at quadrature
                points [ne, nq, ns, ndims]

        Outputs:
        --------
            array: gradient of pressure with respected to physical space
                [ne, nq, ndims]
        '''
        sh = self.get_state_slice("Depth")
        smom = self.get_momentum_slice()
        h = Uq[:, :, sh]
        mom = Uq[:, :, smom]
        g = self.g

        # Compute dp/dU
        dpdU = np.empty_like(Uq)
        dpdU[:, :, sh] = g * h
        dpdU[:, :, smom] = 0. * mom

        # Multiply with dU/dx
        return np.einsum('ijk, ijkl -> ijl', dpdU, grad_Uq)


class ShallowWater1D(ShallowWater):
    '''
    This class corresponds to classic 1D shallow water equations.
    It inherits attributes and methods from the ShallowWater class.
    See ShallowWater for detailed comments of attributes and methods.

    Additional methods and attributes are commented below.
    '''
    NUM_STATE_VARS = 2
    NDIMS = 1

    def set_maps(self):
        super().set_maps()

        d = {
            shallowwater_fcn_type.SmoothIsentropicFlow:
                shallowwater_fcns.SmoothIsentropicFlow,
            shallowwater_fcn_type.MovingShock: shallowwater_fcns.MovingShock,
            shallowwater_fcn_type.DepthWave: shallowwater_fcns.DepthWave,
            shallowwater_fcn_type.RiemannProblem:
                shallowwater_fcns.RiemannProblem,
            shallowwater_fcn_type.ShuOsherProblem:
                shallowwater_fcns.ShuOsherProblem,
        }

        self.IC_fcn_map.update(d)
        self.exact_fcn_map.update(d)
        self.BC_fcn_map.update(d)

        self.source_map.update({
            shallowwater_source_type.StiffFriction:
                shallowwater_fcns.StiffFriction,
        })

        self.conv_num_flux_map.update({
            base_conv_num_flux_type.LaxFriedrichs:
                shallowwater_fcns.LaxFriedrichs1D,
            shallowwater_conv_num_flux_type.Roe: shallowwater_fcns.Roe1D,
        })

    class StateVariables(Enum):
        Depth = "\\h"
        XMomentum = "\\h u"

    def get_state_indices(self):
        ih = self.get_state_index("Depth")
        ihu = self.get_state_index("XMomentum")

        return ih, ihu

    def get_state_slices(self):
        sh = self.get_state_slice("Depth")
        shu = self.get_state_slice("XMomentum")

        return sh, shu

    def get_momentum_slice(self):
        ihu = self.get_state_index("XMomentum")
        smom = slice(ihu, ihu + 1)

        return smom

    def get_conv_flux_interior(self, Uq):
        # Get indices of state variables
        ih, ihu = self.get_state_indices()

        h = Uq[:, :, ih]  # [n, nq]
        hu = Uq[:, :, ihu]  # [n, nq]

        # Get velocity
        u = hu / h
        # Get squared velocity
        u2 = u ** 2

        # Calculate pressure using the Ideal Gas Law
        p = 0.5 * self.g * h ** 2  # [n, nq]

        # Assemble flux matrix
        F = np.empty(Uq.shape + (self.NDIMS,))  # [n, nq, ns, ndims]
        F[:, :, ih, 0] = hu  # Flux of mass
        F[:, :, ihu, 0] = h * u2 + p  # Flux of momentum

        return F, (u2, h, p)
