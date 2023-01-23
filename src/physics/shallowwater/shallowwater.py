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
            euler_BC_type.SlipWall: euler_fcns.SlipWall,
            euler_BC_type.PressureOutlet: euler_fcns.PressureOutlet,
        })

    def set_physical_params(self, GravitationalAcceleration=9.81):
        '''
        This method sets physical parameters.

        Inputs:
        -------
            GravitationalAcceleration: free-fall gravitational acceleration
            SpecificHeatRatio: ratio of specific heats

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
            varq = 0.5 * g * h**2
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
    pass


class ShallowWater2D(ShallowWater):
    pass
