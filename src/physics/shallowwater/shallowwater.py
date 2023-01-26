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
            # shallowwater_BC_type.PressureOutlet:
            #     shallowwater_fcns.PressureOutlet,
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
        # The name "Density" is chosen for compatibility with the
        # PositivityPreserving(base.LimiterBase) class attribute names
        sh = self.get_state_slice("Density")
        smom = self.get_momentum_slice()
        h = Uq[:, :, sh]
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
        sh = self.get_state_slice("Density")
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
            shallowwater_fcn_type.DepthWave: shallowwater_fcns.DepthWave,
            shallowwater_fcn_type.SteadyState: shallowwater_fcns.SteadyState,
        }

        self.IC_fcn_map.update(d)
        self.exact_fcn_map.update(d)
        self.BC_fcn_map.update(d)

        # self.source_map.update({
        #     shallowwater_source_type.StiffFriction:
        #         shallowwater_fcns.StiffFriction,
        # })

        self.conv_num_flux_map.update({
            base_conv_num_flux_type.LaxFriedrichs:
                shallowwater_fcns.LaxFriedrichs1D,
        })

    class StateVariables(Enum):
        Density = "h"
        XMomentum = "h u"

    def get_state_indices(self):
        ih = self.get_state_index("Density")
        ihu = self.get_state_index("XMomentum")

        return ih, ihu

    def get_state_slices(self):
        sh = self.get_state_slice("Density")
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


class ShallowWater2D(ShallowWater):
    '''
    This class corresponds to 2D shallow water equations.
    It inherits attributes and methods from the ShallowWater class.
    See ShallowWater for detailed comments of attributes and methods.

    Additional methods and attributes are commented below.
    '''
    NUM_STATE_VARS = 3
    NDIMS = 2

    def __init__(self):
        super().__init__()

    def set_maps(self):
        super().set_maps()

        d = {
            shallowwater_fcn_type.IsentropicVortex:
                shallowwater_fcns.IsentropicVortex,
            shallowwater_fcn_type.TaylorGreenVortex:
                shallowwater_fcns.TaylorGreenVortex,
            shallowwater_fcn_type.SteadyState: shallowwater_fcns.SteadyState,
            shallowwater_fcn_type.GravityRiemann:
                shallowwater_fcns.GravityRiemann,
        }

        self.IC_fcn_map.update(d)
        self.exact_fcn_map.update(d)
        self.BC_fcn_map.update(d)

        self.source_map.update({
            shallowwater_source_type.StiffFriction:
                shallowwater_fcns.StiffFriction,
            shallowwater_source_type.TaylorGreenSource:
                shallowwater_fcns.TaylorGreenSource,
            shallowwater_source_type.GravitySource:
                shallowwater_fcns.GravitySource,
        })

        self.conv_num_flux_map.update({
            base_conv_num_flux_type.LaxFriedrichs:
                shallowwater_fcns.LaxFriedrichs2D,
            shallowwater_conv_num_flux_type.Roe: shallowwater_fcns.Roe2D,
        })

    class StateVariables(Enum):
        Density = "h"
        XMomentum = "h u"
        YMomentum = "h v"

    def get_state_indices(self):
        ih = self.get_state_index("Density")
        ihu = self.get_state_index("XMomentum")
        ihv = self.get_state_index("YMomentum")

        return ih, ihu, ihv

    def get_momentum_slice(self):
        ihu = self.get_state_index("XMomentum")
        ihv = self.get_state_index("YMomentum")
        smom = slice(ihu, ihv + 1)

        return smom

    def get_conv_flux_interior(self, Uq):
        # Get indices/slices of state variables
        ih, ihu, ihv = self.get_state_indices()
        smom = self.get_momentum_slice()

        h = Uq[:, :, ih]  # [n, nq]
        hu = Uq[:, :, ihu]  # [n, nq]
        hv = Uq[:, :, ihv]  # [n, nq]
        mom = Uq[:, :, smom]  # [n, nq, ndims]

        # Get velocity in each dimension
        u = hu / h
        v = hv / h
        # Get squared velocities
        u2 = u ** 2
        v2 = v ** 2

        # Calculate pressure using the Ideal Gas Law
        p = 0.5 * self.g * h ** 2  # [n, nq]
        # Get off-diagonal momentum
        huv = h * u * v

        # Assemble flux matrix
        F = np.empty(Uq.shape + (self.NDIMS,))  # [n, nq, ns, ndims]
        F[:, :, ih, :] = mom  # Flux of mass in all directions
        F[:, :, ihu, 0] = h * u2 + p  # x-flux of x-momentum
        F[:, :, ihv, 0] = huv  # x-flux of y-momentum
        F[:, :, ihu, 1] = huv  # y-flux of x-momentum
        F[:, :, ihv, 1] = h * v2 + p  # y-flux of y-momentum

        return F, (u2, v2, h, p)
