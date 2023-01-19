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
    pass


class ShallowWater1D(ShallowWater):
    pass


class ShallowWater2D(ShallowWater):
    pass
