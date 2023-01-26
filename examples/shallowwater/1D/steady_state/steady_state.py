import numpy as np

TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : 1.0,
	"CFL" : 0.1,
	"TimeStepper" : "SSPRK3",
}

Numerics = {
	"SolutionOrder" : 1,
	"SolutionBasis" : "LagrangeSeg",
	"Solver" : "DG",
	"L2InitialCondition" : False,
	"ApplyLimiters" : "PositivityPreserving",
}

Output = {
	"AutoPostProcess" : True,
}

Mesh = {
	"File" : None,
	"ElementShape" : "Segment",
	"NumElemsX" : 100,
	"xmin" : 0.,
	"xmax" : 1.,
}

Physics = {
	"Type" : "ShallowWater",
	"ConvFluxNumerical" : "LaxFriedrichs",
	"GravitationalAcceleration" : 9.81,
}

InitialCondition = {
	"Function" : "SteadyState",
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
	"x1" : {
		"BCType" : "SlipWall"
		},
	"x2" : {
		"BCType" : "SlipWall"
		}
}
