import numpy as np

TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : 1.,
	"CFL" : 0.1,
	"TimeStepper" : "SSPRK3",
}

Numerics = {
	"SolutionOrder" : 2,
	"SolutionBasis" : "LagrangeSeg",
	"Solver" : "DG",
}

Mesh = {
	"File" : None,
	"ElementShape" : "Segment",
	"NumElemsX" : 80,
	"xmin" : 0.,
	"xmax" : 2*np.pi,
}

Physics = {
	"Type" : "Burgers",
	"ConvFluxNumerical" : "LaxFriedrichs",
}

InitialCondition = {
	"Function" : "SimpleGaussian",
	"x0" : 1,
}

ExactSolution = {
	"Function" : "SimpleGaussian",
	"x0" : 1,
}

Output = {
	"Prefix" : "Data",
}

BoundaryConditions = {
	"x1" : {
		"BCType" : "Extrapolate",
	},
	"x2" : {
		"BCType" : "Extrapolate",
	},
}

# SourceTerms = {
# 	"Source1" : { # Name of source term ("Source1") doesn't matter
# 		"Function" : "SimpleSource",
# 		"nu" : 0.,
# 	},
# }

SourceTerms = {
	"Source1" : { # Name of source term ("Source1") doesn't matter
		"Function" : "SimpleGaussianSource",
		"x0" : 1.,
	},
}