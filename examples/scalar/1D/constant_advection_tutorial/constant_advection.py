import numpy as np

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 16,
    "xmin" : -1.,
    "xmax" : 1.,
    "PeriodicBoundariesX" : ["x1","x2"]
}

Physics = {
	"Type" : "ConstAdvScalar",
	"ConvFluxNumerical" : "LaxFriedrichs", 
	"ConstVelocity" : 1.,
}

InitialCondition = {
	"Function" : "Sine",
	"omega" : 2.*np.pi,
}

ExactSolution = InitialCondition.copy()

Numerics = {
	"SolutionOrder" : 2,
	"SolutionBasis" : "LagrangeSeg",
	"Solver" : "DG",
}

TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : 0.5,
	"CFL" : 0.1,
	"TimeStepper" : "RK4"
}

Output = {
	"AutoPostProcess" : False,
}

















