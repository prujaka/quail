import sys; sys.path.append('../../../src'); sys.path.append('./src')
import numpy as np
import code
import solver.DG as Solver
import physics.euler.euler as Euler
import meshing.common as MeshCommon
import processing.post as Post
import processing.plot as Plot
import general


### Mesh
Periodic = False
mesh = MeshCommon.mesh_1D(Uniform=True, nElem=25, xmin=-1., xmax=1., Periodic=Periodic)


### Solver parameters
EndTime = 0.1
nTimeStep = 100
InterpOrder = 2
Params = general.SetSolverParams(InterpOrder=InterpOrder,EndTime=EndTime,nTimeStep=nTimeStep,
								 InterpBasis="LagrangeEqSeg",TimeScheme="RK4",InterpolateIC=True)


### Physics
EqnSet = Euler.Euler1D(Params["InterpOrder"], Params["InterpBasis"], mesh)
EqnSet.SetParams(GasConstant=1.,SpecificHeatRatio=3.,ConvFlux="LaxFriedrichs")
# Initial conditions
EqnSet.set_IC(IC_type="SmoothIsentropicFlow", a=0.9)
EqnSet.set_exact(exact_type="SmoothIsentropicFlow", a=0.9)

# Boundary conditions
if not Periodic:
	for ibfgrp in range(mesh.nBFaceGroup):
		BFG = mesh.BFaceGroups[ibfgrp]
		if BFG.Name is "Left":
			EqnSet.set_BC(BC_type="StateAll", fcn_type="SmoothIsentropicFlow", a=0.9)
		elif BFG.Name is "Right":
			EqnSet.set_BC(BC_type="StateAll", fcn_type="SmoothIsentropicFlow", a=0.9)

### Solve
solver = Solver.DG(Params,EqnSet,mesh)
solver.solve()


### Postprocess
# Error
TotErr,_ = Post.L2_error(mesh, EqnSet, solver, "Density")
# Plot
Plot.PreparePlot()
Plot.PlotSolution(mesh, EqnSet, solver, "Energy", PlotExact=True, Equidistant=True)
Plot.ShowPlot()


# code.interact(local=locals())
