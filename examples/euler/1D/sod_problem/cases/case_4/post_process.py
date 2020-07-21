import code

import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

### Postprocess
plot.PreparePlot()
skip=0

fname = "exact_shu.pkl"
solver1 = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver1.Time)
solver1.Time = 2.0
mesh1 = solver1.mesh
physics1 = solver1.physics


fname = "data_final.pkl"
solver2 = readwritedatafiles.read_data_file(fname)
print('Solution Final Time:', solver2.Time)
mesh2 = solver2.mesh
physics2 = solver2.physics


# Density
plot.plot_solution(mesh1, physics1, solver1, "Density", plot_numerical=False, plot_exact=True, plot_IC=False, create_new_figure=True, 
			ylabel=None, fmt='k-.', legend_label="Exact", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False,skip=skip, show_elem_IDs=True)
plot.plot_solution(mesh2, physics2, solver2, "Density", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=False, 
			ylabel=None, fmt='b-', legend_label="Case 4", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False,skip=skip, show_elem_IDs=False)

plot.SaveFigure(FileName='density', FileType='pdf', CropLevel=2)

# Velocity 
plot.plot_solution(mesh1, physics1, solver1, "Velocity", plot_numerical=False, plot_exact=True, plot_IC=False, create_new_figure=True, 
			ylabel=None, fmt='k-.', legend_label="Exact", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False,skip=skip, show_elem_IDs=True)
plot.plot_solution(mesh2, physics2, solver2, "Velocity", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=False, 
			ylabel=None, fmt='b-', legend_label="Case 4", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False,skip=skip, show_elem_IDs=False)

plot.SaveFigure(FileName='velocity', FileType='pdf', CropLevel=2)

# Velocity 
plot.plot_solution(mesh1, physics1, solver1, "Pressure", plot_numerical=False, plot_exact=True, plot_IC=False, create_new_figure=True, 
			ylabel=None, fmt='k-.', legend_label="Exact", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False,skip=skip, show_elem_IDs=True)
plot.plot_solution(mesh2, physics2, solver2, "Pressure", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=False, 
			ylabel=None, fmt='b-', legend_label="Case 4", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False,skip=skip, show_elem_IDs=False)

plot.SaveFigure(FileName='pressure', FileType='pdf', CropLevel=2)
plot.ShowPlot()