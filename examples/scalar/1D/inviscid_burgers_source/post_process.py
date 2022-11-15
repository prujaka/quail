import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles

# Read data file
fname = "Data_final.pkl"
solver = readwritedatafiles.read_data_file(fname)

# Unpack
mesh = solver.mesh
physics = solver.physics

# Compute L2 error
post.get_error(mesh, physics, solver, "Scalar")

''' Plot '''
plot.prepare_plot()
# Initial condition
plot.plot_solution(mesh, physics, solver, "Scalar", plot_numerical=False,
		plot_exact=False, plot_IC=True, create_new_figure=False,
		ylabel=None, fmt='k--', legend_label="IC")

# DG solution
plot.plot_solution(mesh, physics, solver, "Scalar", plot_numerical=True,
		plot_exact=False, plot_IC=False, create_new_figure=True,
		ylabel=None, fmt='go', legend_label="DG")

# Exact solution
# plot.plot_solution(mesh, physics, solver, "Scalar", plot_exact=True,
# 		plot_numerical=False, create_new_figure=False, fmt='k-')

plot.show_plot()
