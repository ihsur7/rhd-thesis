def Visualise_2(main_df, data_df, time_array):
    global coords_df_2, plotter, grid, vtkpoints, spacing
    coords_df_2 = main_df[['x', 'y', 'z']].copy()
    for i in ['t'+str(i) for i in list(range(len(time_array)))]:
        coords_df_2 = coords_df_2.join(data_df[i])

    vtkpoints = pvgeo.points_to_poly_data(coords_df_2)
    bounds = vtkpoints.bounds
    margin = 100
    n = 500 #600
    ldim = bounds[-1] + margin*2
    grid = pyvista.UniformGrid((n,n,n))
    grid.origin = [bounds[0] - margin]*3
    spacing = ldim/(n-1)
    grid.spacing = [spacing]*3

    plotter = pyvista.Plotter(notebook=False)
    plotter.add_slider_widget(change_t, [0,len(time_array)-1], value = 0, title='Time')
    plotter.show()

def change_t(value):
    global vox_valid
    time = 't'+str(int(value))
    
    vox = grid.interpolate(vtkpoints,radius=spacing*2,progress_bar=False)
    mask = vox[time]>0
    vox_valid = vox.extract_points(mask, adjacent_cells=False)
    plotter.add_mesh(vox_valid)