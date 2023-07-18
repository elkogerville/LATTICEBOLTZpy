def boltzman_lattice(objects, Nx, Ny, v0, timesteps, **kwargs):
    """
    this function performs a boltzman lattice hydrodynamical simulation for a fluid given boundaries
    ------------------------------------------------------------------------------------------------
    objects [boolean array]: Nx x Ny boolean array; where True designates a point inside given boundary
    and False a point outside boundary
    Nx, Ny [integers]: simulation x, y, boundary space
    v0 [float]: initial velocity
    timesteps [integer]: simulation timesteps
    --------
    **kwargs
    rh0 [float]: average density
    tau [float]: viscocity
    flow_dir [integer]: from 0 to 9, indicates initial velocity direction [3 == to the right, goes CW]
    plot_number [integer]: number of timesteps per plot
    plot_curl [boolean]: default = True, plots curl; if False will plot velocity instead
    ------------------------------------------------------------------------------------
    OUTPUT [matplotlib plot]: sequence of timestep plots
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    #######################
    # simulation parameters
    #######################
    
    # simulation bounds
    Nx = int(Nx)
    Ny = int(Ny)
    timesteps = int(timesteps)
    # average density
    rho0 = 100
    if ('rho0') in kwargs:
        rho0 = kwargs['rho0']
    # viscocity
    tau = 0.6
    if ('tau') in kwargs:
        tau = kwargs['tau']
    # define fluid velocity as rightward    
    flow_dir = 3 
    if ('flow_dir') in kwargs:
        flow_dir = kwargs['flow_dir']
    # ensure fluid direction does not exceed number of lattice nodes    
    if (flow_dir < 0) | (flow_dir > 9):
        print('flow direction must be in range: 0≤flow_dir≤9')
        raise SystemExit
    
    # number of timesteps per plot
    plot_number = 30
    if ('plot_number') in kwargs:
        plot_number = kwargs['plot_number']
    plot_number = int(plot_number)
    plot_curl = True
    if ('plot_curl') in kwargs:
        plot_curl = kwargs['plot_curl']
    
    # number of lattice nodes
    lattice = 9
    # discrete velocities in x, y, directions
    vdx = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    vdy = np.array([0, 1, 1, 0,-1,-1, -1,  0,  1])
    # discrete probabilities of a particle traveling to a node; normalized to 1
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])
    
    ####################
    # initial conditions
    ####################
    # initiate (Nx x Ny x 9) fluid lattice with random variations
    F = np.ones((Ny, Nx, lattice)) + .01*np.random.rand(Ny, Nx, lattice)
    
    # assign direction of flow: rightward velocity by default
    F[:, :, 3] += v0
    # calculate initial rho density
    rho = np.sum(F,2)
    for i in range(lattice):
        F[:,:,i] *= rho0 / rho
    
    ############
    # simulation
    ############
    
    print('\nsimulation running....  /ᐠ –ꞈ –ᐟ\<[pls be patient]')
    
    for time in range(timesteps):
    
        # stream particles in direction of flow
        for i, vx, vy in zip(range(lattice), vdx, vdy):
            F[:, :, i] = np.roll(F[:, :, i], vx, axis = 1)
            F[:, :, i] = np.roll(F[:, :, i], vy, axis = 0)

        # calculate collisions with boundary
        boundary = F[objects, :]
        # reflect velocities
        boundary = boundary[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

        # calculate fluid properties
        rho = np.sum(F, 2)
        xvel = np.sum(F * vdx, 2) / rho
        yvel = np.sum(F * vdy, 2) / rho
        
        # calculate equilibrium force
        F_equil = np.zeros(F.shape)
        for i, vx, vy, w in zip(range(lattice), vdx, vdy, weights):
            F_equil[:, :, i] = rho*w * (1 + 3*(vx*xvel+vy*yvel) + 9*(vx*xvel+vy*yvel)**2/2 - 3*(xvel**2+yvel**2)/2 )
        
        # calculate next timestep
        F += -(1./tau) * (F - F_equil)

        # set boundary
        F[objects, :] = boundary
        # set velocity inside cylinder to zero
        xvel[objects] = 0
        yvel[objects] = 0
        
        # plot curl
        if plot_curl == True:
            if time%plot_number == 0:
                # initiate figure
                fig, ax = plt.subplots(figsize = (7, 7))
                plt.cla() # clear figure
                # calculate curl
                curl = (np.roll(xvel, -1, axis=0) - np.roll(xvel, 1, axis=0)) - (np.roll(yvel, -1, axis=1) - np.roll(yvel, 1, axis=1))
                curl[objects] = np.nan
                curl = np.ma.array(curl, mask = objects)
                # plot curl and boundaries
                plt.imshow(curl, cmap = 'bwr', origin = 'lower')
                plt.imshow(~objects, cmap = 'gray', alpha = .2, origin = 'lower', interpolation = 'antialiased')
               
                plt.pause(.001)
                plt.show()
                
        # plot velocity
        if plot_curl is not True:
            if time%plot_number == 0:
                # initiate figure
                fig, ax = plt.subplots(figsize = (7, 7))
                plt.cla() # clear figure
                # plot velocity and boundaries
                plt.imshow(np.sqrt(xvel**2+yvel**2), cmap = 'PuOr')
                plt.imshow(~objects, cmap = 'gray', alpha = .2, origin = 'lower', interpolation = 'antialiased')

                plt.pause(.001)
                plt.show()
                #plt.savefig('animation1.png', dpi = 1000)
    print('simulation complete [yay!!! (ﾐΦ ﻌ Φﾐ)✿ *ᵖᵘʳʳ*]')
    return 0
