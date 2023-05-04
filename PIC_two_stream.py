# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 17:08:06 2021

@author: James.Bland
"""

# from scipy.constants import epsilon_0, m_e, e
import numpy as np
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from celluloid import Camera

e = 1
epsilon_0 = 1
m_e = 1
# %% 
# discretise the charge density, E and electrostatic potential on this grid

# Poisson equation solved on this grid

# Electric field on grid

# Electric field on superparticle

def density_to_potential(rho_on_grid, laplace_mat):
    
    potential_grid = spsolve(laplace_mat, -rho_on_grid * epsilon_0)
    
    return potential_grid


def potential_to_Efield(phi_on_grid, dx):
    
    Efield_on_grid = - np.gradient(phi_on_grid, dx)
    
    return Efield_on_grid


def grid_to_superparticle(pos_particle, pos_grid, var_grid):
    
    var_particle = np.interp(pos_particle, pos_grid, var_grid)
    
    return var_particle


def Efield_to_acc(Efield):
    
    acc = -e * Efield / m_e
    
    return acc


def update_pos_vec(pos_particle, vel_particle, acc_particle, dt, box_len):
    
    
    vel_particle += acc_particle * dt / 2.0
    
    pos_particle += vel_particle * dt
    
    pos_particle = np.mod(pos_particle, box_len)

    vel_particle += acc_particle * dt /2.0
    
    return pos_particle, vel_particle


def particle_to_grid(pos_particle, pos_grid, dx):
    
    pos_diff = np.abs(pos_particle.reshape(-1, 1) - pos_grid.reshape(1, -1)) /dx
    
    pos_diff[pos_diff >= 1] = 1
    
    pos_weight = 1 - pos_diff
    
    sum_on_grid = np.sum(pos_weight, axis=0)
    
    return sum_on_grid


def laplace_matrix(n_grid, dx):
    
    e = np.ones(n_grid)
    diags = np.array([-1,0,1])
    vals  = np.vstack((e,-2*e,e))
    Lmtx = sp.spdiags(vals, diags, n_grid, n_grid);
    Lmtx = sp.lil_matrix(Lmtx) # tranform mtx type to modify entries
    Lmtx[0,n_grid-1] = 1
    Lmtx[n_grid-1,0] = 1
    Lmtx /= dx**2
    Lmtx = sp.csr_matrix(Lmtx) # transform mtx type

    return Lmtx

def PIC_two_stream(dx, x_max, dt, t_max, num_par, v0, vth, pert):
    
    pos_grid = np.arange(0, x_max+dx, dx)

    n_grid = pos_grid.size
    n_half = np.ceil(num_par/2).astype('int')
    
    t_grid = np.arange(0, t_max, dt)

    np.random.seed(42)
    
    pos_particle = np.random.random_sample(num_par) * x_max
    
    vel_particle = vth * np.random.randn(num_par) + v0
    vel_particle[n_half:] *= -1
    
    vel_particle *= (1 + pert*np.sin(2*np.pi*pos_particle /x_max))
    
    laplace_mat =  laplace_matrix(n_grid, dx)
    
    fig = plt.figure(figsize = (5,4), dpi=80)
    
    camera = Camera(fig)
    # plt.cla()
    plt.scatter(pos_particle[:n_half], vel_particle[:n_half], s=.4, color='blue', alpha=0.5)
    plt.scatter(pos_particle[n_half:], vel_particle[n_half:], s=.4, color='red', alpha=0.5)
    plt.axis([0,x_max,-2*v0,2*v0])
    # plt.pause(0.001)
    camera.snap()
    
    pos_particle_array = np.zeros([ t_grid.size, pos_particle.size])
    vel_particle_array = np.zeros([ t_grid.size, pos_particle.size])

    for ii, time in enumerate(t_grid):
        
        number_grid = particle_to_grid(pos_particle, pos_grid, dx)
        
        density_grid = e * 0.01* ((num_par/ n_grid) - number_grid)
        
        potential_grid = density_to_potential(density_grid, laplace_mat)
    
        Efield_grid = potential_to_Efield(potential_grid, dx)
        
        Efield_particle = grid_to_superparticle(pos_particle, pos_grid, Efield_grid)
        
        acc_particle = Efield_to_acc(Efield_particle)
        
        pos_particle, vel_particle = update_pos_vec(pos_particle, vel_particle, acc_particle, dt, x_max)
        
        pos_particle_array[ii, :] = pos_particle
        vel_particle_array[ii, :] = vel_particle

        # plt.cla()
        plt.scatter(pos_particle[:n_half], vel_particle[:n_half], s=.4, color='blue', alpha=0.5)
        plt.scatter(pos_particle[n_half:], vel_particle[n_half:], s=.4, color='red', alpha=0.5)
        plt.axis([0,x_max,-2*v0,2*v0])
        # plt.pause(0.001)
        camera.snap()
        
    animation = camera.animate()
    animation.save('pic.gif', writer='imagemagick', fps=5)
        
    return pos_particle_array, vel_particle_array, n_half
        

def animate_func(pos_particle, vel_particle, n_half):
    
    plt.scatter(pos_particle[:n_half], vel_particle[:n_half], s=.4, color='blue', alpha=0.5)
    plt.scatter(pos_particle[n_half:], vel_particle[n_half:], s=.4, color='red', alpha=0.5)
    particles = plt.axis([0,x_max,-2*v0,2*v0])
    
    return particles
    
    
if __name__ == '__main__':


    dx = 50/400
    x_max = 50
    dt = .1
    t_max = 50
    
    num_par = 40000
    v0 = 3
    vth = 1
    pert=0.1
    
    pos_particle_array, vel_particle_array, n_half = PIC_two_stream(dx, x_max, 
                                                                    dt, t_max, num_par, v0, vth, pert)

    # animate = lambda iTime : animate_func(pos_particle_array[iTime, :], vel_particle_array[iTime, :], n_half)
    
    # fig = plt.figure()
    
    # ani = animation.FuncAnimation(fig, animate, frames=pos_particle_array.shape[0],
    #                           interval=10, blit=True)
    
    # ani.save('pic', fps=30)