import numpy as np
import matplotlib.pyplot as plt
import os
import pyvista
from matplotlib import colors
from matplotlib import tri as mtri

from matplotlib import animation
import torch

'''ONLY WORKS FOR 2D PROBLEMS'''
class figure_maker():
    def __init__(self, Re, coord, pred, sol):
        
        '''TODO: Lets split this up to show:
        Each of the channels for each run. (1x3).
        And then we call this function 3 times with:
        1. Prediction
        2. Solution
        3. Difference (relative)
        '''
        
        self.Re = Re 
        self.coord = coord.squeeze(0)
        self.n_channels = sol.shape[-1]

        if self.coord.shape[-1] == 3:
            time_series,__ = torch.sort(torch.unique(self.coord[:,-1]))
            
            self.n_nodes = int(self.coord.shape[0]/len(time_series))
            self.sol = sol.reshape(len(time_series),self.n_nodes,self.n_channels)
            self.pred = pred.reshape(len(time_series),self.n_nodes,self.n_channels)
            self.dif = self.sol - self.pred
            self.coord = self.coord[:self.n_nodes,:2] 
        else:
            self.sol = sol
            self.pred = pred
            self.dif = sol - pred
            

        plt.rcParams["image.cmap"] = "inferno"
        self.colourmaps = ["inferno", "inferno", "twilight"]
        self.fig, self.ax = plt.subplots(3, 3, figsize=(14, 8))

        # use sol as the basis:
        self.val_min = torch.min(self.sol, dim=1).values
        self.val_max = torch.max(self.sol, dim=1).values
        #self.val_norm = [None, None, colors.TwoSlopeNorm(vmin=self.val_min[0,-1], vcenter=0, vmax=self.val_max[0,-1])]

        # Set background color to black
        self.fig.set_facecolor("black")
        for i in range(3):
            for j in range(3):
                self.ax[i,j].set_facecolor("black")

        self.triang = mtri.Triangulation(
                self.coord[:,0],
                self.coord[:,1],
                #self.faces[num],
        )
        
        self.time_series = time_series

    def get_frame(self, idx):
        
        time_step = self.time_series[idx]
        
        for channel in range(3):
            for i, type in enumerate([self.sol, self.pred, self.dif]):
                # Channel 0
                self.ax[channel,i].cla()
                self.ax[channel,i].set_aspect("equal")
                self.ax[channel,i].tripcolor(self.triang, type[idx,:,channel], vmin=self.val_min[idx,channel], vmax=self.val_max[idx,channel], cmap=self.colourmaps[i])
                self.ax[channel,i].add_patch(plt.Circle((0, 0), radius=1, fc='grey', zorder=10, edgecolor='k'))
        
        self.ax[0,0].set_title(f"Vortex Shedding Re {self.Re:n} at dt={time_step:n}", color="white")
        self.ax[0,0].set_ylabel(f"Ux", color="white")
        self.ax[1,0].set_ylabel(f"Uy", color="white")
        self.ax[2,0].set_ylabel(f"Pressure", color="white")
        self.ax[2,0].set_xlabel(f"Ground-Truth", color="white")
        self.ax[2,1].set_xlabel(f"Prediction", color="white")
        self.ax[2,2].set_xlabel(f"Difference", color="white")

        self.fig.subplots_adjust(
            left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.05, hspace=0.05
        )

        return self.fig
    
    def get_anim(self):
        ani = animation.FuncAnimation(
                self.fig,
                self.get_frame,
                frames=len(self.time_series),
                interval = 1
            )

        return ani


if __name__ == '__main__':
    
    directory = os.path.realpath(__file__).split('/')
    directory = '/'.join(directory[:-1])

    #cases = ['c5', 'c6']
    Re_numbers = [60, 65]

    cases = ['c6']
    # Re_numbers = [50]

    for case in cases:
        for Re in Re_numbers:

            sol, coords = get_solution_file(Re=Re, case=case)

            # data = np.loadtxt(f'{directory}/c6/postProcessing/forceCoeffs_object/0/coefficient.dat', skiprows=13)
            # output_dict = {'Time': data[:,0],
            #             'Cd': data[:,1],
            #             'Cl': data[:,4],
            #             'Cl(f)': data[:,5],
            #             'Cl(r)': data[:,6]
            #     }
            # sol['force_coeffs'] = output_dict
            print(f'Solution for {Re:n} loaded and making animation...')
            figure_class = figure_maker(Re=Re,coord=coords,sol=sol,case=case)
            ani = animation.FuncAnimation(
                figure_class.fig,
                figure_class.get_frame,
                frames=len(sol['time_data']),
                interval = 1
            )
            ani.save(f"{directory}/{case}/animation_Re{Re:n}.gif")
            print(f'Animation saved in: {directory}/{case}/animation_Re{Re:n}.gif')