from matplotlib import pyplot as plt
import numpy as np
import torch
from matplotlib import ticker
from matplotlib.colors import LogNorm, SymLogNorm

def symmetric_log(x, epsilon=1e-10):
    return np.sign(x) * np.log(np.abs(x) + epsilon)

class PlotGrid:    
    def __init__(self, num_row, num_col, font_size=18, cmap='jet'):    
      self.num_row = num_row    
      self.num_col = num_col    
      self.cmap = cmap    
      self.fig = plt.figure(    
                  figsize=(4*num_col, 3*num_row),    
                  dpi=150,    
                  constrained_layout=True     
                  )
      self.font_size = font_size
      self.prev_plot_ind = 0

    def ij_to_l(self, ind):
      assert len(ind) == 2, "ind should be (x,y) tuple or list"
      return ind[0] * self.num_col + ind[1] + 1

    #if ind is None then assume we are just appending a subfigure to the left of the previous subfigure
    def get_grid_ind(self, ind=None):
      if ind is not None:
        l_ind = self.ij_to_l(ind)
        self.prev_plot_ind=l_ind
      else:
        l_ind = self.prev_plot_ind+1
        self.prev_plot_ind += 1
      return l_ind

    def make_blank_sub_plot(self, n):
      self.prev_plot_ind += n

    def make_img_sub_plot(self, data, plot_title, ind=None, min=None, max=None, xticks = None, yticks = None, log_norm=False):
      l_ind = self.get_grid_ind(ind)
      if isinstance(data, torch.Tensor):
        data = data.numpy()
      ax = self.fig.add_subplot(self.num_row, self.num_col, l_ind)    
      ax.set_title(plot_title, fontsize = self.font_size)    
      if min is None:
        min = np.min(data)
      if max is None:
        max = np.max(data)
      if log_norm:
        if min < 0:
          print("using symlog since there are negative values")
          im = ax.imshow(data, cmap=self.cmap, extent=[0,1,0,1], aspect=1, norm=SymLogNorm(linthresh=1e-2))
        else:
          im = ax.imshow(data, cmap=self.cmap, extent=[0,1,0,1], aspect=1, norm=LogNorm(vmin=min, vmax=max))
      else:
        im = ax.imshow(data, cmap=self.cmap, vmin=min, vmax=max, extent=[0,1,0,1], aspect=1)
      cbar = self.fig.colorbar(im, ax=ax)    
      if xticks is not None:
        ax.set_xticks(xticks)
      if yticks is not None:
        ax.set_yticks(yticks)
      for axis in [ax, cbar.ax]:
        cbar.ax.tick_params(labelsize=self.font_size-2)
   
    def make_text_sub_plot(self, text, title, ind=None, font_size=12):
      l_ind = self.get_grid_ind(ind)
      ax = self.fig.add_subplot(self.num_row, self.num_col, l_ind)
      ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=font_size)
      ax.axis('off')
      ax.set_title(title, fontsize = self.font_size)

    def make_line_sub_plot(self, a, x, title, ind=None, legend=None):
      l_ind = self.get_grid_ind(ind)
      ax = self.fig.add_subplot(self.num_row, self.num_col, l_ind)

      # Plot each vector in 'a' as a line
      if legend is not None and legend:
        for vector, label in zip(a, legend):  # Assuming 'legend' is a list of labels corresponding to 'a'
            ax.plot(x, vector, label=label)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=self.font_size-4)
      else:
         for vector in a:
            ax.plot(x, vector)

      # Set plot title and adjust font size
      ax.set_title(title, fontsize=self.font_size)
      # Format y-axis ticks with scientific notation
      formatter = ticker.ScalarFormatter(useMathText=True)
      formatter.set_scientific(True)
      # formatter.set_powerlimits((-2, 2))  # This controls the range for using scientific notation
      ax.yaxis.set_major_formatter(formatter)
      
      ax.tick_params(labelsize=self.font_size-2)

    def make_box_plot(self, data, title, ind=None):
      l_ind = self.get_grid_ind(ind)
      ax = self.fig.add_subplot(self.num_row, self.num_col, l_ind)
      ax.boxplot(data)
      ax.set_title(title, fontsize=self.font_size)
      ax.tick_params(labelsize=self.font_size-2)
    
    def make_spy_plot(self, data, title, ind=None):
      l_ind = self.get_grid_ind(ind)
      ax = self.fig.add_subplot(self.num_row, self.num_col, l_ind)
      ax.spy(data, markersize=1e-10*data.shape[0])
      ax.set_title(title, fontsize=self.font_size)
      ax.set_xticks([])
      ax.set_yticks([])
      # ax.tick_params(labelsize=self.font_size-2)  

    def save(self, file_name):    
      self.fig.savefig(file_name, bbox_inches='tight')  
