from pylab import *
import os
# Get an overview on the color maps from
# https://matplotlib.org/stable/tutorials/colors/colormaps.html

try:
    colormap = sys.argv[1]  #e.g. 'cool'
    N = int(sys.argv[2])
except:
    raise Exception(\
        'Usage: python3 get_colors.py <colormap_name> <number_of_colors>. \
            (E.g. python3 get_colors.py viridis 4)')

# Include here the colormap name you want to extract colors from
cmap = cm.get_cmap(colormap, N)
colors = []

for i in range(cmap.N):
    rgba = cmap(i)
    colors.append(matplotlib.colors.rgb2hex(rgba))

print(colors)