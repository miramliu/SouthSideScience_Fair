import os
import numpy as np
import imageio
from PIL import Image

# pop-out plots
import matplotlib
import matplotlib.pyplot as plt


# plot defaults
im_kwargs = {'cmap':'gray', 'vmin':0}

# load the data
data_dir = 'recon_process/'
mat_files = sorted([data_dir+f for f in os.listdir(data_dir) if 'matrix' in f])  # png
arr_files = sorted([data_dir+f for f in os.listdir(data_dir) if 'array' in f])  # npy

# camera
image = Image.open('camera.png').convert('L')
cam = np.asarray(image)
cam = cam/np.max(cam)
cam = -cam + 1
cy,cx = cam.shape
print('cam', cx, cy)

cam_width = 100
cam_length = cam_width*cy/cx


# load all the arrays
arrs = [np.load(f) for f in arr_files] 

# load the xcat
xcat = np.load(arr_files[-1])

# crop
Ncrop = 50
xcat = xcat[Ncrop:-Ncrop, Ncrop:-Ncrop]

# save shape
Nx, Ny = xcat.shape
print(Nx, Ny)


def get_angle(v1, v2):
    ''' function for getting angle between two 2D vectors'''
    def unit_vector(vector):
        return np.array(vector) / np.linalg.norm(vector)
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))*180/np.pi



pad = 30
dN = 300
R0 = 450 #np.sqrt((Nx/2)**2 + (Ny/2)**2)
a0 = 30    # degrees
a_fan = 40 # degrees

# plot image, get ROIs
fig,[ax,ax2] = plt.subplots(1,2, figsize=[10,7], dpi=200)
#ax1.axis('off')
ax2.axis('off')
fig.tight_layout(pad=0)

def show_xcat():
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(-dN, Nx+dN)
    ax.set_ylim(Ny+dN, -dN)

    ax.set_facecolor((0,0,0))
    ax.imshow(xcat, **im_kwargs)
    #ax.imshow(xcat_circ, **im_kwargs)

    # add source path circle
    circ1 = plt.Circle( (Nx/2, Ny/2), R0, fill=False, color='w', lw=10)
    ax.add_artist(circ1)

    # draw "X" for exit
    ax.plot(Nx+dN-pad, -dN+pad, 'ws', markersize=6)
    ax.plot(Nx+dN-pad, -dN+pad, 'rx', markersize=6, markeredgewidth=1)

    ax.set_title('Patient in CT scanner')
        
    # show start, a0
    ax.plot(Nx/2, Ny/2-R0, 'ro', markerfacecolor="None", markersize=10)
    #ax.text(Nx/2+30, Ny/2-R0+25, 'start', color='r', fontsize=8)
    ax.arrow(Nx/2-40, Ny/2-R0, -50, 5, color='r', head_width=15)



# initialize plots
show_xcat()
#ax1.imshow(betas[0], **im_kwargs)
ax2.imshow(arrs[0], cmap='gray')
ax2.set_title('Output image')

signal = plt.ginput(1, timeout=-1)
x1, y1 = signal[0][0], signal[0][1]
while x1<Nx+dN-pad and y1>-dN+pad:
    ax.cla()
    

    #ax.imshow(cam, extent=(x1-cam_width, x1+cam_width, y1+cam_length, y1-cam_length), cmap='gray')
    show_xcat()

    ax.plot(x1, y1, '+', color='r', markersize=5)
    a0 = get_angle([0,-1], [x1-Nx/2,y1-Ny/2])
    if x1>Nx/2:
        a0=-a0 + 360
        
    for angle in range(0,int(a0)):
        t0 = -angle*np.pi/180
        x0 = Nx/2 + R0*np.sin(t0)
        y0 = Ny/2 - R0*np.cos(t0)
        ax.plot(x0, y0, 'o', color='r', markersize=5)

    col='cornflowerblue'
    # draw the source/detector wedge
    t0 = -a0*np.pi/180
    x0 = Nx/2 + R0*np.sin(t0)
    y0 = Ny/2 - R0*np.cos(t0)
    ax.plot(x0, y0, 'o', color=col, markersize=5)

    a1 = -a0 + 90 - a_fan/2
    a2 = -a0 + 90 + a_fan/2
    wedge_det = matplotlib.patches.Wedge((x0, y0), 2*R0, a1, a2, fill=False, color=col)
    ax.add_artist(wedge_det)
    
    # draw the updated matrix/beta
    i_img = int(1000*a0/360)
    #ax1.imshow(betas[i_img])
    ax2.imshow(arrs[i_img], cmap='gray')

    fig.canvas.draw()

    # loop
    signal = plt.ginput(1, timeout=-1)
    x1, y1 = signal[0][0], signal[0][1]


plt.close()

