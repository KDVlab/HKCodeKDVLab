# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:02:36 2024

@author: WORKSTATION
"""

import imageio, sys, os, importlib
import matplotlib.pyplot as plt
import tifffile as tf
import cv2 as cv
import numpy as np
from scipy import signal

from scipy.optimize import curve_fit

#%%

#Specify the location of the Tools folder
CodeDR=r"C:\Users\WORKSTATION\Desktop\HamzaCode\HKCodeKDVLab"
#Specify where the data is and where plots will be saved
dataDR=r"F:/ferro/air/largedrops/run_7"

#Use telegram to notify
tokenloc = r"F:\ferro\token.txt"


os.chdir(CodeDR) #Set  current working direcotry to the code directory


sys.path.append('./Scripts/FerroFluidTrack') #Add the tools to the system path so modules can be imported

#Import required modules
import AirDropFunctions as adf
importlib.reload(adf)



import FrametoTimeAndField as ftf
importlib.reload(ftf)

#Remove to avoid cluttering path
sys.path.remove('./Scripts/FerroFluidTrack') #Remove tools from path

#Set working directory to data location
os.chdir(dataDR)

#%%
#Import the images of interest and a base image for background subtraction
tifobj = tf.TiffFile('run_MMStack_Pos0.ome.tif')
numFrames = len(tifobj.pages)
ims =  tf.imread('run_MMStack_Pos0.ome.tif',key=slice(0,numFrames))


#%%
leftcrop = 100
rightcrop = 100
botcrop = 100
topcrop = 0
newims=np.zeros((ims.shape[0],ims.shape[1]-(botcrop+topcrop),ims.shape[2]-(leftcrop+rightcrop)))
dat = np.load('run3locs.npy')
xshifts = (dat[:,2]+dat[:,3])/2
xshifts = xshifts-xshifts[0]
for i, im in enumerate(ims):
	xshift = int(xshifts[i])
	print(xshift)
	newims[i] = adf.cropper(im, [[leftcrop+xshift,topcrop],[-rightcrop+xshift,-botcrop]])
#%%

import matplotlib.patches as patches
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",'font.size': 16,
})

import matplotlib.animation as animation


from matplotlib_scalebar.scalebar import ScaleBar


gaussvals = np.loadtxt('guassconvert7.csv',delimiter=',')[:,2]
maxgauss = np.max(gaussvals)

pixsize = 2.24e-6
insecperframe = .5
xrealtime = 80
skipframe = 4
savename = 'stabilforce.mp4'

inFPS = 1/insecperframe
outputFPS = inFPS*xrealtime


dim = newims.shape[1:]
dimr = dim[1]/dim[0]
# ax refers to the axis propertis of the figure
fig, ax = plt.subplots(1,1,figsize=(4,4/dimr))

im = ax.imshow(newims[0],cmap='gray')
#txt = ax.text(.2, .05, 'B={x:.1f}G'.format(x=gaussvals[0]),fontsize=12, ha='center',transform=plt.gca().transAxes)
txt = ax.text(.4, .05, r'B: ',fontsize=12, ha='center',transform=plt.gca().transAxes)
rect = patches.Rectangle((600, 1010), 400*gaussvals[0]/maxgauss, 30, linewidth=1, edgecolor='r', facecolor='red')
ax.add_patch(rect)
scalebar = ScaleBar(pixsize,frameon=False,location='upper left',pad=0.5) 
ax.add_artist(scalebar)
ax.axis('off')

ax.get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
ax.get_yaxis().set_visible(False) # this removes the ticks and numbers for y axis

fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)



def animate(i):
	im.set_data(newims[i])
	rect.set_width(400*gaussvals[i]/maxgauss)
	#txt.set_text('B={x:.1f}G'.format(x=gaussvals[i]))
	return im,rect,


ani = animation.FuncAnimation(fig, animate, frames=np.arange(0,len(ims),skipframe), interval=insecperframe/1000/xrealtime,
                     repeat_delay=1000, blit=True)


#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
print(len(ims)/outputFPS)
plt.show()
#%%

Writer = animation.writers['ffmpeg']
writer = Writer(fps=outputFPS/skipframe,extra_args=['-vcodec', 'libx264'])
ani.save(savename,writer=writer,dpi=200)