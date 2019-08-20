'''
Functions to find properties of droplet such as contact angle etc
'''
import numpy as np
from scipy.optimize import curve_fit
from scipy.misc import derivative
from Tools.EdgeDetection import *

def circle(x,a,b,r):
	'''
	Simply a function for a circle
	'''
	return np.sqrt(r**2-(x-a)**2)+b

def pol2ndorder(x,a,b,c):
	return a+b*x+c*x**2

def pol3rdorder(x,a,b,c,d):
	return a+b*x+c*x**2+d*x**3

def pol4thorder(x,a,b,c,d,e):
	return a+b*x+c*x**2+d*x**3+e*x**4

def slopeptline(x,m,x0,y0):
	'''
	slope point form for a line
	'''
	return m*(x-x0)+y0

def linfx(x,m,b):
	'''
	equation of a line
	'''
	return m*x + b


def splitlinefinder(locs,centerybuff):
	'''
	Finds the rough center location based on some buffer
	'''
	#Which area to use for splitting line (ie how far up y)
	splitlineavregion=np.max(locs[:,0])-centerybuff
	splitline=np.mean(locs[:,1][locs[:,0]>splitlineavregion])
	return splitline

def datafitter(locs,left,pixelbuff,zweight,fitfunction,fitguess):
	'''
	This function takes a numpy array of edge location xy values and returns
	the location of the contact point as well as a fitted function
	Parameters:
	    locs: the input array
	    left: True for left side of droplet, false for right
	    pixelbuff: How many pixels in xy to include for fit
	    cfitguess: Guess's for fit parameters
	    zweight: anything below 1 gives extra weight to the zero

	Circle fitting can be a bit buggy, need to be fairly close with parameters.
	Better to overestimate radius somewhat.
	'''

	#Get the min or max position
	if left==True:
	    contactloc=np.argmin(locs[:,0])
	else:
	    contactloc=np.argmax(locs[:,0])

	contactx=locs[contactloc,0]
	contacty=locs[contactloc,1]

	#Set up trimmed Data set for fit using buffered area and only positive values
	#Will need to change to also include data from reflection
	if left==True:
	    conds=np.logical_and.reduce((locs[:,0]<contactx+pixelbuff[0],locs[:,1]>contacty,locs[:,1]<contacty+pixelbuff[1]))
	else:
	    conds=np.logical_and.reduce((locs[:,0]>contactx-pixelbuff[0],locs[:,1]>contacty,locs[:,1]<contacty+pixelbuff[1]))
	    
	trimDat=locs[conds]-[contactx,contacty]

	#Set up weighting
	sigma = np.ones(len(trimDat[:,1]))
	sigma[np.argmin(trimDat[:,1])] = zweight
	#The fitter is annoyingly dependant on being close to the actual parameters values to get a good guess
	popt, pcov = curve_fit(fitfunction, trimDat[:,0], trimDat[:,1],p0=fitguess, sigma=sigma,maxfev=5000)
	def paramfunc(x):
	    return fitfunction(x,*popt)
	m0=derivative(paramfunc,0)
	#Return angle in degrees
	thet=np.arctan(m0)*180/np.pi
	return [contactx,contacty,thet,m0,popt,pcov]

def flipper(toflip,x1,y1,x2,y2):
	'''
	Flips an array of xy points about a line defined by x1,x2,y1,y2
	'''
	#Create line to flip
	l=np.array([x2-x1,y2-y1])
	#Shift array to be flipped so they are vectors from origin of l
	shiftarray= toflip - [x1,x2]
	projection = l * np.dot(toflip, l) / np.dot(l, l)
	return 2*projection-shiftarray

def rotator(torotate,angle,ox,oy):
	"""
	Rotate an x,y list of points counterclockwise by a given angle around a given origin.
	"""
	'''
	ox, oy = origin
	px, py = point

	qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
	qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
	'''
	#shift array to origin
	shiftedarray=torotate-[ox,oy]
	rotatedarray=np.zeros(shiftedarray.shape)
	#Apply rotation transform
	rotatedarray[:,0] = np.cos(angle) * shiftedarray[:,0] - np.sin(angle) * shiftedarray[:,1]
	rotatedarray[:,1] = np.sin(angle) * shiftedarray[:,0] + np.cos(angle) * shiftedarray[:,1]
	return rotatedarray+[ox,oy]

def xflipandcombine(toflip):
	'''
	Flips an already rotated edge point array and combines the top and the bottom
	'''
	#Find where to flip
	locleft=np.argmin(toflip[:,0])
	locright=np.argmin(toflip[:,0])
	avflipy=(toflip[locleft,1]+toflip[locright,1])/2
	#Subtract the minimum in y
	#centeredarray=toflip-[0,toflip[np.argmin(toflip[:,0])][0]]

	#Seperate and flip the negative values
	topvalues=toflip[toflip[:,1]>avflipy]
	bottomvalues=toflip[toflip[:,1]<avflipy]*[1,-1]+[0,2*avflipy]
	return np.concatenate([topvalues,bottomvalues])


def angledet(x1,y1,x2,y2):
	'''
	determine angle needed to rotate to get line horizontal from an x and y point
	'''
	dx=x2-x1
	dy=y2-y1
	return np.arctan(dy/dx)

def linedet(MultipleEdges,ylims=False):
	'''
	Takes a python array of edge xy locations and returns a list of endpoints for use with fitting
	ylims is if there is a need to only search in a certain location
	'''
	#Create empty arrays
	numIm=len(MultipleEdges)
	leftxy=np.zeros([numIm,2])
	rightxy=np.zeros([numIm,2])
	#Subtract background if needed and select image, droplet should be high so invert
	if  (not isinstance(ylims, (list, tuple, np.ndarray)) ) and ylims == False:
		for i in range(numIm):
			#Get the indexes of the minimum and maximum x points
			#Can be modified to extract some other property from each of the xy arrays (ie other than argmin)
			leftindex = np.argmin(MultipleEdges[i][:,0])
			rightindex = np.argmax(MultipleEdges[i][:,0])
			#Get the actual values
			leftxy[i] = [MultipleEdges[i][leftindex,0], MultipleEdges[i][leftindex,1]]
			rightxy[i] = [MultipleEdges[i][rightindex,0], MultipleEdges[i][rightindex,1]]
	else:
		#Define center and buffer range
		ycen=(ylims[0]+ylims[1])/2
		yran=np.abs(ylims[1]-ylims[0])/2
		for i in range(numIm):
			#Select a slice in y to analyze
			cond = np.abs(MultipleEdges[i][:,1]-ycen) < yran #the condition to limit to the yrange of interest
			lefttrimidx = np.where(cond)[0] #Get the indices with the condition
			leftindex = lefttrimidx[MultipleEdges[i][:,0][lefttrimidx].argmin()] #Find the argument of the minimum in that region
			rightindex = lefttrimidx[MultipleEdges[i][:,0][lefttrimidx].argmax()] #Same for the right
			#Get the actual values
			leftxy[i] = [MultipleEdges[i][leftindex,0], MultipleEdges[i][leftindex,1]]
			rightxy[i] = [MultipleEdges[i][rightindex,0], MultipleEdges[i][rightindex,1]]

	return leftxy, rightxy

def thetdet(edgestack,ylims=False):
	'''Find the angle to rotate a stack of images based on the location of droplet edges
	Returns the angle and a point with which to rotate'''
	leftlineinfo, rightlineinfo = linedet(edgestack,ylims)
	#Contact points needed for rotation, actually return the ones from the datafitter
	allcontactpts=np.concatenate([leftlineinfo,rightlineinfo])

	#Fit a line to the contact points 
	fitlineparam,firlinecov = curve_fit(linfx,allcontactpts[:,0],allcontactpts[:,1])

	#create 2 random points based on the line for the angle detection function
	leftedge=[0,linfx(0,*fitlineparam)]
	rightedge=[1,linfx(1,*fitlineparam)]
	#Find the angle with horizontal
	thet=angledet(*leftedge,*rightedge)
	return thet, leftedge

def edgestoproperties(edgestack,lims,fitfunc,fitguess,ylims=False):
	'''
    Takes a edgestack and returns a list of angles for the right and left 
    positions and angles
    edgestack is a python list of numpy arrays containing the edges
    lims is the x and y pixel range to use for fitting
    fitfunc is the function to use to find the angle
	fitguesss is the guess for those parameters
	ylims is optional for when need to select a specific y region because the pipette is farther than the droplet
    '''
	#Create arrays to store data
	numEd=len(edgestack)
	dropangle = np.zeros([numEd,2])
	contactpts = np.zeros([numEd,2,2])
	paramlist=np.zeros([numEd,2,len(fitguess)])

	thetatorotate, leftedge = thetdet(edgestack,ylims)
	rotateprop = [thetatorotate,leftedge]

	for i in range(numEd):
		rotatededges=rotator(edgestack[i],-thetatorotate,*leftedge)
		combovals=xflipandcombine(rotatededges)
		fitl=datafitter(combovals,True,lims,1,fitfunc,fitguess)
		fitr=datafitter(combovals,False,lims,1,fitfunc,fitguess)
		dropangle[i] = [fitl[2],fitr[2]]
		contactpts[i] = [[fitl[0],fitl[1]],[fitr[0],fitr[1]]]
		paramlist[i] = [fitl[-2],fitr[-2]]
	return dropangle, contactpts, paramlist, rotateprop

