import numpy as np
import random
import glob
from astropy.io import fits
from astropy import wcs, coordinates
from astropy import units as u
from skimage import draw
from util import *

def normalizer(data):
	data = data-data.min()
	data = data/data.max()
	return data

class DataProvider(object):
    
	def __init__(self,files_path,nx,ny, margin=1000):
		
		self.files_path = files_path

		self.margin = margin
		self.nx,self.ny = nx,ny

		self.files = glob.glob(files_path)
		self.n_files = len(self.files)
		self.path = "/".join(files_path.split('/')[:-1])+"/"
		print 'number of files: ',self.n_files

	def __call__(self, n=1 , r0=1, r_sc=0, coord=False):
		margin = self.margin
		i = np.random.randint(self.n_files)

		with fits.open(self.files[i]) as hdulist:
		  data = normalizer(hdulist[0].data)
		  header = hdulist[0].header
		  lx = header['NAXIS1']
		  ly = header['NAXIS2']
		  coord_sys = wcs.WCS(header)

		model_file = self.path+self.files[i].split('/')[-1].split('_')[1]+'.txt'
		sources = np.loadtxt(model_file, dtype={'names': ('name', 'ra', 'dec', 'I'),
		                                'formats': ('S10', 'f4', 'f4', 'f4')})
		source_names, ra, dec, intensity = sources['name'],sources['ra'],sources['dec'],sources['I']
		num_sources = len(source_names)
		radec_coords = coordinates.SkyCoord(ra, dec, unit='deg', frame='fk5')
		coords_ar = np.vstack([radec_coords.ra*u.deg, radec_coords.dec*u.deg,
		                       np.zeros(num_sources), np.zeros(num_sources)]).T
		xy_coords = coord_sys.wcs_world2pix(coords_ar, 0)
		x_coords, y_coords = xy_coords[:,0], xy_coords[:,1]

		intensity = intensity-intensity.min()
		intensity = intensity/intensity.max()

		lx,ly = data.shape[2],data.shape[3]

		cat = np.zeros(data.shape)
		for i,j,k in zip(x_coords.astype(int), y_coords.astype(int),intensity):
				rr, cc = draw.circle(j, i, radius=int(r_sc*k)+r0, shape=cat.shape[-2:])
				cat[0,0, rr, cc] = 1

		X = np.zeros((n, self.nx, self.ny, 1))
		Y = np.zeros((n, self.nx, self.ny, 1))
		if coord:
			crd = [[] for i in range(n)]
		
		i0,j0 = np.random.randint(margin,lx-self.nx-margin),np.random.randint(margin,ly-self.ny-margin)
		X[0,:,:,0] = data[0,0,i0:i0+self.nx,j0:j0+self.ny]
		Y[0,:,:,0] = cat[0,0,i0:i0+self.nx,j0:j0+self.ny]
		if coord:
			for i,j in zip(y_coords.astype(int), x_coords.astype(int)):
				if i0 <= i < i0+self.nx and j0 <= j < j0+self.ny:
					crd[0].append([i-i0,j-j0])

		for k in range(1, n):
			i0,j0 = np.random.randint(margin,lx-self.nx-margin),np.random.randint(margin,ly-self.ny-margin)
			X[k,:,:,0] = data[0,0,i0:i0+self.nx,j0:j0+self.ny]
			Y[k,:,:,0] = cat[0,0,i0:i0+self.nx,j0:j0+self.ny]
			if coord:
				for i,j in zip(y_coords.astype(int), x_coords.astype(int)):
					if i0 <= i < i0+self.nx and j0 <= j < j0+self.ny:
						crd[k].append([i-i0,j-j0])

		if coord:
			return X, Y, np.array(crd[0])
		else:
			return X, Y

def augment(img):
    # ROTATION
    k = np.random.randint(4)
    img = np.rot90(img, k)
    
    # FLIPPING
    pf = np.random.uniform(0,1)
    if (pf>0.5):
        img = np.fliplr(img)
    return img


def slide_batch(x,y,hlp,n_ps=100,n_vs=100,n_bg=200,shuffle=True):
	n_p,lx,ly,n_ch = x.shape

	assert n_p==1,'Unsupported number of patch! It should be 1. '

	n_sample = n_ps+n_vs+n_bg
	batch_x = np.zeros((n_sample,2*hlp+1,2*hlp+1,n_ch))
	batch_y = np.zeros((n_sample,2))  

	ymap = y[0,:,:,0]
	ymap = magnifier(ymap,radius=4,value=1)+magnifier(ymap,radius=35,value=1)
#	if (y[0,:,:,0]==1).sum()<1:
#		return None,None

	lst = np.argwhere(ymap==2)
	ps_lst = lst[(lst[:,0]>2*hlp) & (lst[:,0]<lx-2*hlp) & (lst[:,1]>2*hlp) & (lst[:,1]<ly-2*hlp)]
	if len(ps_lst)<2:
		return None,None

	lst = np.argwhere(ymap==1)
	vc_lst = lst[(lst[:,0]>hlp) & (lst[:,0]<lx-hlp) & (lst[:,1]>hlp) & (lst[:,1]<ly-hlp)]

	lst = np.argwhere(ymap==0)
	bg_lst = lst[(lst[:,0]>hlp) & (lst[:,0]<lx-hlp) & (lst[:,1]>hlp) & (lst[:,1]<ly-hlp)]

	i_d = 0
# Point sources
	for nb in range(n_ps):
		rx,ry = random.choice(ps_lst)
		patch = x[0,rx-hlp:rx+hlp+1,ry-hlp:ry+hlp+1,:]
		batch_x[i_d,:,:,:] = augment(patch)
		batch_y[i_d,1] = 1
		i_d = i_d+1			    

# Vicinity
	for nb in range(n_vs):
		rx,ry = random.choice(vc_lst)
		patch = x[0,rx-hlp:rx+hlp+1,ry-hlp:ry+hlp+1,:]
		batch_x[i_d,:,:,:] = patch
		batch_y[i_d,0] = 1
		i_d = i_d+1			    

# Background
	for nb in range(n_bg):
		rx,ry = random.choice(bg_lst)
		patch = x[0,rx-hlp:rx+hlp+1,ry-hlp:ry+hlp+1,:]
		batch_x[i_d,:,:,:] = patch
		batch_y[i_d,0] = 1
		i_d = i_d+1			    

	if shuffle:
		sh_ind = np.arange(n_sample)
		random.shuffle(sh_ind)
		batch_x = batch_x[sh_ind,:,:,:]
		batch_y = batch_y[sh_ind,:]

	return batch_x,batch_y

#class PreProcessDataProvider(object):
#    
#	def __init__(self,files_path,nx,ny, margin=1000):
#		
#		self.files_path = files_path

#		self.margin = margin
#		self.nx,self.ny = nx,ny

#		self.files = glob.glob(files_path)
#		self.n_files = len(self.files)
#		self.path = "/".join(files_path.split('/')[:-1])+"/"
#		print 'number of files: ',self.n_files

#	def __call__(self, n=1 , radius=10, coord=False):
#		margin = self.margin
#		i = np.random.randint(self.n_files)

#		with fits.open(self.files[i]) as hdulist:
#		  data = normalizer(hdulist[0].data)
#		  header = hdulist[0].header
#		  lx = header['NAXIS1']
#		  ly = header['NAXIS2']
#		  coord_sys = wcs.WCS(header)

#		model_file = self.path+self.files[i].split('/')[-1].split('_')[1]+'.txt'
#		sources = np.loadtxt(model_file, dtype={'names': ('name', 'ra', 'dec', 'I'),
#		                                'formats': ('S10', 'f4', 'f4', 'f4')})
#		source_names, ra, dec, intensity = sources['name'],sources['ra'],sources['dec'],sources['I']
#		num_sources = len(source_names)
#		radec_coords = coordinates.SkyCoord(ra, dec, unit='deg', frame='fk5')
#		coords_ar = np.vstack([radec_coords.ra*u.deg, radec_coords.dec*u.deg,
#		                       np.zeros(num_sources), np.zeros(num_sources)]).T
#		xy_coords = coord_sys.wcs_world2pix(coords_ar, 0)
#		x_coords, y_coords = xy_coords[:,0], xy_coords[:,1]

#		intensity = intensity-intensity.min()
#		intensity = intensity/intensity.max()

#		lx,ly = data.shape[2],data.shape[3]

#		cat = np.zeros(data.shape)
#		for i,j,k in zip(x_coords.astype(int), y_coords.astype(int),intensity):
#				cat[0,0, j, i] = 1

#		cat[0,0, :, :] = horn_kernel(cat[0,0, :, :],radius=radius)
#		cat[0,0, :, :] = gaussian_kernel(cat[0,0, :, :],sigma=radius)

#		X = np.zeros((n, self.nx, self.ny, 1))
#		Y = np.zeros((n, self.nx, self.ny, 1))
#		if coord:
#			crd = [[] for i in range(n)]
#		
#		i0,j0 = np.random.randint(margin,lx-self.nx-margin),np.random.randint(margin,ly-self.ny-margin)
#		X[0,:,:,0] = data[0,0,i0:i0+self.nx,j0:j0+self.ny]
#		Y[0,:,:,0] = cat[0,0,i0:i0+self.nx,j0:j0+self.ny]
#		if coord:
#			for i,j in zip(y_coords.astype(int), x_coords.astype(int)):
#				if i0 <= i < i0+self.nx and j0 <= j < j0+self.ny:
#					crd[0].append([i-i0,j-j0])

#		for k in range(1, n):
#			i0,j0 = np.random.randint(margin,lx-self.nx-margin),np.random.randint(margin,ly-self.ny-margin)
#			X[k,:,:,0] = data[0,0,i0:i0+self.nx,j0:j0+self.ny]
#			Y[k,:,:,0] = cat[0,0,i0:i0+self.nx,j0:j0+self.ny]
#			if coord:
#				for i,j in zip(y_coords.astype(int), x_coords.astype(int)):
#					if i0 <= i < i0+self.nx and j0 <= j < j0+self.ny:
#						crd[k].append([i-i0,j-j0])

#		Y = Y-Y.min()
#		Y = ((Y/Y.max()*10)+1)*X
#		Y = Y-Y.min()
#		Y = Y/Y.max()
#		Y = Y*(X.max()-X.min())
#		Y = Y+X.min()

#		if coord:
#			return X, Y, np.array(crd[0])
#		else:
#			return X, Y
