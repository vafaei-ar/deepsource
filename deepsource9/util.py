import os
import numpy as np
from skimage import draw
from skimage import measure
from astropy.io import fits
from astropy import units as u
from astropy import wcs, coordinates
from scipy.ndimage.filters import gaussian_filter

def standard(X):
	"""
	standard : This function makes data ragbe between 0 and 1.
	
	Arguments:
		X (numoy array) : input data.
	
	--------
	Returns:
		standard data.
	"""
	xmin = X.min()
	X = X-xmin
	xmax = X.max()
	X = X/xmax
	return X

def fetch_data(image_file,model_file,do_standard=True):
    
	"""
	fetch_data : This function reads image and model.
	
	Arguments:
		image_file (string) : path to image file. 
		model_file (string) : path to model file. 
		do_standard (logical) (default=True) : if true, minimum/maximum value of image will be set to 0/1.
		
	--------
	Returns:
		image, x coordinates, y coordinates 
	"""
    
	with fits.open(image_file) as hdulist:
		  data = hdulist[0].data
		  header = hdulist[0].header
		  lx = header['NAXIS1']
		  ly = header['NAXIS2']
		  coord_sys = wcs.WCS(header)

	model_file = model_file
	sources = np.loadtxt(model_file, dtype={'names': ('name', 'ra', 'dec', 'I'),
		                              'formats': ('S10', 'f4', 'f4', 'f4')})
	ra, dec = sources['ra'],sources['dec']
	num_sources = len(ra)
	radec_coords = coordinates.SkyCoord(ra, dec, unit='deg', frame='fk5')
	coords_ar = np.vstack([radec_coords.ra*u.deg, radec_coords.dec*u.deg,
		                     np.zeros(num_sources), np.zeros(num_sources)]).T
	xy_coords = coord_sys.wcs_world2pix(coords_ar, 0)
	x_coords, y_coords = xy_coords[:,0], xy_coords[:,1]

	if do_standard==True:
		data = standard(data)

	return np.moveaxis(data, 0, -1), x_coords, y_coords

def fetch_data_3ch(image_file,model_file,do_standard=True):
    
	"""
	fetch_data_3ch : This function reads 3 images of 3 robust and model.
	
	Arguments:
		image_file (string) : path to robust 0 image file. 
		model_file (string) : path to model file. 
		do_standard (logical) (default=True) : if true, minimum/maximum value of image will be set to 0/1.
		
	--------
	Returns:
		image, x coordinates, y coordinates 
	"""

	data0, x_coords, y_coords = fetch_data(image_file,model_file,do_standard=do_standard)
#	lx,ly = data0[0,:,:,0].shape

	try:
		  data1, x_coords, y_coords = fetch_data(image_file.replace('robust-0','robust-1'),model_file,do_standard=do_standard)
	except:
		assert 0,'Robust 1 does not exist.'
		  
	try:
		  data2, x_coords, y_coords = fetch_data(image_file.replace('robust-0','robust-2'),model_file,do_standard=do_standard)
	except:
		assert 0,'Robust 1 does not exist.'

	return np.concatenate((data0,data1,data2), axis=-1), x_coords, y_coords

def cat2map(lx,ly,x_coords,y_coords):
	"""
	cat2map : This function converts a catalog to a 0/1 map which are representing background/point source.
	
	Arguments:
		lx (int): number of pixels of the image in first dimension.
		ly (int): number of pixels of the image in second dimension.
		x_coords (numpy array): list of the first dimension of point source positions. 
		y_coords (numpy array): list of the second dimension of point source positions. 
	
	--------
	Returns:
		catalog image as nupmy array.
	"""
	cat = np.zeros((lx,ly))
	for i,j in zip(x_coords.astype(int), y_coords.astype(int)):
		  cat[j, i] = 1
	return cat

def magnifier(y,radius=15,value=1):
	"""
	magnifier (numpy array): This function magnifies any pixel with value one by a given value.
	
	Arguments:
		y : input 2D map.
		radius (int) (default=15) : radius of magnification.
		value (float) (default=True) : the value you want to use in magnified pixels.
		
	--------
	Returns:
		image with magnified objects as numpy array.
	"""
	mag = np.zeros(y.shape)
	for i,j in np.argwhere(y==1):
		  rr, cc = draw.circle(i, j, radius=radius, shape=mag.shape)
		  mag[rr, cc] = value
	return mag

def circle(y,radius=15):
	"""
	circle : This function add some circles around any pixel with value one.

	Arguments:
		y (numpy array): input 2D map.
		radius (int) (default=15): circle radius.
		
	--------
	Returns:
		image with circles around objects.
	"""
	mag = np.zeros(y.shape)
	for i,j in np.argwhere(y==1):
		  rr, cc = draw.circle_perimeter(i, j, radius=radius, shape=mag.shape)
		  mag[rr, cc] = 1
	return mag

def horn_kernel(y,radius=10,step_height=1):
	"""
	horn_kernel : Horn shape kernel.
	
	Arguments:
		y (numpy array): input 2D map.
		radius (int) (default=15): effective radius of kernel.
	
	--------
	Returns:
		kerneled image.
	"""
	mag = np.zeros(y.shape)
	for r in range(1,radius):
		for i,j in np.argwhere(y==1):
				rr, cc = draw.circle(i, j, radius=r, shape=mag.shape)
				mag[rr, cc] += 1.*step_height/radius
	return mag

def gaussian_kernel(y,sigma=7):
	"""
	gaussian_kernel: Gaussian filter.
	
	Arguments:
		y (numpy array): input 2D map.
		sigma (float) (default=7): effective length of Gaussian smoothing.
		
	--------
	Returns:
		kerneled image.
	"""
	return gaussian_filter(y, sigma)

def ch_mkdir(directory):
	"""
	ch_mkdir : This function creates a directory if it does not exist.
	
	Arguments:
		directory (string): Path to the directory.
		
	--------
	Returns:
		null.		
	"""
	if not os.path.exists(directory):
		  os.makedirs(directory)

#def ps_extract(xp):
#	xp = xp-xp.min() 
#	xp = xp/xp.max()

#	nb = []
#	for trsh in np.linspace(0,0.2,200):
#		  blobs = measure.label(xp>trsh)
#		  nn = np.unique(blobs).shape[0]
#		  nb.append(nn)
#	nb = np.array(nb)
#	nb = np.diff(nb)
#	trshs = np.linspace(0,0.2,200)[:-1]
#	thrsl = trshs[~((-5<nb) & (nb<5))]
#	if thrsl.shape[0]==0:
#		trsh = 0.1
#	else:
#		trsh = thrsl[-1]
#2: 15, 20
#3: 30,10
#4: 50, 10
#	nnp = 0
#	for tr in np.linspace(1,0,1000):
#		blobs = measure.label(xp>tr)
#		nn = np.unique(blobs).shape[0]
#		if nn-nnp>50:
#				break
#		nnp = nn
#		trsh = tr

#	blobs = measure.label(xp>trsh)
#	xl = []
#	yl = []
#	pl = []
#	for v in np.unique(blobs)[1:]:
#		filt = blobs==v
#		pnt = np.round(np.mean(np.argwhere(filt),axis=0)).astype(int)
#		if filt.sum()>10:
#			xl.append(pnt[1])
#			yl.append(pnt[0])
#			pl.append(np.mean(xp[blobs==v]))
#	return np.array([xl,yl]).T,np.array(pl)
