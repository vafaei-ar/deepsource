import numpy as np
import pywt
import cv2
from astropy.stats import sigma_clip
from astropy.convolution import convolve, AiryDisk2DKernel,Box2DKernel,Gaussian2DKernel,MexicanHat2DKernel
from astropy.convolution import Ring2DKernel,Tophat2DKernel,TrapezoidDisk2DKernel
import myroutines as myr
from util import standard


kernels = [AiryDisk2DKernel(3),Box2DKernel(5),Gaussian2DKernel(2),Gaussian2DKernel(4),MexicanHat2DKernel(2)
           ,MexicanHat2DKernel(4),Tophat2DKernel(2),TrapezoidDisk2DKernel(2),Ring2DKernel(7,2)]
kernel_names = ['AiryDisk3','Box5','Gaussian2','Gaussian4',\
'MexicanHat2','MexicanHat4','Tophat2','TrapezoidDisk2','Ring']

#wts = ['db38','sym20','coif17','bior2.8','bior3.9',\
#'bior4.4','bior5.5','bior6.8','dmey','rbio1.5',\
#'rbio2.8','rbio6.8']
wts = ['db38','sym20','coif17','dmey']

def wavelet(data, wlf, threshold):
	"""
	wavelet: this function .
	
	Arguments:
		data (numpy array): input data.
		wlf: wavelet fucntion.
		threshold: threshold of high pass filter.
		
	--------
	Returns:
		filtered data.	
	"""
	wavelet = pywt.Wavelet(wlf)
	levels  = pywt.dwt_max_level(data.shape[0], wavelet)
	WaveletCoeffs = pywt.wavedec2(data, wavelet, level=levels)
	NewWaveletCoeffs = map (lambda x: pywt.threshold(x,threshold,'greater'),WaveletCoeffs)
	data = pywt.waverec2( NewWaveletCoeffs, wavelet)
	return data

def preproc(raw,return_names=False,funcs=[]):
	"""
	preproc: this function .
	
	Arguments:
		raw (numpy array): input data.
		return_names (logical) (default=False): If True, the function returns a list of filter names.
		funcs (list) (default=[]): list of the functions you want to add as channels to the raw data.
		
	--------
	Returns:
		preprocessed data.	
	"""
	d = np.expand_dims(raw, axis=2)
	names = ['Data']
	for i,func in enumerate(funcs):
		pr = func(raw) 
		pr = np.expand_dims(pr, axis=2)
		d = np.concatenate((d,pr), axis=2)
		names.append('func'+str(i))
#	pr = sigma_clip(raw, sigma=3., axis=0)
#	pr = np.expand_dims(pr, axis=2)
#	d = np.concatenate((d,pr), axis=2)
#	names.append('sigma clip')

	for i,krnl in enumerate(kernels):
#		  pr = standard(raw)
		  pr = convolve(raw, krnl)
		  pr = np.expand_dims(pr, axis=2)
		  d = np.concatenate((d,pr), axis=2)
	names += kernel_names

#	kernel = np.ones((5,5),np.uint8)

#	for i in range(3):
#		  pr = cv2.erode(raw ,kernel,iterations = i)
#		  pr = np.expand_dims(pr, axis=2)
#		  d = np.concatenate((d,pr), axis=2)
#		  names.append('erode'+str(i))

#	for i in range(3):
#		  pr = cv2.dilate(raw ,kernel,iterations = i)
#		  pr = np.expand_dims(pr, axis=2)
#		  d = np.concatenate((d,pr), axis=2)
#		  names.append('dilate'+str(i))

#	morph_names = [attr for attr in dir(cv2) if not callable(getattr(cv2, attr)) and attr.startswith("MORPH")]
#	morph_names.remove('MORPH_HITMISS')
#	morphs = []
#	for m in morph_names:
#		  exec("morphs.append(cv2.%s)" % (m))
#	for m in morphs:
#		  pr = cv2.morphologyEx(raw, m, kernel)
#		  pr = np.expand_dims(pr, axis=2)
#		  d = np.concatenate((d,pr), axis=2)    
#	names += morph_names

	for wt in wts:
		  pr = wavelet(raw, wt, 0.0)
		  pr = np.expand_dims(pr, axis=2)
		  d = np.concatenate((d,pr), axis=2)  
	names += wts

	for c in range(1,7):
		  pr = myr.curvelet(raw,6,c,30,0)
		  pr = np.expand_dims(pr, axis=2)
		  d = np.concatenate((d,pr), axis=2)  
		  names.append('C'+str(c))   
		  
	if return_names:
		  return d,names
	else:
		  return d

def preprocess(X,n_ch=None,funcs=[]):
	"""
	preprocess: this function .
	
	Arguments:
		X (numpy array): input data.
		n_ch (int) (default=None): number of channels (this argument can be set automatically if you leave it None, but if you set it, it can be run faster.).
		funcs (list) (default=[]): list of the functions you want to add as channels to the raw data.
		
	--------
	Returns:
		filtered data.	
	"""

	n_patch = X.shape[0]
	n_pix = X.shape[1]

	if n_ch is None:
		n_ch = preproc(X[0,:,:,0],funcs=funcs).shape[-1]

	Xo = np.zeros((n_patch,n_pix,n_pix,n_ch))

	for i in range(n_patch):
		Xo[i] = preproc(X[i,:,:,0],return_names=False,funcs=funcs)

	return Xo





