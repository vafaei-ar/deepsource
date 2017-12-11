import os
import numpy as np
from skimage import draw
from skimage import measure
from util import fetch_data

def standard(X):
	xmin = X.min()
	X = X-xmin
	xmax = X.max()
	X = X/xmax
	return X

def cent_of_mass(d,filt):
    indx = np.where(filt)[0]
    indy = np.where(filt)[1]
    li = 0
    lj = 0
    mass = 0
    for i,j in zip(indx,indy):
        val = d[i,j]
        mass += val
        li += val*i
        lj += val*j
    li /= mass
    lj /= mass
    return np.round(np.array([li,lj])).astype(int)

def ps_blob_detect(xp,loc_det,jump_lim=50,area_lim=10,threshold_0=1.,return_area=False,verbose=False):
	xp = xp-xp.min() 
	xp = xp/xp.max()
#2: 15, 20
#3: 30,10
#4: 50, 10
	nnp = 0
	for tr in np.exp(np.linspace(np.log(threshold_0),np.log(1e-3),500)):
		blobs = measure.label(xp>tr)
		nn = np.unique(blobs).shape[0]
		if verbose:
			print 'Threshold:',tr,', # of PS:',nnp
		if nn-nnp>jump_lim:
				break
		nnp = nn
		trsh = tr

	blobs = measure.label(xp>trsh)
	xl = []
	yl = []
	al = []
	for v in np.unique(blobs)[1:]:
		filt = blobs==v


		if loc_det=='mean':
			pnt = np.round(np.mean(np.argwhere(filt),axis=0)).astype(int)
		if loc_det=='peak':
			pnt = np.array([np.where(xp==np.max(xp[filt]))[0][0],np.where(xp==np.max(xp[filt]))[1][0]]).astype(int)
		if loc_det=='com':
			pnt = cent_of_mass(xp,filt)


		if filt.sum()>area_lim:
			xl.append(pnt[1])
			yl.append(pnt[0])
			if return_area:
				al.append(np.mean(xp[blobs==v]))
	if return_area:
		return np.array([xl,yl]).T,np.array(al)
	else:
		return np.array([xl,yl]).T

def ps_extract(image_file,model_file,cnn,fetch_func,loc_det,ignore_border=600,jump_lim=50,
               area_lim=10,threshold_0=1.,verbose=False):
    data, x_coords, y_coords = fetch_func(image_file,model_file)

    # Removing borders
    data = data[:,ignore_border:-ignore_border,ignore_border:-ignore_border,:]

    demand_image = cnn.conv_large_image(data,pad=10,lw=400)

    pred = ps_blob_detect(demand_image,loc_det=loc_det,jump_lim=jump_lim,area_lim=area_lim,
                              threshold_0=threshold_0,verbose=verbose)
    print pred.shape[0],' point sources are found!'

    x_coords, y_coords = pred[:,0].astype(int),pred[:,1].astype(int)

    pix_val = []
    prb_val = []
    for i,j in zip(x_coords, y_coords):
        pix_val.append(data[0,j,i,0])
        prb_val.append(demand_image[j,i])

    x_coords, y_coords = pred[:,0]+ignore_border,pred[:,1]+ignore_border

    return np.stack((x_coords, y_coords, np.array(pix_val), np.array(prb_val)), axis=-1)

def visualize_cross_match(image_file,model_file,catalog,border=10):
    import matplotlib.pylab as plt
#    from matplotlib import gridspec
    data, x_coords, y_coords = fetch_data(image_file,model_file)
    plt.plot(x_coords,y_coords,ls='none', markeredgecolor='red',marker='o',markersize=7,
             mew=1, markerfacecolor='none',label='Truth')

    crop_filt = (catalog[:,0]>x_coords.min()-border) & (catalog[:,0]<x_coords.max()+border) \
                    & (catalog[:,1]>y_coords.min()-border) & (catalog[:,1]<y_coords.max()+border)

    print (~crop_filt).sum(),' points are in noisy borders!'
    catalog_crop = catalog[crop_filt]
    plt.plot(catalog_crop[:,0],catalog_crop[:,1],ls='none',color='b',marker='x',markersize=5, 
            mew=1, markerfacecolor=None,label='pred.')

    plt.xlim(x_coords.min()-border,x_coords.max()+border)
    plt.ylim(y_coords.min()-border,y_coords.max()+border)
    plt.xticks([])
    plt.yticks([])
    plt.title(str(catalog_crop.shape[0])+' out of 300')
    plt.legend(bbox_to_anchor=(1.3, 1.025))

