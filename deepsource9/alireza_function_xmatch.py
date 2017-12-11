import os
import logging
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from astropy.io import fits
from astropy import wcs, coordinates
from astropy import units as u


def get_file_paths(root_dir, valid_extensions):
    """
    Function to retrieve a list of file paths of files contained in a given root directory.
    """
    file_paths = []
    file_names = []
    skip, read = 0, 0
    for fn in os.listdir(root_dir):
        name, ext = os.path.splitext(fn)
        if ext.lower() not in valid_extensions:
            skip += 1
            continue
        else:
            read += 1
            file_paths.append(os.path.join(root_dir, fn))
            file_names.append(name)
    if read > 0:
        logging.info("Found {} files in {} (skipping {}).".format(read, root_dir, skip))
    else:
        raise Exception("Supplied directory doesn't contain any files that are of the formats: {}".format(valid_extensions))
    return file_paths, file_names


def get_matching_model_path(image_name, model_paths, model_names):
    """
    Function to find the matching model file name for a given image.
    """
    unique_name = image_name.split("_")[1]
    assert len(unique_name) > 0, "Problem with name format of {}".format(image_name)
    assert 'image' in unique_name, "Problem with name format of {}".format(image_name)
    model_file_path = [fp for (fp, fn) in zip(model_paths, model_names) if unique_name == fn]
    assert len(model_file_path) == 1, "Multiple model files found for image {}".format(image_name)
    return model_file_path[0]


def get_radec_skycoords(ra, dec):
    """
    Function to get ra-dec skycoords
    """
    return coordinates.SkyCoord(ra, dec, unit='deg', frame='fk5')


def get_source_coords(fits_header, model_path):
    """
    Function to retrieve the ra-dec and x-y coordinates of sources for a given fits image and
    its corresponding model file.
    fits_header : header of the fits file obtained by fits.open(path_to_fits_image)[0].header
    model_path : full path to the model file corresponding to fits_header
    """
    coord_sys = wcs.WCS(fits_header)
    model_df = pd.read_csv(model_path, sep=" ", skiprows=1, usecols=[0,1,2,3], header=None)
    source_id = model_df[[0]].values.reshape(-1)
    ra = model_df[[1]].values.reshape(-1)
    dec = model_df[[2]].values.reshape(-1)
    flux = model_df[[3]].values.reshape(-1)
    num_sources = len(ra)
    ra_dec_skycoords = get_radec_skycoords(ra, dec)
    coords_ar = np.vstack([ra_dec_skycoords.ra*u.deg, ra_dec_skycoords.dec*u.deg,
                           np.zeros(num_sources), np.zeros(num_sources)]).T
    xy_coords = coord_sys.wcs_world2pix(coords_ar, 0)
    x_coords, y_coords = xy_coords[:,0], xy_coords[:,1]
    return ra, dec, x_coords, y_coords, source_id, flux


def get_xy_coords(fits_header, ra, dec):
    """
    Function to obtain the x-y coordinates given a list of ra-dec coordinates and 
    corresponding image path.
    fits_header : header of the fits file obtained by fits.open(path_to_fits_image)[0].header
    ra : list of ra's
    dec : list of dec's
    """
    coord_sys = wcs.WCS(fits_header)
    num_sources = len(ra)
    ra_dec_skycoords = get_radec_skycoords(ra, dec)
    coords_ar = np.vstack([ra_dec_skycoords.ra, ra_dec_skycoords.dec,
                           np.zeros(num_sources), np.zeros(num_sources)]).T
    xy_coords = coord_sys.wcs_world2pix(coords_ar, 0)
    ra_deg = ra
    dec_deg = dec
    x_coords, y_coords = xy_coords[:,0], xy_coords[:,1]
    return x_coords, y_coords


def get_radec_coords(fits_header, x, y):
    """
    Function to obtain the ra-dec coordinates given a list of x-y coordinates and 
    corresponding image path.
    fits_header : header of the fits file obtained by fits.open(path_to_fits_image)[0].header
    x : list of x coordinates
    y : list of y coordinates
    """
    coord_sys = wcs.WCS(fits_header)
    num_sources = len(x)
    coords_ar = np.vstack([x, y, np.zeros(num_sources), np.zeros(num_sources)]).T
    radec_coords = coord_sys.wcs_pix2world(coords_ar, 0)
    ra_deg = radec_coords[:,0]
    dec_deg = radec_coords[:,1]
    ra_dec = [[item1, item2] for (item1, item2) in zip(ra_deg, dec_deg)]
    return ra_deg, dec_deg


def get_pix_flux_radec(fits_path, ra, dec):
    """
    Function to get the center pixel flux and 9-pixel averaged flux around the center pixel.
    fits_path : full path to fits image
    ra : list of ra coordinates of sources
    dec : list of dec coordinates of sources
    """
    
    assert len(ra) == len(dec), "Error:  lengths of ra and dec are not the same"
    header = fits.open(fits_path)[0].header
    image_data = fits.open(fits_path)[0].data[0,0,:,:]
    coord_sys = wcs.WCS(header)
    ra_dec = [[item1, item2] for (item1, item2) in zip(ra, dec)]
    ra_dec_coords = coordinates.SkyCoord(ra_dec, unit=(u.deg, u.deg), frame='fk5')
    num_sources = len(ra)
    coords_ar = np.vstack([ra_dec_coords.ra, ra_dec_coords.dec,
                           np.zeros(num_sources), np.zeros(num_sources)]).T
    xy_coords = coord_sys.wcs_world2pix(coords_ar, 0)
    x = xy_coords[:,0]
    y = xy_coords[:,1]
    center_pix = []
    avg9_pix = []

    for xval, yval in zip(x, y):
        xval = int(np.round(xval))
        yval = int(np.round(yval))
        center_pix.append(image_data[yval, xval])
        patch = image_data[yval-1:yval+2, xval-1:xval+2]
        avg9_pix.append(np.sum(patch)/9.0)
    
    return np.array(center_pix), np.array(avg9_pix)


def get_separation(ref_radec, target_radec):
    """
    Function to calculate the separation between a reference and target source.
    ref_radec : skycoord ra-dec format of the reference source
    target_radec : skycoord ra-dec format of the target source
    The skycoord format is obtained from astropy's coordinates module as follow:
       my_sky_coord = astropy.coordinates.SkyCoord(ra, dec, unit='deg', frame='fk5')
    """
    ref_target_sep = ref_radec.separation(target_radec)
    return ref_target_sep.arcsecond


def get_noise(file_name, width=100):
    
    '''This function returns the  standard deviation of an image file.
    Parameters
    ----------
    file_name : string
        Address of image file.
    
    Returns
    noise : float
        noise of image.
    *********************
    '''
    
    with fits.open(file_name) as hdulist:
        data = hdulist[0].data
        strip1 = data[0,0,:width,:]
        strip2 = data[0,0,-width:,:]
        strip3 = data[0,0,:,:width].T
        strip4 = data[0,0,:,-width:].T

        strip_t = np.concatenate((strip1,strip2,strip3,strip4),axis=0)
        
    noise = strip_t.std()

    return noise


def get_matches(arcsec_sep, ref_ra, ref_dec, ra, dec):
    """
    Function to get the sources at positions ra dec that lie within arcsec_sep distance 
    from the reference source at ref_ra and ref_dec.
    arcsec_sep : the separation limit (radius of circle) for the crossmatching (in arcseconds)
    ref_ra : single ra for a reference source
    ref_dec : single dec for a reference source
    ref_radec_skycoords : 
    """
    deg_sep = arcsec_sep/3600.0
    ra_high = ref_ra + deg_sep
    ra_low = ref_ra - deg_sep
    dec_high = ref_dec + deg_sep
    dec_low = ref_dec - deg_sep
    ra_filtered_idx = np.where((ra>=ra_low)&(ra<=ra_high))[0]
    dec_filtered_idx = np.where((dec>=dec_low)&(dec<=dec_high))[0]

    neighborhood_idx = list(set(ra_filtered_idx).intersection(set(dec_filtered_idx)))
    ref_radec_skycoords = coordinates.SkyCoord(ref_ra, ref_dec, unit='deg', frame='fk5')
    
    if len(neighborhood_idx) > 0:
        xmatch_gt_idx = []
        sep = []
        for idx in neighborhood_idx:
            radec_skycoords = get_radec_skycoords(ra[idx], dec[idx])
            sep_val = get_separation(ref_radec_skycoords, radec_skycoords)
            if sep_val <= arcsec_sep:
                xmatch_gt_idx.append(idx)
                sep.append(sep_val)
        if len(xmatch_gt_idx) == 0:
            xmatch_gt_idx_final = []
            sep_final = []
        else:
            xmatch_gt_idx_final = xmatch_gt_idx
            sep_final = sep
        return xmatch_gt_idx_final, sep_final
    else:
        return [], []


def do_crossmatch(ra_det, dec_det, x_det, y_det, fits_header,
                  image_path, image_name, model_file_path,
                  sep_limit, noise_val):
    """
    Function to perform crossmatching and calculate completeness and purity curves.
    ra_x_det : List or ra values from some detection algorithm (proposed source positions)
               If input_format flag is set to 'radec', ra_x_det and dec_det are interpreted as
               ra and dec values in degrees.  These values should be floats.  If the 'input_format'
               flag is set to 'xy', ra_x_det and dec_y_det are interpreted as x and y pixel positions,
               respectively, and should be given as integers.
    dec_y_det : Similar to ra_x_det, but for dec and y coordinates
    image_dir : Path to the directory containing the fits images
    image_name : Name (and .fits extension) of the fits image
    model_file_path : Full path to the model file corresponding to the given fits image 'image_name'.
    sep_limit : Separation limit for doing crossmatching between ground truth sources and detection 
                sources (in arcsec).  Radius of circle centered at the ground truth source.
    input_format : Either 'radec' for ra-dec coordinates or 'xy' for x-y coordinates.  ra-dec 
                   coordinates should be specified as float values, while x-y coordinates should
                   be specified as integer values (default is 'radec')
    """

    # assert input_format in ['radec', 'xy'], "Error:  input format type '{}' not supported".format(input_format)
    assert len(ra_det) == len(dec_det)
    assert len(ra_det) >= 1

    if isinstance(ra_det, list):
        ra_det = np.asarray(ra_det)
    if isinstance(dec_det, list):
        dec_det = np.asarray(dec_det)

    print('image_path used:', image_path)
    print('model_file_path used:', model_file_path)
    print('sep limit:', sep_limit)
    print('noise val:', noise_val)

    # get image data and source coordinates in ra-dec and x-y formats
    # gt means: ground truth (actual true sources)
    ra_gt, dec_gt, x_gt, y_gt, gt_ids, _ = get_source_coords(fits_header, model_file_path)
    num_gt = len(ra_gt)
    num_det = len(ra_det)
    print("Number of actual sources: {}".format(num_gt))
    print("Number of detections: {}".format(num_det))

    # get pixel flux values from ground truth and deticted positions
    gt_flux, _ = get_pix_flux_radec(image_path, ra_gt, dec_gt)
    det_flux, _ = get_pix_flux_radec(image_path, ra_det, dec_det)
    
    # TP, FP, FN
    gt_id_tp = []
    gt_flux_tp = []
    gt_x_tp = []
    gt_y_tp =[]
    gt_ra_tp = []
    gt_dec_tp = []
    gt_snr_tp = []
    
    gt_id_fn = []
    gt_flux_fn = []
    gt_x_fn = []
    gt_y_fn = []
    gt_ra_fn = []
    gt_dec_fn = []
    gt_snr_fn = []

    det_id_tp = []
    det_flux_tp = []
    det_x_tp = []
    det_y_tp = []
    det_ra_tp = []
    det_dec_tp = []
    det_snr_tp = []

    det_id_fp = []
    det_flux_fp = []
    det_x_fp = []
    det_y_fp = []
    det_ra_fp = []
    det_dec_fp = []
    det_snr_fp = []

    sep_tp = []
    noise_tp = []
    noise_fp = []
    noise_fn = []
    
    for gt_i in tqdm(np.arange(num_gt), desc="Iterating over ground truth sources"):
        det_match_idx, arcsec_sep = get_matches(sep_limit,
                                                ra_gt[gt_i],
                                                dec_gt[gt_i],
                                                ra_det, dec_det)
        if len(det_match_idx) == 1:
            gt_id_tp.append(gt_ids[gt_i])
            gt_flux_tp.append(gt_flux[gt_i])
            gt_x_tp.append(x_gt[gt_i])
            gt_y_tp.append(y_gt[gt_i])
            gt_ra_tp.append(ra_gt[gt_i])
            gt_dec_tp.append(dec_gt[gt_i])
            gt_snr_tp.append(gt_flux[gt_i]/noise_val)

            det_id_tp.append(det_match_idx[0])
            det_flux_tp.append(det_flux[det_match_idx[0]])
            det_x_tp.append(x_det[det_match_idx[0]])
            det_y_tp.append(y_det[det_match_idx[0]])
            det_ra_tp.append(ra_det[det_match_idx[0]])
            det_dec_tp.append(dec_det[det_match_idx[0]])
            det_snr_tp.append(det_flux[det_match_idx[0]]/noise_val)
            sep_tp.append(arcsec_sep[0])
            noise_tp.append(noise_val)

        elif len(det_match_idx) > 1:
            closest_sep_idx = np.argmin(arcsec_sep)
            success_flag1 = False
            success_flag2 = False
            for ii, det_match_idx_val in enumerate(det_match_idx):
                if ii == closest_sep_idx:
                    gt_id_tp.append(gt_ids[gt_i])
                    gt_flux_tp.append(gt_flux[gt_i])
                    gt_x_tp.append(x_gt[gt_i])
                    gt_y_tp.append(y_gt[gt_i])
                    gt_ra_tp.append(ra_gt[gt_i])
                    gt_dec_tp.append(dec_gt[gt_i])
                    gt_snr_tp.append(gt_flux[gt_i]/noise_val)

                    det_id_tp.append(det_match_idx_val)
                    det_flux_tp.append(det_flux[det_match_idx_val])
                    det_x_tp.append(x_det[det_match_idx_val])
                    det_y_tp.append(y_det[det_match_idx_val])
                    det_ra_tp.append(ra_det[det_match_idx_val])
                    det_dec_tp.append(dec_det[det_match_idx_val])
                    det_snr_tp.append(det_flux[det_match_idx_val]/noise_val)

                    sep_tp.append(arcsec_sep[ii])
                    noise_tp.append(noise_val)
                    success_flag1 = True
                else:
                    det_id_fp.append(det_match_idx_val)
                    det_flux_fp.append(det_flux[det_match_idx_val])
                    det_x_fp.append(x_det[det_match_idx_val])
                    det_y_fp.append(y_det[det_match_idx_val])
                    det_ra_fp.append(ra_det[det_match_idx_val])
                    det_dec_fp.append(dec_det[det_match_idx_val])
                    det_snr_fp.append(det_flux[det_match_idx_val]/noise_val)
                    noise_fp.append(noise_val)
                    success_flag2 = True
            if not success_flag1:
                raise Exception("Error:  did not get the closest matching detection!")
            if not success_flag2:
                raise Exception("Error:  did not convert tp to fp for multiple detections!")
        elif len(det_match_idx) == 0:
            gt_id_fn.append(gt_ids[gt_i])
            gt_flux_fn.append(gt_flux[gt_i])
            gt_x_fn.append(x_gt[gt_i])
            gt_y_fn.append(y_gt[gt_i])
            gt_ra_fn.append(ra_gt[gt_i])
            gt_dec_fn.append(dec_gt[gt_i])
            gt_snr_fn.append(gt_flux[gt_i]/noise_val)
            noise_fn.append(noise_val)
        else:
            raise Exception("Error here!")

    # add the remaining fp detections (taking care not to count the current fp detections twice)
    det_missed_idx = [xx for xx in np.arange(len(ra_det)) if ((xx not in det_id_tp) and (xx not in det_id_fp))]
    for ii, det_missed_idx_val in enumerate(det_missed_idx):
        det_id_fp.append(det_missed_idx_val)
        det_flux_fp.append(det_flux[det_missed_idx_val])
        det_x_fp.append(x_det[det_missed_idx_val])
        det_y_fp.append(y_det[det_missed_idx_val])
        det_ra_fp.append(ra_det[det_missed_idx_val])
        det_dec_fp.append(dec_det[det_missed_idx_val])
        det_snr_fp.append(det_flux[det_missed_idx_val]/noise_val)
        noise_fp.append(noise_val)

    logging.info("Number of TP (ground truth, detections): {}, {}".format(len(gt_id_tp), len(det_id_tp)))
    logging.info("Number of FP (detections): {}".format(len(det_id_fp)))
    logging.info("Number of FN (ground truth): {}".format(len(gt_id_fn)))

    # construct tp dataframe and save to file
    tp_df = pd.DataFrame(data={'ground_truth_id':gt_id_tp})
    tp_df['ground_truth_flux'] = gt_flux_tp
    tp_df['ground_truth_x'] = gt_x_tp
    tp_df['ground_truth_y'] = gt_y_tp
    tp_df['ground_truth_ra'] = gt_ra_tp
    tp_df['ground_truth_dec'] = gt_dec_tp
    tp_df['ground_truth_snr'] = gt_snr_tp
    
    tp_df['matching_det_id'] = det_id_tp
    tp_df['matching_det_flux'] = det_flux_tp
    tp_df['matching_det_x'] = det_x_tp
    tp_df['matching_det_y'] = det_y_tp
    tp_df['matching_det_ra'] = det_ra_tp
    tp_df['matching_det_dec'] = det_dec_tp
    tp_df['matching_det_snr'] = det_snr_tp

    tp_df['sep'] = sep_tp
    tp_df['noise'] = noise_tp

    # construct fp dataframe and save to file
    fp_df = pd.DataFrame(data={'detection_id':det_id_fp})
    fp_df['detection_flux'] = det_flux_fp
    fp_df['detection_x'] = det_x_fp
    fp_df['detection_y'] = det_y_fp
    fp_df['detection_ra'] = det_ra_fp
    fp_df['detection_dec'] = det_dec_fp
    fp_df['detection_snr'] = det_snr_fp
    fp_df['noise'] = noise_fp
    
    # construct fn dataframe and save to file
    fn_df = pd.DataFrame(data={'ground_truth_id':gt_id_fn})
    fn_df['ground_truth_flux'] = gt_flux_fn
    fn_df['ground_truth_x'] = gt_x_fn
    fn_df['ground_truth_y'] = gt_y_fn
    fn_df['ground_truth_ra'] = gt_ra_fn
    fn_df['ground_truth_dec'] = gt_dec_fn
    fn_df['ground_truth_snr'] = gt_snr_fn
    fn_df['noise'] = noise_fn
    
    # convert lists to np arrays
    det_flux_tp = np.array(det_flux_tp)
    det_flux_fp = np.array(det_flux_fp)
    gt_flux_tp = np.array(gt_flux_tp)
    gt_flux_fn = np.array(gt_flux_fn)
    
    # filter out negative pixel flux values by setting these to 0
    det_flux_tp[det_flux_tp < 0.0] = 0.0
    det_flux_fp[det_flux_fp < 0.0] = 0.0
    gt_flux_tp[gt_flux_tp < 0.0] = 0.0
    gt_flux_fn[gt_flux_fn < 0.0] = 0.0

    # write tp, fp, fn to file if given a catalog_folder_path
    # if catalog_folder_path:
    #     if not (os.path.exists(catalog_folder_path) and os.path.isdir(catalog_folder_path)):
    #         os.makedirs(catalog_folder_path)
    #     fname, ext = os.path.splitext(image_name)
    #     tp_df.to_csv(os.path.join(catalog_folder_path,fname+"_TP.csv"), sep=',', index=False)
    #     fp_df.to_csv(os.path.join(catalog_folder_path,fname+"_FP.csv"), sep=',', index=False)
    #     fn_df.to_csv(os.path.join(catalog_folder_path,fname+"_FN.csv"), sep=',', index=False)

    return tp_df, fp_df, fn_df

def calc_completeness_purity(tp_df, fp_df, fn_df, image_path, robust, noise, quality_threshold_val=0.9):
    """
    Function to calculate completeness and purity (and optionally save it).
    tp_df : dataframe containing true positives 
            (columns:  ground_truth_id, ground_truth_pixel_flux, matching_detector_id, matching_detector_flux)
    fp_df : dataframe containing false positives 
            (columns: detector_id, detector_pixel_flux)
    fn_df : dataframe containing false negatives
            (columns: ground_truth_id, ground_truth_pixel_flux)
    save_path : Default is None.  If given, the completeness, purity and bin centers will be saved to this folder.
                ============================================================================================
                IMPORTANT:  NaN's are encoded with values of -1.  When loading in these values for plotting,
                use df.loc[df.loc[:,'completeness']<0,'completeness'] = np.nan to replace the negative values
                with NaN's.  Do the same for 'purity' column.
                ============================================================================================
    """
    
    # if robust == 0:
    #     bin_min = 0.0#3.36190915107e-08
    #     bin_max = 1.52733127834e-06

    # if robust == 1:
    #     bin_min = 0.0
    #     bin_max = 1.9101912585e-07

    # if robust == 2:
    #     bin_min = 0.0
    #     bin_max = 1.17020283597e-06
    # bin_min = 0.0
    # bin_max = 1.17020283597e-06
    
    bin_min = 0.0
    bin_max = 2e-5

    # construct bins and bin arrays
    bin_width = 1.0e-8
    bins = np.arange(bin_min, bin_max+bin_width, bin_width)
    bins_center = bins[0:-1] + (bins[1:]-bins[0:-1])/2.

    det_flux_tp = tp_df['matching_det_flux'].values
    det_flux_fp = fp_df['detection_flux'].values
    gt_flux_tp = tp_df['ground_truth_flux'].values
    gt_flux_fn = fn_df['ground_truth_flux'].values

    # calculate histograms
    det_flux_tp_bins, _ = np.histogram(det_flux_tp, bins=bins)
    det_flux_fp_bins, _ = np.histogram(det_flux_fp, bins=bins)
    gt_flux_tp_bins, _ = np.histogram(gt_flux_tp, bins=bins)
    gt_flux_fn_bins, _ = np.histogram(gt_flux_fn, bins=bins)

    # calculate purity & completeness
    purity_bins = det_flux_tp_bins/(det_flux_tp_bins + det_flux_fp_bins)
    completeness_bins = gt_flux_tp_bins/(gt_flux_tp_bins + gt_flux_fn_bins)
    # completeness_bins = det_flux_tp_bins/(det_flux_tp_bins + gt_flux_fn_bins)

    # count sources above quality threshold value
    purity_quality, _ = trs_find(bins_center/noise, purity_bins, trs=quality_threshold_val, dx=1e-2, dx_min=1e-5)
    completeness_quality, _ = trs_find(bins_center/noise, completeness_bins, trs=quality_threshold_val, dx=1e-2, dx_min=1e-5)
    quality_threshold = max(purity_quality, completeness_quality)
    number_of_sources = (det_flux_fp > quality_threshold*noise).sum() + (det_flux_tp > quality_threshold*noise).sum()
    
    # if save_path:
    #     bins_center2 = np.copy(bins_center)/noise
    #     cp_df = pd.DataFrame(data={#'snr_bin_edges':bins/noise,
    #                                'snr_bin_centers':bins_center2,
    #                                'completeness':completeness_bins,
    #                                'purity':purity_bins})
    #     cp_df.fillna('-1', inplace=True)
    #     cp_df.to_csv(save_path, sep=',', index=False)
        
    return bins/noise, bins_center/noise, purity_bins, completeness_bins, number_of_sources


def trs_find(x,y,trs,dx=1e-2,dx_min=1e-5):
    """
    trs_find : Threshold finder.
    
    x : numpy array of x values.
    y : numpy axis of y values.
    x0 : initial value which is supposed to get decreased until approaching the largest root less than this initial value.
    trsh : the threshold.
    dx : initial step.
    dx_min : accuracy.
    """
    non_nan_idx = np.where(~np.isnan(y))
    x = x[non_nan_idx]
    y = y[non_nan_idx]
    x0 = np.nanmax(x)
    f = interp1d(x,y, kind='linear')
    
    if (np.nanmin(y) > trs):
        return np.nanmin(x), y[x==np.nanmin(x)]
    if (np.nanmax(y) < trs):
        return np.nanmax(x), y[x==np.nanmax(x)]

    while dx > dx_min:
        x0 = x0-dx
        if (x0 <= np.nanmin(x)):
            return np.nanmin(x), y[x==np.nanmin(x)]
        if f(x0)<trs:
            x0 = x0+dx
            dx = dx/2.
    return x0,f(x0)

def alireza_function(image_path, model_path, catalog_path, robust, robust_scale_factor=1.0, q_thresh=0.9, detector='pybdsf'):
    """
    image_path : full path to fits image
    model_path : full path to model file
    catalog_path : full path to pybdsf ra-dec detection catalog, or cnn x-y detection catalog
    robust : robust value (0, 1 or 2)
    robust_scale_factor : value to scale crossmatching radius with
    q_thresh : quality threshold value above which to count sources (default is 0.9)
    detector : type of detector used ('pybdsf' or 'cnn').  Default is 'pybdsf'
    """

    robust = int(robust)
    assert robust in [0,1,2], "Error with robust value given!"
    
    print('-------------------------------------------------------------')
    print('Processing image: {}'.format(image_path))

    # get image name only
    if image_path[-1] == '/':
        image_name = image_path.split('/')[-2]
    else:
        image_name = image_path.split('/')[-1]

    # get header from image_path
    fits_header = fits.open(image_path)[0].header

    # read detections from catalog_path and convert to ra-dec or x-y
    if detector == 'pybdsf':
        pybdsf_det_df = pd.read_csv(catalog_path, header=None, skiprows=6)
        ra_det = list(pybdsf_det_df.ix[:,2].values)
        dec_det = list(pybdsf_det_df.ix[:,4].values)
        x_det, y_det = get_xy_coords(fits_header, ra_det, dec_det)
    elif detector == 'cnn':
        cnn_det_df = pd.read_csv(catalog_path, header=None, skiprows=0, sep=" ")
        x_det = [int(v) for v in cnn_det_df.ix[:,0].values]
        y_det = [int(v) for v in cnn_det_df.ix[:,1].values]
        ra_det, dec_det = get_radec_coords(fits_header, x_det, y_det)
    else:
        raise Exception("Error with detector flag!")

    # sep_limit:  9.0 (robust 0);  2.5 (robust 1);  2.0 (robust 2)f
    if robust == 0:
        sep_limit = 9.0*robust_scale_factor #arcsec
    if robust == 1:
        sep_limit = 2.5*robust_scale_factor #arcsec
    if robust == 2:
        sep_limit = 2.0*robust_scale_factor #arcsec

    # get noise
    noise_val = get_noise(image_path)

    # do crossmatching
    tp_df, fp_df, fn_df = do_crossmatch(ra_det, dec_det,
                                        x_det, y_det,
                                        fits_header,
                                        image_path,
                                        image_name,
                                        model_path,
                                        sep_limit,
                                        noise_val)

    # calculate completeness & purity
    snr_edges, snr_centers, purity, completeness, quality_counts = calc_completeness_purity(tp_df,
                                                                                            fp_df,
                                                                                            fn_df,
                                                                                            image_path,
                                                                                            robust,
                                                                                            noise_val,
                                                                                            quality_threshold_val=q_thresh)
    
    out_dict = {'tp':tp_df, 'fp':fp_df, 'fn':fn_df,
                'snr_centers':snr_centers,
                'purity':purity, 'completeness':completeness,
                'quality_counts':quality_counts}
    
    return out_dict


image_path = "/media/etiennev/E49E73079E72D192/data/different_field_25sept/fits_robust2/stimela-image_image161_25SEP_robust-2-image.fits"
model_path = "/media/etiennev/E49E73079E72D192/data/different_field_25sept/model_files/image161.txt"
# catalog_path_pybdsf = "/media/etiennev/E49E73079E72D192/data/different_field_25sept/bdsf_2sig-isl_2sig-peak_robust2/stimela-image_image101_25SEP_robust-2-image.csv"
catalog_path_cnn = "/media/etiennev/E49E73079E72D192/data/different_field_25sept/alireza_results/cnn2_robust2/image-161_rb-2.txt"
robust_val = 2
scale_factor = 3.0

output_dict = alireza_function(image_path, model_path, catalog_path_cnn, robust_val, robust_scale_factor=scale_factor, q_thresh=0.9, detector='cnn')
# print(output_dict)

