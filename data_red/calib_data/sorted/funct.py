import numpy as np
import funct as polfun
from astropy.io import fits
from itertools import product as carthprod
import shutil
import os

from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.path import Path
from scipy import interpolate
from scipy.stats import poisson
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset





#################### FUNCTIONS #################################################
#################### Generic functions ####################



# Returns a list will all subdirectories in a specific folder as well as the
# contained files
def mk_lsts(dir_name):
    
    #dir_lst = [dirs[0] for dirs in os.walk(dir_name)]
    dir_lst = next(os.walk(dir_name))[1]
    file_lst = next(os.walk(dir_name))[2]
    return np.array(dir_lst), np.array(file_lst)



# Checks whether fname is a fits or reg file and extracts the corresonding header and data or coordinates and radius respectively
def extract_data(fname):

    # Put data and header files into two separate lists
    if fname.endswith("fit") or fname.endswith("fits"):
        hdul = fits.open(fname)
        header = hdul[0].header             
        data = hdul[0].data
        hdul.close()
        
    return header, data
    


# Function which creates a directory
def createdir(direc, replace=False):  
    if not os.path.exists(direc):
        os.makedirs(direc)
    elif (os.path.exists(direc) and replace):
        shutil.rmtree(direc)
        os.makedirs(direc)    



#################### END GENERIC FUNCTIONS ####################





#################### FUNCTIONS FOR FLUX LIST COMPUTATIONS ####################



# Calculates normalized flux/count differences between the ordinary and extraordinary target spectra or fluxes as specified on p.39 of the FORS2 user manual. The corresponding standard deviation is also computed. N.B.: The fluxes must have been derived for the same angle of the retarder waveplate!
def fluxdiff_norm(f_or, f_ex, sigma_or, sigma_ex):
    
    # Compute normalized flux difference
    flux_sum = f_or + f_ex
    fluxdiff_norm = (f_or - f_ex) / flux_sum
    
    # Compute standard deviation
    temp_or = f_ex**2 / flux_sum**4
    temp_ex = f_or**2 / flux_sum**4
    sigma_F = 2 * np.sqrt(temp_or * sigma_or**2 + temp_ex * sigma_ex**2)
        
    return fluxdiff_norm, sigma_F
    
    
    
# Finds a star's center in the neighbourhood of a certain pixel by iterating through a region surrounding this approximate pixel and storing the pixel containing the highest count rate
def find_center(coord, data_array, window_size):
    
    # Initiate center
    center = coord
    # Read data_array shape and the aproximate pixel's x- and y-coordinates
    aprox_x, aprox_y = coord[0], coord[1]
    Ny, Nx = data_array.shape
    
    # Define window corners
    xmin = max(1, aprox_x - window_size)
    xmax = min(Nx, aprox_x + window_size)
    ymin = max(1, aprox_y - window_size)
    ymax = min(Ny, aprox_y + window_size)
    
    # Iterate through region and find highest pixel value
    countmax = 0
    for x in np.arange(xmin, xmax, 1):
        for y in np.arange(ymin, ymax, 1):
            
            pix_val = data_array[y,x]
            if pix_val >= countmax:
                countmax = pix_val
                center = [x,y]
                
    return np.array(center)
                
            
    
# Function which calculates the aperture count rate for a star centered at pixel coordinates [px, py] for an aperture radius r (sky annulus subtraction not included).
def apersum_old(image, px, py, r, absthreshold=np.inf):
    
    
    # Determine the aperture limits
    ny, nx = image.shape
    apx_min = max(1, px - r)
    apx_max = min(nx, px + r)
    apy_min = max(1, py - r)
    apy_max = min(ny, py + r)
    
    
    # Compute the total count rate within the aperture
    pixvals, apsum, apcount = [], 0.0, 0
    for i in range(apx_min, apx_max+1):
        for j in range(apy_min, apy_max+1):
            
            # Calculate squared distance to central pixel
            dx = i - px
            dy = j - py
            d2 = dx**2 + dy**2
            
            # Store the current pixel's count value
            pixval = image[j-1,i-1]
            # Ommit nan and infinite values
            if np.isnan(pixval) or np.isinf(pixval) or abs(pixval) > absthreshold:
                #print("In apersum_old: inf or nan pixel encountered!")
                continue
            
            # Append pixval to pixvals for calculation of median
            pixvals.append(pixval)
            
            # Add to aperture sum
            if d2 <= r**2:
                apsum += pixval
                apcount += 1      
    
    # Determine the mean and the median inclusiding Poissonian error estimators
    mean, med = apsum/apcount, np.median(pixvals)
    mean_err, med_err = np.sqrt(np.abs(mean)/apcount), 1/(4 * apcount * poisson.pmf(med, mean)**2)
    
    #print("DEBIG [apsum, apcount, mean]:\t\t", apsum, apcount, mean)
    #print("DEBUG [med, mean, poisson(med,mean)]:\t\t", med, mean, poisson.pmf(med,mean))
    
    return [apsum, mean, med], [np.sqrt(apsum), mean_err, med_err]
    
    

# Function which calculates the aperture count rate for a star centered at pixel coordinates [px, py] for an aperture radius r. The parameters minAn_r and maxAn_r define the inner and outer radii of the sky annulus.
def apersum(image, px, py, r, minAn_r, maxAn_r):
    
    
    # Determine the shape of the array
    ny, nx = image.shape
    # Determine the aperture limits
    x_min, x_max = max(1, px - maxAn_r), min(nx, px + maxAn_r)
    y_min, y_max = max(1, py - maxAn_r), min(ny, py + maxAn_r)

    
    # Compute the total count rate within the aperture and annulus
    [apsum, ansum, ansum2] = np.zeros(3) # [ADU]
    [apcount, ancount] = np.zeros(2, dtype=int)
    for i in range(x_min, x_max+1):
        for j in range(y_min, y_max+1):
            
            # Calculate squared distance to central pixel
            dx = i - px
            dy = j - py
            d2 = dx**2 + dy**2
            
            # Store the current pixel's count value
            pixval = image[j-1,i-1]
            # Filter out nan values
            if np.isnan(pixval) or np.isinf(pixval):
                continue
            
            # Add pixel value
            if d2 <= r**2:
                apsum += pixval
                # Count number of pixels in aperture
                apcount += 1
            elif (d2 >= minAn_r**2) and (d2 <= maxAn_r**2):
                ansum += pixval
                ansum2 += pixval**2
                # Count number of pixels in annulus
                ancount += 1
        
    
    # Estimate the standard deviation of the number of counts within the annulus
    av_an, av_an2 = ansum/float(ancount), ansum2/float(ancount)
    sigma_an = np.sqrt(av_an2 - av_an**2)
    
    
    # Iterate through the annulus several more times, in order to rule out the possibility of including any outliers
    itnos = 20
    for n in range(itnos):
        
        # Check whether annulus pixel values are within 2 sigma reach of the average annulus rate
        [ansum, ansum2] = np.zeros(2) # [ADU]
        ancount = 0
        for i in range(x_min, x_max+1):
            for j in range(y_min, y_max+1):
    
                # Calculate squared distance to central pixel
                dx = i - px
                dy = j - py
                d2 = dx**2 + dy**2
                
                # Store the current pixel's count value
                pixval = image[j-1,i-1]
                # Skip nan and inf values
                if np.isnan(pixval) or np.isinf(pixval):
                    continue
                
                if ( ((d2 >= minAn_r**2) and (d2 <= maxAn_r**2)) 
                                         and (abs(pixval - av_an) <= 2.*sigma_an) ):
                    ansum += pixval
                    ansum2 += pixval**2
                    # Count number of pixels in annulus
                    ancount += 1
                
        
        # Reevaluate the standard deviation and the average pixel value within the anulus        
        av_an, av_an2 = ansum/float(ancount), ansum2/float(ancount)
        sigma_an = np.sqrt(av_an2 - av_an**2)
        #print("\n\n\t\t\t\titeration:\t{} \n\t\t\t\t sigma_an:\t{}".format(n, sigma_an))
        #TODO UNCOMMENT
        
    # Compute and return calibrated aperture flux
    apscal = apsum - apcount*av_an          
    return apscal
                       
  
    
# Function for computing the median flux within a certain anulus
def ansum(image, px, py, minAn_r, maxAn_r, itnos=21):

    # Determine the shape of the array
    ny, nx = image.shape
    # Determine the aperture limits
    x_min, x_max = max(1, px - maxAn_r), min(nx, px + maxAn_r)
    y_min, y_max = max(1, py - maxAn_r), min(ny, py + maxAn_r)

    # Iterate through the annulus several more times, in order to rule out the possibility of including any outliers
    av_an, sigma_an = 0, np.inf
    for n in range(itnos):
        
        # Check whether annulus pixel values are within 2 sigma reach of the average annulus rate
        [ansum, ansum2] = np.zeros(2) # [ADU]
        ancount, pixvals_an = 0, []
        for i in range(x_min, x_max+1):
            for j in range(y_min, y_max+1):
    
                # Calculate squared distance to central pixel
                dx = i - px
                dy = j - py
                d2 = dx**2 + dy**2
                
                # Store the current pixel's count value
                pixval = image[j-1,i-1]
                # Skip nan or inf values
                if np.isnan(pixval) or np.isinf(pixval):
                    continue
                # Append anulus pixval to pixvals
                pixvals_an.append(pixval)
                
                if ( ((d2 >= minAn_r**2) and (d2 <= maxAn_r**2)) 
                                         and (abs(pixval - av_an) <= 2.*sigma_an) ):
                    ansum += pixval
                    ansum2 += pixval**2
                    # Count number of pixels in annulus
                    ancount += 1
                
        
        # Reevaluate the standard deviation and the average pixel value within the anulus        
        av_an, av_an2 = ansum/float(ancount), ansum2/float(ancount)
        sigma_an = np.sqrt(av_an2 - av_an**2)
        # print("\n\n\t\t\t\titeration:\t{} \n\t\t\t\t sigma_an:\t{}".format(n, sigma_an))
        #TODO UNCOMMENT
        
    
    # Compute the Poissonian sample mean and median as well as the corresponding errors
    mean, med = av_an, np.median(pixvals_an)
    mean_err, med_err = np.sqrt(np.abs(mean)/ancount), 1/(4 * ancount * poisson.pmf(med, mean)**2)
    
    #print("DEBIG [ansum, ancount, av_an]:\t\t", ansum, ancount, av_an)
    #print("DEBUG [med, mean, poisson(med,mean)]:\t\t", med, mean, poisson.pmf(med,mean))
    
    return [ansum, av_an, np.median(pixvals_an)], [np.sqrt(ansum), mean_err, med_err]
    
    

# Function which computes normalized flux differences as well as the ordinary and extraordinary counts for a preselection regions in various images defined by 'loc_lsts'. 
def compute_fluxlsts(data_dirs, bias, masterflat_norm, loc_lsts, r_range, datasavedir):

    # Compute the linear polarization degrees for each template of exposures taken of Vela1_95 and WD1615_154
    for i, data_dir in enumerate(data_dirs): 
        print("\n\n\n{}".format(data_dir))
        
        
        # Create a list with all the template directories within data_dir
        [tpl_dirlst, tpl_flst] = mk_lsts(data_dir)
        tplnamemask = np.array([(len(tpl_name) < 5) for tpl_name in tpl_dirlst])
        tpl_dirlst = np.concatenate( (np.sort(tpl_dirlst[tplnamemask]), 
                                      np.sort(tpl_dirlst[np.logical_not(tplnamemask)])) )
        tpl_dirlst = np.delete(tpl_dirlst, np.argwhere(tpl_dirlst=="appended"))

        # A 1D lst for storing the exposures containing the least amount of stars per template
        least_expnos = np.zeros(len(tpl_dirlst), dtype=np.int64) 
        
           
        
        # Initiate lists containing the ordinary flux, the extraordinary flux and the normalized flux differences, for each template (axis=0), each exposure file (axis 1), each selected star within the exposure (axis2), and each aperture radius (axis 3)
        O_0lst, sigmaO_0lst = [], []
        E_0lst, sigmaE_0lst = [], []
        F_0lst, sigmaF_0lst = [], []
        # List for storing the filters of each exposure as well as the q-index and xcoordinate of the STD star within each template (axis1) and each exposure (axis2)
        filter_lst = []
        pos0_lst = []
        # Array for storing the images of all templates and exposures using the correct offsets w.r.t. the first exposure of template 1
        frames_jk = np.zeros(len(tpl_dirlst), 4, 1084, 2098)
        
        for j, tpl_name in enumerate(tpl_dirlst):
            print("\n\n\t {}".format(tpl_name))     
            tpl_dir = data_dir + '/' + tpl_name
            
            
            # Create a list with filenames of files stored within tpldir
            expdir_lst, expfile_lst = mk_lsts(tpl_dir)
            expfile_lst = np.sort(expfile_lst)
            
            # Initial setting for the least amount of detected stars within the template
            N_stars = 1e18        
            
            
            # Skip 'appended' subdirectory
            if (tpl_name == "appended") or (len(expfile_lst) != 4):
                print("\t\t skipped")
                continue
            
            
            # Initiate first sublists for distinguishing different exposures
            O_1lst, sigmaO_1lst = [], []
            E_1lst, sigmaE_1lst = [], []
            F_1lst, sigmaF_1lst = [], []
            # Initiate sublist which tracks the positions of each star for each expsoure within template 'j'
            pos1_lst = []
            for k, f in enumerate(expfile_lst):
                print("\n\t\t {}".format(f))
                
                
                # Skip non-fits files
                if not f.endswith(".fits"):
                    print("\t\t\t skipped")
                    continue 
                
                
                header, data = extract_data(tpl_dir + '/' + f)
                # Subtract bias
                datacor1 = data - bias
                datacor2 = datacor1 / masterflat_norm
                # Save corrected image
                savedir = tpl_dir + "/corrected2" 
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                elif os.path.exists(savedir) and k==0:
                    shutil.rmtree(savedir)
                    os.makedirs(savedir)
                hdu = fits.PrimaryHDU(datacor2)
                hdulist = fits.HDUList([hdu])
                hdulist.writeto(savedir + '/' + f.split(".fits")[0] + "_COR.fits")
                
                
                
                # Specify observation parameters
                expno = header["HIERARCH ESO TPL EXPNO"]
                filt_name = header["HIERARCH ESO INS FILT1 NAME"]
                filt_id = header["HIERARCH ESO INS FILT1 ID"]
                ret_angle = header["HIERARCH ESO INS RETA2 POSANG"] * np.pi / 180. #rad
                woll_angle = header["HIERARCH ESO INS WOLL POSANG"] * np.pi / 180. #rad
                print("\t\t\t\tFILTER_ID: {A}; \t FILTER_NAME: {B}".format(A=filt_id, B = filt_name))
                print("\t\t\t\tWollangle: {A}; \t Retangle: {B}".format(A=woll_angle, B = np.round(ret_angle, 2)))                
                
                
                
                # Initiate second sublist of F for distinguishing between different stars within the current exposure
                O_2lst, sigmaO_2lst = [], []
                E_2lst, sigmaE_2lst = [], []
                F_2lst, sigmaF_2lst = [], []
                # List which tracks the center position of the PSF's of all stars within the k-th exposure of template 'j'
                pos2_lst = []
                for q, coord in enumerate(loc_lsts[i]):
                    # Finds the central pixel of the selected stars within the specific exposure                    
                    coord1, coord2, PSFr = coord[0:2], [coord[0],coord[2]], coord[3]
                    center1 = find_center(coord1, data, 15) #TODO TODO NOTE: NO SKY APERTURES!
                    center2 = find_center(coord2, data, 15)
                    centers = [center1, center2]
                    
                    
                    # Initiate third sublist of F for distinguishing between different aperture radii
                    O_3lst, sigmaO_3lst = [], []
                    E_3lst, sigmaE_3lst = [], []
                    F_3lst, sigmaF_3lst = [], [] 
                    for l, R in enumerate(r_range):
                    
                        # Lists for temporary storage of aperture sum values and corresponding shotnoise levels
                        apsum_lst, shotnoise_lst = [], []
                        for center in centers:
                            
                            # Define sky annulus inner and outer radii\
                            if R < PSFr:
                                minRan, maxRan = int(PSFr), int(1.5*PSFr)
                            else:
                                minRan, maxRan = int(R), int(1.5*R)
                            # Compute cumulative counts within aperture
                            apsum = apersum(data, center[0], center[1], 
                                            R, minRan, maxRan) #TODO TODO TODO UNCOMMENT
                            #apsum = apersum(data, center[0], center[1], R)
                            apsum_lst.append(apsum)
                            # Compute photon shot noise within aperture 
                            shotnoise = np.sqrt(apsum)
                            shotnoise_lst.append(shotnoise)
                        
                        # Compute normalised flux differences for current aperture size
                        F, sigmaF = fluxdiff_norm(apsum_lst[1], apsum_lst[0], 
                                                  shotnoise_lst[1], shotnoise_lst[0]) 
                        
                        
                        
                        # Append results to third sublist
                        O_3lst.append(apsum_lst[1]), sigmaO_3lst.append(shotnoise_lst[1])
                        E_3lst.append(apsum_lst[0]), sigmaE_3lst.append(shotnoise_lst[0])
                        F_3lst.append(F), sigmaF_3lst.append(sigmaF)
                    # Append the third sublist to the second sublist
                    O_2lst.append(O_3lst), sigmaO_2lst.append(sigmaO_3lst)
                    E_2lst.append(E_3lst), sigmaE_2lst.append(sigmaE_3lst)
                    F_2lst.append(F_3lst), sigmaF_2lst.append(sigmaF_3lst)
                    # Append centers to pos2_lst
                    pos2_lst.append(centers) 
                # Append second sublist to first sublist
                O_1lst.append(O_2lst), sigmaO_1lst.append(sigmaO_2lst)
                E_1lst.append(E_2lst), sigmaE_1lst.append(sigmaE_2lst)
                F_1lst.append(F_2lst), sigmaF_1lst.append(sigmaF_2lst)
                # Append pos2_lst to pos1_lst
                pos1_lst.append(pos2_lst)
            # Append first sublist to main list 
            O_0lst.append(O_1lst), sigmaO_0lst.append(sigmaO_1lst)
            E_0lst.append(E_1lst), sigmaE_0lst.append(sigmaE_1lst)
            F_0lst.append(F_1lst), sigmaF_0lst.append(sigmaF_1lst)      
            # Append filter name to filter_lst and append pos1_lst to pos0_lst
            filter_lst.append(filt_name), pos0_lst.append(pos1_lst)
        # Transform into arrays for future computations
        O_0lst, sigmaO_0lst = np.array(O_0lst), np.array(sigmaO_0lst)
        E_0lst, sigmaE_0lst = np.array(E_0lst), np.array(sigmaE_0lst)
        F_0lst, sigmaF_0lst = np.array(F_0lst), np.array(sigmaF_0lst) 
        filter_lst, pos0_lst = np.array(filter_lst), np.array(pos0_lst)
        
        # Save the flux arrays
        savedir = datasavedir +'/'+ data_dir.rsplit("/",2)[1] + "/fluxlsts"
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        np.save(savedir + "/O_0lst.npy", O_0lst), np.save(savedir + "/sigmaO_0lst.npy", sigmaO_0lst)
        np.save(savedir + "/E_0lst.npy", E_0lst), np.save(savedir + "/sigmaE_0lst.npy", sigmaE_0lst)
        np.save(savedir + "/F_0lst.npy", F_0lst), np.save(savedir + "/sigmaF_0lst.npy", sigmaF_0lst)
        # Save filt_lst and pos0_lst
        np.save(savedir + "/filter_lst.npy", filter_lst)
        np.save(savedir + "/pos0_lst.npy", pos0_lst)
# END COMPUTE_FLUXLSTS



#################### END FUNCTIONS FOR FLUX LIST COMPUTATIONS ####################





#################### FUNCTIONS FOR COMPUTING POLARIZATION ####################



# Function for loading the lists specifying the ordinary, extraordinary fluxes and the normalized flux differences for all templates, exposures, selected regions and aperture radii, as well as for loading the list specifying the used filter used for each template. 'Loaddir' specifies the directory where the list structures are stored.
def load_lsts(loaddir):
    # Store current directory
    currdir = os.getcwd()
    
    # Move to directory 
    os.chdir(loaddir)
    O_jkqr, sigmaO_jkqr = np.load("O_0lst.npy"), np.load("sigmaO_0lst.npy")
    E_jkqr, sigmaE_jkqr = np.load("E_0lst.npy"), np.load("sigmaE_0lst.npy")
    F_jkqr, sigmaF_jkqr = np.load("F_0lst.npy"), np.load("sigmaF_0lst.npy")
    
    # Load filter_lst and pos0_lst
    filter_lst = np.load("filter_lst.npy")
    pos0_lst = np.load("pos0_lst.npy")
    os.chdir(currdir)
    print("List structures loaded...")
    
    return np.array([O_jkqr, E_jkqr, F_jkqr]), np.array([sigmaO_jkqr, sigmaE_jkqr, sigmaF_jkqr]), filter_lst, pos0_lst



# Compute Stokes Q and U as well as the degree angle of linear polarization and all corresponding error margins for all templates, exposures, selected regions and aperture radii
def compute_pol(F_lst, sigmaF_lst):
    # Compute Stokes variables
    Q_jqr = 0.5 * F_lst[:,0,:,:] - 0.5 * F_lst[:,2,:,:]
    U_jqr = 0.5 * F_lst[:,1,:,:] - 0.5 * F_lst[:,3,:,:]
    
    # Compute standard deviations
    sigmaQ_jqr = 0.5 * np.sqrt(sigmaF_lst[:,0,:,:]**2 + sigmaF_lst[:,2,:,:]**2)
    sigmaU_jqr = 0.5 * np.sqrt(sigmaF_lst[:,1,:,:]**2 + sigmaF_lst[:,3,:,:]**2)
    
    # Compute degree of linear polarization and polarization angle
    fracUQ_jqr = np.divide(U_jqr, Q_jqr) # --
    P_jqr = np.sqrt(Q_jqr**2 + U_jqr**2) # --
    Phi_jqr = 0.5 * np.arctan(fracUQ_jqr) * (180. / np.pi) # deg
    
    # Compute standard deviation on degree of linear polarization
    tempP = np.sqrt( (Q_jqr * sigmaQ_jqr)**2 + (U_jqr * sigmaU_jqr)**2 )
    sigmaP_jqr = np.divide(tempP, P_jqr)
    # Compute standard deviation on angle of linear polarization
    temp1Phi = 1. / (np.ones(fracUQ_jqr.shape) + fracUQ_jqr)
    temp2Phi = temp1Phi * np.sqrt(sigmaU_jqr**2 + fracUQ_jqr**2 * sigmaQ_jqr**2)
    sigmaPhi_jqr = 0.5 * np.divide(temp2Phi, Q_jqr)        
    
    return (np.array([Q_jqr, U_jqr, P_jqr, Phi_jqr]), 
            np.array([sigmaQ_jqr, sigmaU_jqr, sigmaP_jqr, sigmaPhi_jqr]))



# Function for selection the correct aperture radius for each region specified in the 4D list outputs of compute_fluxlsts
def select_r(jkqr_lst, regindex, radindex):
    
    jkq_lst = jkqr_lst[:,:,regindex,radindex]
    
    return jkq_lst 
    
    

# Function for computing arry normalizations over the average along a specific axis 'ax', as well as the corresponding errors
def compute_norm(arr, sigma_arr=None, ax=0):
    # Store the input array shape and define the new array shape for arithmetic purposes
    arrshape = np.array(arr.shape)
    arrnewshape = np.where(arrshape!=arrshape[ax], arrshape, 1)
    
    # Initialize sigma_arr
    if sigma_arr==None:
        sigma_arr = np.zeros(arrshape)
    
    # Compute the normalization factors
    norm = np.average(arr, weights = 1. / sigma_arr**2, axis=ax)
    normerr = np.sqrt(1. / np.sum(1./sigma_arr**2, axis=ax))
    
    # Transform shape for arithmetical purposes, such that the normalization factor is equal for all elements along 'ax'
    norm2, normerr2 = norm.reshape(arrnewshape), normerr.reshape(arrnewshape)
    
    # Normalization of input array and computation of corresponding errors
    arr_norm = arr / norm2
    tempErr = sigma_arr**2 + (arr**2 * normerr2**2) / norm2**2
    arr_normErr = (1./norm2) * np.sqrt(tempErr)
    
    return arr_norm, arr_normErr
    
       
    
    
    
#################### END FUNCTIONS FOR COMPUTING POLARIZATION ####################





#################### PLOT FUNCTIONS ####################



# Two functions for brightening/darkening html hex string colors
def clamp(val, minimum=0, maximum=255):
    if val < minimum:
        return minimum
    if val > maximum:
        return maximum
    return val

def colorscale(hexstr, scalefactor):
    """
    Scales a hex string by ``scalefactor``. Returns scaled hex string.

    To darken the color, use a float value between 0 and 1.
    To brighten the color, use a float value greater than 1.

    >>> colorscale("#DF3C3C", .5)
    #6F1E1E
    >>> colorscale("#52D24F", 1.6)
    #83FF7E
    >>> colorscale("#4F75D2", 1)
    #4F75D2
    """

    hexstr = hexstr.strip('#')

    if scalefactor < 0 or len(hexstr) != 6:
        return hexstr

    r, g, b = int(hexstr[:2], 16), int(hexstr[2:4], 16), int(hexstr[4:], 16)

    r = clamp(r * scalefactor)
    g = clamp(g * scalefactor)
    b = clamp(b * scalefactor)

    return "#%02x%02x%02x" % (r, g, b)



# Function for showing and saving png image
def saveim_png(data, savedir, fname, colmap='coolwarm', 
               orig=None, datextent=None, interp='None',
               scatterpoints=None, scattercol='b', 
               xtag='X', ytag='Y', title=None):
    
    # Show and save image    
    plt.figure()
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.imshow(data, cmap=colmap, origin=orig, 
               norm=norm, interpolation=interp, extent=datextent)
    if scatterpoints != None:
        plt.scatter(scatterpoints[0], scatterpoints[1], c=scattercol)
    plt.colorbar()
    plt.xlabel(xtag, fontsize=20), plt.ylabel(ytag, fontsize=20)
    if not title is None:
        plt.title(title, fontsize=24)
    plt.tight_layout()
    # Create savedir       
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    plt.savefig(savedir + '/' + fname + ".png")
    # plt.show()
    plt.close()



# Show and save offset_arr
def save3Dim_png(Xgrid, Ygrid, Zdata, savedir, fname, dataplttype='scatter',
                 fit=False, fitXgrid=None, fitYgrid=None, fitZdata=None, 
                 fitdataplttype='surface', colour='r', colmap='coolwarm', 
                 rowstride=1, colstride=1, lw=0, xtag='X', ytag='Y', ztag='Z', title=None):
    
    # Initiate plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    # Plot data
    if dataplttype == 'scatter':
        datplt = ax.scatter(Xgrid, Ygrid, Zdata, color=colour)
    elif dataplttype == 'surface':
        datplt = ax.plot_surface(Xgrid, Ygrid, Zdata, cmap=colmap, 
                                 rstride=rowstride, cstride=colstride, 
                                 linewidth=lw, antialiased=False)
        fig.colorbar(datplt, shrink=1.0, aspect=20)
    
    # Plot fitdata
    if fit == True:
        if fitdataplttype == 'scatter':
            fitplt = ax.scatter(fitXgrid, fitYgrid, fitZdata, color=colour)
        elif fitdataplttype == 'surface':
            fitplt = ax.plot_surface(fitXgrid, fitYgrid, fitZdata, cmap=colmap, 
                                     rstride=rowstride, cstride=colstride, 
                                     linewidth=lw, antialiased=False)
            fig.colorbar(fitplt, shrink=1.0, aspect=20)
    
    # Save plot
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    ax.set_xlabel(xtag, fontsize=20), ax.set_ylabel(ytag, fontsize=20)
    ax.set_zlabel(ztag, fontsize=20)
    if not title is None:
        ax.set_title(title, fontsize=24)
    
    plt.savefig(savedir + '/' + fname + ".png")
    # plt.show()
    plt.close()   
    
    

# Function for saving arrays to fits files
def savefits(data, savedir, fname):
        
    # Save to fits files
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    # Prevent saving conflicts
    if os.path.exists(savedir + '/' + fname + ".fits"):
        os.remove(savedir + '/' + fname + ".fits")
    # Save to fits file
    hdu = fits.PrimaryHDU(data)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(savedir + "/{}".format( fname+ ".fits"))
    
    
    
# Function for saving arrays to numpy savefiles
def savenp(data, savedir, fname):
        
    # Save to fits files
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    # Prevent saving conflicts
    if os.path.exists(savedir + '/' + fname + ".npy"):
        os.remove(savedir + '/' + fname + ".npy")
    # Save to numpy file
    np.save(savedir + '/' + fname, data)
    


# Aperture radius vs linear polarization              
def apRvsPl(ax, r_range, pixscale, P, sigmaP, esoP=0., esosigmaP=0., colour='b', tag=None):
    
    # Plot lines defining given polarization degrees
    tempr = np.linspace(0., r_range[-1]*pixscale, 100)
    esoLOW = np.tile(esoP-esosigmaP, len(tempr))
    ax.plot(tempr, esoLOW, color='0.2', linestyle = '--')
    esoHIGH = np.tile(esoP+esosigmaP, len(tempr))
    ax.plot(tempr, esoHIGH, color='0.2', linestyle = '--')
        
     # Plot the computed polarization degrees as function of aperture radius
    ax.errorbar(r_range*pixscale, P, yerr = sigmaP, marker='o', color=colour, label=tag)
    


# Function for plotting a circle
def plot_circle(ax, radius, xc, yc, colour='b', tag=None):

    X = np.linspace(-1.*radius, radius, 1000) + xc
    R = np.tile(radius, 1000)
    Y = np.sqrt(R**2 - X**2) - yc
    
    ax.plot(X, Y, color=colour, label=tag)
    ax.plot(X, -1.*Y, color=colour, label=tag)
    


# Function for plotting a line
def plot_line(ax, x_range, xc, yc, a, colour='b', tag=None):
    
        y_range = a*(x_range - xc) + yc
        
        ax.plot(x_range, y_range, color=colour, label=tag)

    
    
# Function for plotting Q as a function of U. Parameters are as follows:
# - QU:             1D list specifying Q, sigmaQ, U and sigmaU respectively
# - PlPhi:          1D list specifying P, sigmaP, Phi and sigmaPhi respectively
# - colour (opt):   character specifying the colour which should be used within the plot
# - PLPHI (opt):    1D list specifying known P, sigmaP, Phi and sigmaPhi respectively
# - Checks (opt):   Boolean for determining whether to plot the given polarizations
def QvsU(fig, ax, QU, offsetangle=0., PLPHI=np.zeros(4), checkPphi=[False,False], plot_inset=False, inset_ax=None, colour='b', tag=None):

    # Decompose list
    [Q,sigmaQ,U,sigmaU] = QU
    # Offset corrections
    offsetangle = offsetangle * np.pi / 180. #rad
    Qcorr = Q*np.cos(offsetangle) - U*np.sin(offsetangle)
    Ucorr = Q*np.sin(offsetangle) + U*np.cos(offsetangle)
    
    
    # Plot initial QvsU point
    if tag == "b_HIGH" and offsetangle!=0.:
        uncorrtag = "uncorrected"
    else:
        uncorrtag = None
    ax.errorbar(Q, U, xerr = sigmaQ, yerr = sigmaU, fmt='.', markersize = 16., color = "0.3", label=uncorrtag) # STD star
    
    # Plot QvsU point after offset correction
    ax.errorbar(Qcorr, Ucorr, xerr = sigmaQ, yerr = sigmaU, fmt='.', markersize = 16., color = colour, label = tag) # STD star
    
    
    '''
    # Plot polarization degree circles and angle lines
    plotline = np.linspace(-Pl*1.5, Pl*1.5, 100)
    x_plotline = plotline * np.cos(Phi/180.*np.pi)
    ylow_plotline = plotline * np.sin(Phi/180.*np.pi) 
    yhigh_plotline = plotline * np.sin(Phi/180.*np.pi)
    plt.plot(x_plotline, y_plotline, color='k') # Line indicating calculated polarization angle
    '''
    # Plot given polarization degrees and angles
    [checkP, checkphi] = checkPphi
    [PL,sigmaPL,PHI,sigmaPHI] = PLPHI # [-], [deg]
    if checkP == True:
        plot_circle(ax, PL-sigmaPL, xc=0., yc=0., colour=colour) # ESO documentation inner polarization circle
        plot_circle(ax, PL+sigmaPL, xc=0., yc=0., colour=colour) # ESO documentation outer polarization circle
    if checkphi == True:
        PHI, sigmaPHI = PHI/180.*np.pi, sigmaPHI/180.*np.pi # rad
        plotline = np.linspace(-PL*1.5, PL*1.5, 100)                 
        x_plotline2 = plotline * np.cos(PHI)
        ylow_plotline2 = plotline * np.sin(PHI-sigmaPHI) 
        yhigh_plotline2 = plotline * np.sin(PHI+sigmaPHI)
        ax.plot(x_plotline2, ylow_plotline2, color=colour) # ESO doc lower 
        ax.plot(x_plotline2, yhigh_plotline2, color=colour) # ESO doc lower 
        

    if plot_inset == True:
        inset_ax.errorbar(Q, U, xerr = sigmaQ, yerr = sigmaU, fmt='.', markersize = 16., color = '0.3') # STD star
        inset_ax.errorbar(Qcorr, Ucorr, xerr = sigmaQ, yerr = sigmaU, fmt='.', markersize = 16., color = colour) # STD star
        plot_circle(inset_ax, PL-sigmaPL, xc=0., yc=0., colour=colour) # ESO documentation inner polarization circle
        plot_circle(inset_ax, PL+sigmaPL, xc=0., yc=0., colour=colour) # ESO documentation outer polarization circle
        inset_ax.plot(x_plotline2, ylow_plotline2, color=colour)
        inset_ax.plot(x_plotline2, yhigh_plotline2, color=colour)
        inset_ax.axis(xmin=-0.095, xmax=-0.06, ymin=0.0, ymax=0.03)
        # Set inset_ax axes limits
        inset_ax.set_xlim(-0.085, -.065) #TODO REMOVE HARDCODE
        inset_ax.set_ylim(0.001, 0.021) #TODO REMOVE HARDCODE
        # fix the number of ticks on the inset axes
        inset_ax.yaxis.get_major_locator().set_params(nbins=1)
        inset_ax.xaxis.get_major_locator().set_params(nbins=1)
        # draw a bbox of the region of the inset axes in the parent axes and connecting lines between the bbox and the inset axes area
        mark_inset(ax, inset_ax, loc1=2, loc2=4, fc="none", ec="0.7")
        
        
        
# Function for fitting a sine wave
def sine_fit(t, w, phi, amp, b):

    sine = amp*np.sin(w*t + phi) + b
    
    return sine
    
    
    
# Function for fitting a parabola
def parabola_fit(x, a, b):
    
    y = a * x**2 + b
    
    return y
    


# TAKEN FROM http://stackoverflow.com/questions/8850142/matplotlib-overlapping-annotations
def get_text_positions(x_data, y_data, txt_width, txt_height):
    a = list(zip(y_data, x_data))
    text_positions = y_data.copy()
    for index, (y, x) in enumerate(a):
        local_text_positions = [i for i in a if i[0] > (y - txt_height) 
                            and (abs(i[1] - x) < txt_width * 2) and i != (y,x)]
        if local_text_positions:
            sorted_ltp = sorted(local_text_positions)
            if abs(sorted_ltp[0][0] - y) < txt_height: #True == collision
                differ = np.diff(sorted_ltp, axis=0)
                a[index] = (sorted_ltp[-1][0] + txt_height, a[index][1])
                text_positions[index] = sorted_ltp[-1][0] + txt_height
                for k, (j, m) in enumerate(differ):
                    #j is the vertical distance between words
                    if j > txt_height * 2: #if True then room to fit a word in
                        a[index] = (sorted_ltp[k][0] + txt_height, a[index][1])
                        text_positions[index] = sorted_ltp[k][0] + txt_height
                        break
    return text_positions
    
def text_plotter(x_data, y_data, text_positions, axis,txt_width,txt_height):
    i = 0
    for x,y,t in zip(x_data, y_data, text_positions):
        axis.text(x - txt_width, 1.01*t, str(i),rotation=0, color='blue')
        i += 1
        if y != t:
            axis.arrow(x, t,0,y-t, color='red',alpha=0.3, width=txt_width*0.1, 
                       head_width=txt_width, head_length=txt_height*0.5, 
                       zorder=0,length_includes_head=True)   
                       
    


#################### END PLOT FUNCTIONS ####################





#################### FUNCTIONS FOR SLIT APPENDAGE ####################



# Function which cuts each image into approximate slits using the first derivative along Y
def cut_to_slits(slitdata, chipXYranges=[[183,1868],[10,934]]):
    
    # Define the chip edges
    chip_xrange, chip_yrange = chipXYranges
    chipdata = slitdata[chip_yrange[0]:chip_yrange[1], 
                        chip_xrange[0]:chip_xrange[1]]
    rowsno = chipdata.shape[0]
    
    
    # Read out chipdata row for row and cut out slits
    derivs, slits = [], []
    cutend, upedge_lst, lowedge_lst, gapwidths, shapes = 0, [], [], [], []
    itnos = np.arange(1, rowsno-2, 1)
    
    
    # Cut slits out of chipdata
    for i in itnos:
        
        row, next1row, next2row = chipdata[i,:], chipdata[i+1,:], chipdata[i+2,:]    
        rowmed, next1rowmed, next2rowmed = np.median(row), np.median(next1row), np.median(next2row)   
        
        deriv = next1rowmed - rowmed
        nextderiv = next2rowmed - next1rowmed
        derivs.append(deriv)
        
        # Cut out slits and extraordinary and ordinary beams from the data array
        if np.abs(deriv) > 20. and np.abs(nextderiv) < 20.:          
            cutstart = i
        
        if np.abs(deriv) < 20. and np.abs(nextderiv) > 20.:
            prevcutend = cutend
            cutend = i
                       
            # Skips the first peak in the derivatives, so that slit can be cut out correctly
            try:
                slit = chipdata[ cutstart:cutend, : ]                
                if slit.shape[0]>10:
                    slits.append(slit)
                    # Store the slit's upper and lower edge (along y), the slit's width and the gapwidths for calculations of stellar positions on slit in the future
                    upedge_lst.append( cutend + chip_yrange[0] )
                    lowedge_lst.append( cutstart + chip_yrange[0] )
                    gapwidths.append( cutstart - prevcutend ) # [pixels]
                    shapes.append(np.array(slit.shape))
                        
                    '''
                    # Diagnostic plot
                    plt.figure(0)
                    norm = ImageNormalize(stretch=SqrtStretch())
                    plt.imshow(slit, cmap='afmhot', origin='lower', norm=norm)
                    plt.colorbar()
                    plt.show()
                    plt.close()
                    '''
            
            except NameError:
                print("first max")
    
    return slits, upedge_lst, lowedge_lst, gapwidths, np.array(shapes)
    


# Function which embeds a data array in a larger (nrow,ncol)-frame, in order to correctly overlap images. Frameshape should be specified with its first element indicating the number of rows and its second element indicating the number of columns! Conversely, offset should be specified with its first element (X) indicating the column-wise offset and its second element (Y) indicating the row-wise offset! 'Cornerpix' define the pixel indices corresponding to the frame position where the upper left corner of the data array should be placed.
def embed(data, frameshape, offset=np.zeros(2,dtype=int), cornerpix=[0,0]):
    
    # Determine frame limits and create frame accordingly
    nrow, ncol = frameshape[0], frameshape[1]
    frame = np.tile(np.nan, [nrow,ncol])
    
    # Determine data shape
    nrow_dat, ncol_dat = data.shape
    # Embed the data array in the frame, using the specified offset. NOTE: x=columns and y=rows!
    frame[cornerpix[0]+offset[1]:cornerpix[0]+offset[1]+nrow_dat, 
          cornerpix[1]+offset[0]:cornerpix[1]+offset[0]+ncol_dat] = data
    
    return frame
    


# Function for extracting a 2d sub-array from a larger array using a 2d mask
def mask2d(data, mask, fillshape=None, compute_uplcorn=False):
    
    # Create flattened masked array
    idx = mask.sum(1).cumsum()[:-1]
    flatmasked = np.split(data[mask], idx)
    
    # Reshape to form 2d array and find the upper left corner of the cropped region
    lst2d, firstrow = [], None
    for rowno, row in enumerate(flatmasked):
        possrow = list(row)
        if len(possrow) != 0:
            if fillshape is None:
                lst2d.append(possrow)
            # Fill up each row in order to create an array with shape fillshape
            if fillshape != None:
                temprow = list(np.tile(np.nan, fillshape[1]))
                temprow[0:len(possrow)] = possrow
                lst2d.append(temprow)
            # Determine the index of the first row containing a True value
            if compute_uplcorn and (firstrow is None):
                firstrow = rowno

                
    # Determine the index of the first column containing a True value
    for colno, colsum in enumerate(mask.sum(0)):
        if compute_uplcorn and (colsum != 0):
            firstcol = colno
            break
        elif compute_uplcorn == False:
            break
    arr2d = np.array(lst2d)
    
    # Return either the masked array and the upper left corner or solely the masked array
    if compute_uplcorn:
        return arr2d, np.array([firstrow, firstcol])
    else:
        return arr2d
    


# Function which computes O - E for all slits created via cut_to_slits and inserts the results into the array 'frame' in order to recreate a single image of the sky. The parameter 'pixoffs' can be used to specify offsets of E w.r.t. O in x and y. NOTE: X<-->columns, Y<-->rows! 
def align_slits(slits, pixoffs=None, detoffs=False):
    
    # Determine number of slits and slit dimensions
    nrofslits = len(slits)
    # Initiate minimum slit shape parameters
    minNx, minNy = slits[0].shape[1], slits[0].shape[0]
    # Initiate pixoffs
    if pixoffs is None:
        pixoffs= np.zeros((nrofslits/2, 2))
    # print("pixoffs: {}".format(pixoffs)) #TODO Uncomment
    
    
    # Initialize minNx and minNy
    minNx, minNy = np.inf, np.inf
    # Concatenate all slits using the pixoffs parameter
    for n in np.arange(0, nrofslits, 2):
        # Select pixel ofset
        pixoffXY = pixoffs[int(n/2)]
        # Select the O and E slits
        Eslit, Oslit = slits[n], slits[n+1]
        
        
        # Check
        '''
        plt.figure()
        norm = ImageNormalize(stretch=SqrtStretch())
        plt.imshow(Oslit, cmap='afmhot', origin='lower', norm=norm)
        plt.colorbar()
        plt.show()
        plt.close() 
        # Check
        plt.figure()
        norm = ImageNormalize(stretch=SqrtStretch())
        plt.imshow(Eslit, cmap='afmhot', origin='lower', norm=norm)
        plt.colorbar()
        plt.show()
        plt.close() 
        '''
       
        
        
        # Embed the slits in a bigger frame,
        framesize = (2*np.array(Oslit.shape)).astype(int)
        upleft_corner = (0.25*framesize).astype(int)
        frameO = embed(Oslit, framesize, 
                       offset=[0,0], cornerpix=upleft_corner)
        frameE = embed(Eslit, framesize, 
                       offset=pixoffXY, cornerpix=upleft_corner)
        
        
        # Check
        '''
        plt.figure()
        norm = ImageNormalize(stretch=SqrtStretch())
        plt.imshow(frameO, cmap='afmhot', origin='lower', norm=norm)
        plt.colorbar()
        plt.show()
        plt.close() 
        # Check
        plt.figure()
        norm = ImageNormalize(stretch=SqrtStretch())
        plt.imshow(frameE, cmap='afmhot', origin='lower', norm=norm)
        plt.colorbar()
        plt.show()
        plt.close()
        '''
        
        
        # Compute the sum and difference of the frames
        slit_diff = frameO - frameE
        slit_sum = frameO + frameE
        # Compute squared slitdifference for determining offsets or normalized slitdifference for final images
        if detoffs:
            cal_frame = (slit_diff / slit_sum)**2
        else:
            cal_frame = slit_diff / slit_sum

        
        # Check
        '''
        plt.figure()
        norm = ImageNormalize(stretch=SqrtStretch())
        plt.imshow(cal_frame, cmap='afmhot', origin='lower', norm=norm)
        plt.colorbar()
        plt.show()
        plt.close()
        '''
                
        
        # Crop to overlapping region
        overlmask = ~np.isnan(frameO*frameE)
        if detoffs:
            cal_slit, newuplcorn = mask2d(cal_frame, overlmask, compute_uplcorn=True)
            cornshift = upleft_corner - newuplcorn
        else:
            cal_slit = mask2d(cal_frame, overlmask)
        # Determine slit dimensions
        Ny, Nx = cal_slit.shape
        minNx, minNy = min(Nx, minNx), min(Ny, minNy)
        
        
        # Check
        '''
        print("DEBUG: inside align_slits")
        print(frameO)
        print(overlmask)
        print(overlmask.shape, cal_slit.shape)
        plt.figure()
        norm = ImageNormalize(stretch=SqrtStretch())
        plt.imshow(cal_slit, cmap='rainbow', origin='lower', norm=norm)
        plt.colorbar()
        plt.show()
        plt.close()
        '''
        
        
        # Concatenate slits
        if n == 0:
            cal_slits = cal_slit
        else:
            cal_slits = np.concatenate((cal_slits, cal_slit[::,0:minNx]), axis=0)
        
        
        # Reset boolean
        adjust_calslits = False
    
    
    # Remove interslit gaps
    if not detoffs:
        rowmask = np.array( [(np.median(cal_slits[rowno,:])!=-1) 
                              for rowno in range(cal_slits.shape[0])] )
        cal_slits = cal_slits[rowmask,:]
    
    
    # Return the calibrated slits or the calibrated slits and the cornershift
    if detoffs:
        return cal_slits, cornshift
    else:
        return cal_slits



# Function for interpolating a data-array via interpolate.griddata
def interp(data, new_Nx, new_Ny):
    
    # Determine old grid
    Ny, Nx = data.shape
    x, y = np.arange(0, Nx), np.arange(0, Ny)
    xgrid, ygrid = np.meshgrid(x,y)
    points = np.dstack( (xgrid.ravel(),ygrid.ravel()) )[0]
    values = data.ravel()
    
    # Set up new grid
    X, Y = np.linspace(0, Nx, new_Nx), np.linspace(0, Ny, new_Ny)
    Xgrid, Ygrid = np.meshgrid(X,Y)
       
    # Compute interpolation
    interp = interpolate.griddata(points, values, (Xgrid,Ygrid), method='cubic')
    
    return interp
    
    
    
# Function which computes optimal offsets for overlap using the well method
def offsetopt_well(ims, dxrange, dyrange, center, R, 
                   saveims=False, pltsavedir=None, imsavedir=None):
       
    
    # Initialize array to store the normalized flux differences of the evaluated star corresponding to certain offsets in x and y
    offset_arr = np.zeros([len(dyrange),len(dxrange)])
    # Initialize lists
    offset_lst, fitparab_lst, fitgauss_lst = [], [], []    
    # Determine normalized flux differences of star
    for m, dy in enumerate(dyrange):
        for n, dx in enumerate(dxrange):
            
            # Compute nomalized flux for whole image
            interim, cornershift = align_slits(ims, [[dx,dy]], detoffs=True)
            newcent = np.array(center) - cornershift
            
            # Diagnostic plot
            '''
            plt.figure()
            plt.imshow(interim, origin='lower', vmin=0., vmax=0.2)
            plt.colorbar()
            plt.scatter(center[0], center[1], c='k', s=50)
            plt.show()
            plt.close()
            '''
            
            # Determine normalized flux
            [F,_,_], [_,_,_] = apersum_old(interim, center[0], center[1], 
                                           int(np.sqrt(np.max(dxrange)**2 + np.max(dyrange)**2)))
            '''
            if F < 0:
                plt.figure()
                plt.imshow(interim, origin='lower')
                plt.scatter(center[0], center[1], c='k', s=40)
                plt.show()
                plt.close()
            '''# Diagnostic plots
            
            offset_arr[m,n] = F
    offset_lst.append(offset_arr)
    
    
    '''
    plt.figure()
    plt.imshow(offset_arr, origin='lower', vmax=1e5)
    plt.colorbar()
    plt.show()
    plt.close()
    '''# Diagnostic plot
    
    
    # Fit parameter trial input
    y0, x0 = np.unravel_index(offset_arr.argmin(), offset_arr.shape)
    y0, x0 = y0 + dyrange[0], x0 + dxrange[0]
    # Gaussian fit initial fit parameter guesses
    p0_gauss = (x0,y0,np.max(offset_arr),2,2,
                abs(np.max(offset_arr))-abs(np.min(offset_arr)))
                #(x0, y0, z0, sigx, sigy, A)    
    
    
    # Conduct parabolic fit to offset_arr
    fitdata = offset_arr.ravel()
    dxgrid, dygrid = np.meshgrid(dxrange, dyrange)
    popt_gauss, pcov_gauss = curve_fit(gaussian2d, (dxgrid, dygrid), fitdata, p0_gauss)
    print("Absolute 2dGauss min.:\t\t(dx,dy)={}".format(popt_gauss[[0,1]]))
    print("Fit parameter variances:\t\t", np.diag(pcov_gauss))
    
    
    # Warn if the variance of the x0 and y0 becomes too high (i.e. sqrt(sigma_dx**2 + sigma_dy**2 )> 1)
    if np.sqrt( np.sum((np.diag(pcov_gauss)[[0,1]])**2) ) > 1:
        print("High offset variance!!! ---> Closer Inspection Required!")    
    
    
    # Determine the difference between the 2dGaussian model and the flux data
    fitgauss2d = gaussian2d((dxgrid, dygrid), *popt_gauss).reshape(offset_arr.shape)
    modeval_gauss = offset_arr - fitgauss2d
    fitgauss_lst.append(fitgauss2d) 
    
    
    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    fitdat = ax.plot_surface(dxgrid, dygrid, fitgauss2d, cmap='coolwarm', 
                             rstride=1, cstride=1, linewidth=0, antialiased=False)
    pltdat = ax.scatter(dxgrid, dygrid, offset_arr)
    fig.colorbar(fitdat)
    plt.show()
    plt.close()
    '''# Diagnostic plot    
    
    
    
    '''
    var = [5,5,5,10,10,5] # Allowed sigma's for x0, y0, z0, sigx, sigy and A
    if ( np.sum([np.diag(pcov_gauss) >= var[i] for i in range(len(var))]) >= 1 ):
        print("\nPossible problem in fit!\n")
        print("\nVariances:\t", np.diag(pcov_gauss))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        fitdat = ax.plot_surface(dxgrid, dygrid, fitgauss2d, cmap='coolwarm', 
                                 rstride=1, cstride=1, linewidth=0, antialiased=False)
        pltdat = ax.scatter(dxgrid, dygrid, offset_arr)
        fig.colorbar(fitdat)
        plt.show()
        plt.close()
        
        # Try again
        newdxrange = np.arange(x0-(len(dxrange)//2), x0+(len(dxrange)//2))
        newdyrange = np.arange(y0-(len(dyrange)//2), y0+(len(dyrange)//2))
        offsetopt_well(ims, newdxrange, newdyrange, center, R, 
                       saveims=saveims, pltsavedir=pltsavedir, imsavedir=imsavedir)
    '''# Diagnostic plot
    
    
    # Create final slit image using guassian fit
    offsetopt_fitgauss = np.rint(popt_gauss[0:2]).astype(int)
    finalim_gauss = align_slits(ims, np.tile((offsetopt_fitgauss),[len(ims),1]), detoffs=False)
    
    
    
    # PLOTS
    if saveims:
        # Save 2d offset_arr
        saveim_png(offset_arr, pltsavedir+"/", 
                   "offsetwell_optdx{}dy{}".format(offsetopt_fitgauss[0],
                                                offsetopt_fitgauss[1]), 
                   colmap='coolwarm', orig='lower',
                   datextent=[np.min(dxrange), np.max(dxrange), np.min(dyrange), np.max(dyrange)],
                   xtag=r"$\delta_X$", ytag=r"$\delta_Y$")
        # Save offset_arr and paraboloid fit as 3D png
        save3Dim_png(dxgrid, dygrid, offset_arr, pltsavedir, 
                     "offsetwellGauss3D_optdx{}dy{}".format(offsetopt_fitgauss[0],
                                                         offsetopt_fitgauss[1]), 
                     fit=True, fitXgrid=dxgrid, fitYgrid=dygrid, fitZdata=fitgauss2d, 
                     xtag=r"$\delta_X$", ytag=r"$\delta_Y$", ztag=r"$\left( (O-E)/(O+E) \right)^2$")
        
        # Save Gaussian finalim to fits file
        savefits(finalim_gauss, imsavedir, "O-E_dx{}dy{}".format(offsetopt_fitgauss[0],
                                                                 offsetopt_fitgauss[1]))
    
    return(offsetopt_fitgauss, offset_arr, finalim_gauss)
    
    


# Function for fitting a paraboloid to the offset array
def paraboloid(xy, x0=0., y0=0., z0=0, a=1., b=1.):
    
    x, y = xy[0], xy[1]
    z = ((x-x0)/a)**2 + ((y-y0)/b)**2 + z0
    
    return z.ravel()
    
    

# Function for fitting a 2D gaussian to the offset array
def gaussian2d(xy, x0, y0, z0, sigx, sigy, A):
    
    x, y = xy[0], xy[1]
    tempx = (x-x0)**2 / (2 * sigx**2)
    tempy = (y-y0)**2 / (2 * sigy**2)
    z = A*np.exp(-tempx - tempy) + z0
    
    return z.ravel()
    


# Function which computes optimal offsets for overlap using the gradient method
def offsetopt_cd(O, E, crange, drange, center, R, iteration=0, 
                 savetofits=False, pltsavedir=None, imsavedir=None):
    
    # Create the save directories
    createdir(pltsavedir), createdir(imsavedir)
    
    # Determination of c and d (gradient method)
    slitdiff, slitsum = O-E, O+E    
    grady, gradx = np.gradient(O)
    
    
    # Plot gradients and O-E
    if iteration == 0:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
        V_min, V_max = -np.max(np.abs(gradx[~np.isnan(gradx)])), np.max(np.abs(gradx[~np.isnan(gradx)]))
        dat1 = ax1.imshow(gradx, origin='lower', cmap="afmhot", 
                          extent=[-center[0], gradx.shape[1]-center[0], 
                                  -center[1], gradx.shape[0]-center[1]], 
                          vmin=V_min, vmax=V_max)
        dat2 = ax2.imshow(grady, origin='lower', cmap="afmhot", 
                          extent=[-center[0], gradx.shape[1]-center[0], 
                                  -center[1], gradx.shape[0]-center[1]], 
                          vmin=V_min, vmax=V_max)
        dat3 = ax3.imshow(slitdiff, origin='lower', cmap="afmhot", 
                          extent=[-center[0], gradx.shape[1]-center[0], 
                                  -center[1], gradx.shape[0]-center[1]], 
                          vmin=V_min, vmax=V_max)
        plt.colorbar(dat1)
        for ax, title in zip([ax1, ax2, ax3], [r"$\nabla_x (O)$", r"$\nabla_y (O)$", r"$O-E$"]):
            ax.set_xlabel("X [pixel]", fontsize=20)
            if ax == ax1: ax.set_ylabel("Y [pixel]", fontsize=20)
            ax.set_title('{}'.format(title), fontsize=24)
        plt.savefig(pltsavedir+"/grad_O-E.png")
        #plt.show()
        plt.close()
        
        
        # x-gradient profile
        plt.figure()
        #plt.plot(np.arange(0,O.shape[0]), E[:,center[0]], color="y", marker='s', label=r"$E$")
        #plt.plot(np.arange(0,O.shape[0]), O[:,center[0]], color="g", marker='o', label=r"$O$")
        #print(np.max((E-O)[:,center[0]], np.max(gradx[:,center[0]])))
        O_E, Ogradx = (O-E)[center[1],:], gradx[center[1],:]
        normval1 = np.max(np.abs(O_E[~np.isnan(O_E)]))
        normval2 = np.max(np.abs(Ogradx[~np.isnan(Ogradx)]))
        plt.plot(np.arange(0,O_E[~np.isnan(O_E)].shape[0]), 
                 O_E[~np.isnan(O_E)]/normval1, color="r", marker='x', linestyle='--', label=r"$O-E$")
        plt.plot(np.arange(0,Ogradx[~np.isnan(Ogradx)].shape[0]), 
                 Ogradx[~np.isnan(Ogradx)]/normval2, color="b", marker='v', label=r"$\nabla_x (O)$")
        plt.legend(loc='best')
        plt.xlabel("Pixel", fontsize=20), plt.ylabel("Normalized Counts [---]", fontsize=20)
        plt.title("Stellar x-profile", fontsize=20)
        plt.savefig(pltsavedir+"/gradxProfile.png")
        #plt.show()
        plt.close()
        
            
        # y-gradient profile
        plt.figure()
        #plt.plot(np.arange(0,O.shape[0]), E[:,center[0]], color="y", marker='s', label=r"$E$")
        #plt.plot(np.arange(0,O.shape[0]), O[:,center[0]], color="g", marker='o', label=r"$O$")
        #print(np.max((E-O)[:,center[0]], np.max(grady[:,center[0]])))
        O_E, Ogrady = (O-E)[:,center[0]], grady[:,center[0]]
        normval1 = np.max(np.abs(O_E[~np.isnan(O_E)]))
        normval2 = np.max(np.abs(Ogrady[~np.isnan(Ogrady)]))
        plt.plot(np.arange(0,O_E[~np.isnan(O_E)].shape[0]), O_E[~np.isnan(O_E)]/normval1, color="r", marker='x', linestyle='--', label=r"$O-E$")
        plt.plot(np.arange(0,Ogrady[~np.isnan(Ogrady)].shape[0]), 
                 Ogrady[~np.isnan(Ogrady)]/normval2, color="b", marker='v', label=r"$\nabla_y (O)$")
        plt.legend(loc='best')
        plt.xlabel("Pixel", fontsize=20), plt.ylabel("Normalized Counts [---]", fontsize=20)
        plt.title("Stellar y-profile", fontsize=20)
        plt.savefig(pltsavedir+"/gradyProfile.png")
        #plt.show()
        plt.close()
    
    
    
    Qmin_cd, Qmin = np.array([0,0]), np.inf
    for c in crange:
        for d in drange:
            
            temp = (slitdiff - c*gradx - d*grady)**2
            Qlst, Qlst_err = apersum_old(temp, center[0], center[1], R)
            Q = Qlst[0]
            
            '''
            savefits(slitdiff-c*gradx-d*grady, savedir+"/NONABS", 
                     "testc{}d{}".format(np.round([c,d],3)[0], np.round([c,d],3)))
            '''
            
            if Q <= Qmin:
                Qmin, Qmin_cd = Q, np.array([c,d])
                gradoptabs = np.sqrt(temp)
                gradopt = slitdiff - Qmin_cd[0]*gradx - Qmin_cd[1]*grady
    
    if savetofits:
    
        # Diagnostic image
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        dat1 = ax1.imshow(slitdiff, origin='lower', cmap="afmhot")
        plt.colorbar(dat1)
        dat2 = ax2.imshow(gradopt, origin='lower', cmap="afmhot")
        plt.colorbar(dat2)
        #plt.show()
        plt.close()
        
        # Save the gradient corrected slit
        saveim_png(gradopt[center[1]-25:center[1]+25,center[0]-25:center[0]+25], pltsavedir, 
                   "O-E-bg-grad{}_c{}d{}".format(iteration,
                                                 np.round(Qmin_cd[0],3), 
                                                 np.round(Qmin_cd[1],3)),
                   colmap="afmhot", datextent=[-25, 25, -25, 25], 
                   xtag=r"X [pixel]", ytag=r"Y [pixel]")
                   
        savefits(gradopt[~np.isnan(gradopt)], imsavedir, "O-E-bg-grad{}_c{}d{}".format(iteration,
                                                                   np.round(Qmin_cd[0],3), 
                                                                   np.round(Qmin_cd[1],3)))
    
    return (gradopt, Qmin, Qmin_cd)
    
    

# Function for fitting a bivariate polynomial to a dataset z
def polyfit2d(x, y, z, order=4):
    
    # Define an array G in which to store the polynomial variables (power comb. of x and y)
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    
    # Fill G with combinations of powers of x and y
    ij = carthprod(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    
    # Find a least squares fit of G with respect to the data z
    m, _, _, _ = np.linalg.lstsq(G, z)
    
    return m

# Function for evaluating a third order polynomial m at the points x and y
def polyval2d(x, y, m):
    
    # Define the order of the polynomial
    order = int(np.sqrt(len(m))) - 1
    
    # Compute the polynomial
    ij = carthprod(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
        
    return z
    


# Different function for aligning O and E, which shifts each pixel individually
# The two slits and pixoffs are assumed to have the same shape!
#TODO DOESN'T SEEM TO WORK LIKE WE WANT IT TO. USE MEDIAN WHOLE-PIXEL OFFSET OVER ENTIRE SLIT INSTEAD
def align_slits2(slits, pixoffs):
    
    # Extract O and E
    E, O = slits
    slitshape = np.array(O.shape)
    # Extract x- and y-offsets
    pixoffsx, pixoffsy = pixoffs
    #print("DEBUG:\t\t", pixoffsx, pixoffsy)
    
    # Perform pixel-by-pixel offset of E w.r.t. O
    Eoffs = np.tile(np.nan, 2*slitshape)
    lowlcorn = (0.25*slitshape).astype(int)
    for x in range(O.shape[1]):
        for y in range(O.shape[0]):
        
            # Extract current pixel's offsets
            pixeloffx, pixeloffy = pixoffsx[y,x], pixoffsy[y,x]
            if np.isnan(pixeloffx):
                pixeloffx = 0
            if np.isnan(pixeloffy):
                pixeloffy = 0
            
            if x == 301 and y == 21:
                print("DEEEEEEEEEEEBUUUUUUUUUG:\t{},{}".format(pixeloffx, pixeloffy))    
                
            Eoffs[lowlcorn[0]+y+int(pixeloffx), lowlcorn[1]+x+int(pixeloffy)] = E[y,x]
            
    
    
    # Determine the normalized flux difference
    frameO = embed(O, 2*slitshape, cornerpix=lowlcorn)
    
    # Diagnostic plot
    plt.imshow(frameO-Eoffs, origin='lower', cmap='afmhot', vmin=-1, vmax=1)
    plt.colorbar()
    plt.show()
    plt.close()    
    
    slitdiff = frameO - Eoffs
    slitsum = frameO + Eoffs
    cal_frame = slitdiff / slitsum
    
    # Crop to overlapping region
    overlmask = ~np.isnan(frameO*Eoffs)   
    slitdiff_crop =  mask2d(slitdiff, overlmask, fillshape=slitshape)
    slitsum_crop = mask2d(slitsum, overlmask, fillshape=slitshape)
    cal_slit = mask2d(cal_frame, overlmask, fillshape=slitshape)
    
    # Diagnostic plot
    '''
    plt.imshow(cal_slit, origin='lower', cmap='afmhot')
    plt.colorbar()
    #plt.show()
    plt.close()
    '''
    
    return slitdiff_crop, slitsum_crop, cal_slit



# Function for stacking images
def stackim(ims, offs=None):
    
    # Initialize offsets
    if offs is None:
        offs = np.zeros(len(ims), dtype=int)
    
    # Set framesize
    maxNy, maxNx = np.amax([im.shape for im in ims], axis=0)
    framesize = (2*np.array([maxNy,maxNx]).astype(int)
    upleft_corner = (0.25*framesize).astype(int)
    
    # Embed the images in larger arrays
    imembs = []
    for imnr, im in enumerate(ims):
        embed(O, framesize, cornerpix=lowlcorn)
        imemb = embed(im, framesize, offset=offs[imnr], cornerpix=lowlcorn)
        imembs.append(imemb)
    
    # Stack the images
    imstack = np.nanmedian(imembs, axis=0)
    # Cut off nan values
    overlmask = ~np.isnan(np.prod(imembs, axis=0))
    imstack = mask2d(imstack, overlmask)
    
    return imstack
        
    


# Function for aligning a slitpair, using predetermined offset interpolations    
def detslitdiffnorm(slitpair, offs_i, savefigs=False, plotdirec=None, imdirec=None, suffix=None):

    # Extract O and E from slitpair
    E, O = slitpair
    gradyO, gradxO = np.gradient(O)
    # Extract interpolated x- and y-offsets
    offsx_i, offsy_i = offs_i
    
    
    # Determine median whole-pixel offsets
    '''
    offsfactx = np.nanmedian(np.rint(offsx_i)[0])
    offsfacty = np.nanmedian(np.rint(offsy_i)[1])
    if np.isnan(offsfactx): 
        offsfactx = 0
        print("\t\t\tX offsets is All-NAN slice! -> no X offset")
    if np.isnan(offsfacty): 
        offsfacty = 0
        print("\t\t\tY offsets is All-NAN slice! -> no Y offset")
    print("\t\t\txoffs:\t{}\n\t\t\tyoffs:\t{}\n".format(offsfactx, offsfacty))
    # Determine subpixel offsets over the slits
    gradfactx_i, gradfacty_i = offsx_i % 1, offsy_i % 1
    print("\t\t\tmed_xgrad:\t{}\n\t\t\tmed_ygrad:\t{}\n".format(
                                                       np.nanmedian(gradfactx_i), 
                                                       np.nanmedian(gradfacty_i)))
    '''
    
    
    # Determine piecewise whole-pixel offsets
    offsfactx = int(np.nanmedian(np.rint(offsx_i)[0]))
    offsfacty = int(np.nanmedian(np.rint(offsy_i)[1]))
    # Determine piecewise gradient subtraction factors
    gradfactx_i, gradfacty_i = offsx_i % 1, offsy_i % 1
    
    # Align the slits using the median whole-pixel offset
    framesize = 2*np.array(O.shape)
    lowlcorn = (0.25*framesize).astype(int)
    Oembed = embed(O, framesize, cornerpix=lowlcorn)
    Eembed = embed(E, framesize, offset=[offsfactx,offsfacty], cornerpix=lowlcorn)
    O_E = (Oembed - Eembed)[lowlcorn[0]:lowlcorn[0]+O.shape[0],
                            lowlcorn[1]:lowlcorn[1]+O.shape[1]]       
    OplusE = (Oembed + Eembed)[lowlcorn[0]:lowlcorn[0]+O.shape[0],
                               lowlcorn[1]:lowlcorn[1]+O.shape[1]]   
    
    # Determine new y-limits
    Ny, Nx = O_E.shape
    # Determine minimal y-shapes
    Nxmin = np.min([Nx,gradfactx_i.shape[1]])
    Nymin = np.min([Ny,gradfactx_i.shape[0]])
    '''
    O_E_grad = (O_E - gradfactx_i[0:Ny,0:Nx]*gradxO[0:Ny,0:Nx] 
                    - gradfacty_i[0:Ny,0:Nx]*gradyO[0:Ny,0:Nx])
    '''
    # Subtract piecewise gradient
    O_E_grad = (O_E[0:Nymin,0:Nxmin]
                    - gradfactx_i[0:Nymin,0:Nxmin] * gradxO[0:Nymin,0:Nxmin] 
                    - gradfacty_i[0:Nymin,0:Nxmin] * gradyO[0:Nymin,0:Nxmin])
    
    # Save O-E and O-E-grad.
    if savefigs:
        
        # Create the plot directory
        createdir(plotdirec)
        
        f, ax = plt.subplots()
        plotxcent = np.median(range(O_E.shape[1]))
        O_Eplot = ax.imshow(O_E, origin='lower', cmap='afmhot', 
                            extent=[.126*(0-plotxcent), .126*(O_E.shape[1]-plotxcent),
                                    0, 0.126*(O_E.shape[0])], 
                            vmin=-1, vmax=1)
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.linspace(start, end, 3))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(O_Eplot, cax=cax)
        tick_locator = ticker.MaxNLocator(nbins=4) # Specify number of colorbar ticks
        cb.locator = tick_locator
        cb.update_ticks()
        ax.set_xlabel(r"X [arcsec]", fontsize=20)
        ax.set_ylabel(r"Y [arcsec]", fontsize=20)
        ax.set_title(r"O - E", fontsize=26)
        plt.savefig(plotdirec+"/O-E{}.png".format(suffix))
        #plt.show()
        plt.close()   
        
        f, ax = plt.subplots()
        plotxcent = np.median(range(O_E.shape[1]))
        O_E_gradplot = ax.imshow(O_E_grad, origin='lower', cmap='afmhot', 
                                 extent=[.126*(0-plotxcent), .126*(O_E.shape[1]-plotxcent),
                                         0, 0.126*(O_E.shape[0])], 
                                 vmin=-2e3, vmax=2e3)
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.linspace(start, end, 3))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(O_E_gradplot, cax=cax)
        tick_locator = ticker.MaxNLocator(nbins=4) # Specify number of colorbar ticks
        cb.locator = tick_locator
        cb.update_ticks()
        ax.set_xlabel(r"X [arcsec]", fontsize=20)
        ax.set_ylabel(r"Y [arcsec]", fontsize=20)
        ax.set_title(r"O - E - grad.", fontsize=26)
        plt.savefig(plotdirec+"/O-E-grad{}.png".format(suffix))
        #plt.show()
        plt.close()  
        
        # Save to fits files
        savefits(O_E, imdirec, "O-E{}".format(suffix))
        savefits(OplusE, imdirec, "O+E{}".format(suffix))  
        savefits(O_E_grad, imdirec, "O-E-grad{}".format(suffix)) 
    
    return [O_E, OplusE, O_E_grad]
    
    

# Function for determining linear Stokes parameters and polarization degrees and angles
# in each pixel of a given set of slits slitdiffnorm_lst 
# (containing 4 exposures with different retarder angles: 0, 22.5, 45 and 67.5 degree)
def detpol(slitdiffnorm_lst, S_N, offsxy0__45=np.zeros(2), offsxy22_5__67_5=np.zeros(2), corran=0.):
    
    # Extract different retangle exposures
    [slitdiffnorm0, slitdiffnorm22_5,
     slitdiffnorm45, slitdiffnorm67_5] = slitdiffnorm_lst
    # Align exposures
    slitshape = slitdiffnorm0.shape
    print("DEBUG in detpol slitshape = {}".format(slitshape))
    framesize = 2*np.array(slitshape)
    lowlcorn = (0.25*framesize).astype(int)
    if np.sum(offsxy0__45 != np.zeros(2)) != 0:
        slitdiffnorm0 = embed(slitdiffnorm0, framesize, offset=offsxy0__45, cornerpix=lowlcorn)
        slitdiffnorm45 = embed(slitdiffnorm45, framesize, cornerpix=lowlcorn)
    if np.sum(offsxy22_5__67_5 != np.zeros(2)) != 0:
        slitdiffnorm22_5 = embed(slitdiffnorm22_5, framesize, 
                                 offset=offsxy22_5__67_5, cornerpix=lowlcorn)
        slitdiffnorm67_5 = embed(slitdiffnorm67_5, framesize, cornerpix=lowlcorn)
        
    # Compute double differences
    Q_norm = 0.5*slitdiffnorm0 - 0.5*slitdiffnorm45
    U_norm = 0.5*slitdiffnorm22_5 - 0.5*slitdiffnorm67_5    
    if np.sum(offsxy0__45 != np.zeros(2)) != 0:
        Q_norm = Q_norm[lowlcorn[0]:lowlcorn[0]+slitshape[0],
                        lowlcorn[1]:lowlcorn[1]+slitshape[1]]
    if np.sum(offsxy22_5__67_5 != np.zeros(2)) != 0:
        U_norm = U_norm[lowlcorn[0]:lowlcorn[0]+slitshape[0],
                        lowlcorn[1]:lowlcorn[1]+slitshape[1]]
    # Determine error margins (see Bagnulo 2009 appendix formulae A14 and A15)
    sigmaQ_norm = (1/(2*np.sqrt(len(slitdiffnorm_lst))) / S_N)
    sigmaU_norm = (1/(2*np.sqrt(len(slitdiffnorm_lst))) / S_N)    
    print("DEBUG in detpol Q^2 + U^2 - sigma_Q^2:\t{}".format(U_norm**2 + Q_norm**2 - sigmaU_norm**2))
    
    # Diagnostic plot (check exposure alignment)
    plt.imshow(U_norm - Q_norm, origin='lower', vmin=-1, vmax=1)
    plt.colorbar()
    #plt.show()
    plt.close()
    
    
    # Determine degree and angle of linear polarization
    pL = np.sqrt(U_norm**2 + Q_norm**2)
    phiL = 0.5 * np.arctan(U_norm/Q_norm) #radians
    phiL_DEG = (180/np.pi)*phiL + corran #deg
    
    # Determine error margins (see Bagnulo 2009 appendix formulae A14 and A15)
    sigma_pL = np.sqrt( (np.cos(2*phiL))**2 * sigmaQ_norm**2 + 
                        (np.sin(2*phiL))**2 * sigmaU_norm**2 )
    temp = np.sqrt( (np.sin(2*phiL))**2 * sigmaQ_norm**2 + 
                    (np.cos(2*phiL))**2 * sigmaU_norm**2 ) #NOTE: different from sigma_pL
    sigma_phiL = 1./2. * ( temp / pL ) #rad
    sigma_phiL_DEG = (180./np.pi) * sigma_phiL #deg
    
    return([U_norm, Q_norm, pL, phiL_DEG], [sigmaQ_norm, sigmaU_norm, sigma_pL, sigma_phiL_DEG])
    


# Circular array mask
def cmask(data, center, radius):
    x0,y0 = center
    Ny, Nx = data.shape
    xgrid, ygrid = np.ogrid[-y0:Ny-y0,-x0:Nx-x0]
    mask = xgrid*xgrid + ygrid*ygrid <= radius*radius
    
    # Diagnostic plot
    '''
    plt.imshow(data, origin='lower', cmap='afmhot', alpha=0.5)
    plt.imshow(mask, origin='lower', cmap='Greys', alpha=0.5)#, alpha=0.5)
    plt.scatter(center[0], center[1], color='k', s=50)
    plt.show()
    plt.close()
    '''
    
    return(data[mask], mask)
    


# Function for selecting a rotated rectangle from a data array
def createrectmask(data, boxcent, boxsizs, theta):
    
    # Define rotation matrix #theta needs to be in [rad]
    rotmatrix = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])  
    # Set up rectangle corner coordinates
    uplcorn = boxcent + np.matmul(rotmatrix,0.5*np.array([-boxsizs[0],boxsizs[1]]))
    lowlcorn = boxcent + np.matmul(rotmatrix,0.5*np.array([-boxsizs[0],-boxsizs[1]]))
    lowrcorn = boxcent + np.matmul(rotmatrix,0.5*np.array([boxsizs[0],-boxsizs[1]]))
    uprcorn = boxcent + np.matmul(rotmatrix,0.5*np.array([boxsizs[0],boxsizs[1]]))
    rect_verts = np.rint([uplcorn, lowlcorn, lowrcorn, uprcorn]).astype(int)
    
    # Create vertex coordinates for each grid cell
    ny, nx = data.shape
    xgrid, ygrid = np.meshgrid(np.arange(nx),np.arange(ny))
    points = np.vstack((xgrid.flatten(),ygrid.flatten())).T
    
    # Create boolean rotated rectangular mask
    path = Path(rect_verts)
    rectmask = path.contains_points(points)
    rectmask = rectmask.reshape((ny,nx))
    
    # Diagnostic plot
    plt.imshow(data, origin='lower', cmap='afmhot', alpha=0.5, vmin=-10, vmax=60)
    plt.imshow(rectmask, origin='lower', cmap='Greys', alpha=0.5)#, alpha=0.5)
    plt.scatter(rect_verts[:,0], rect_verts[:,1], color='k')
    #plt.show()
    plt.close()
    
    return(data[rectmask], rectmask)
    

#################### END FUNCTIONS FOR SLIT APPENDAGE ####################
#################### END FUNCTIONS #############################################




