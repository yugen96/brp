import numpy as np
from astropy.io import fits
from itertools import product as carthprod
import shutil
import os

from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import interpolate
from scipy.stats import poisson
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
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
def apersum_old(image, px, py, r):
    
    
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
            if np.isnan(pixval) or np.isinf(pixval):
                continue
            
            if np.isnan(pixval):
                print("DEBUG!!!!!")
                print(pixval)
            
            # Append pixval to pixvals for calculation of median
            pixvals.append(pixval)
            
            # Add to aperture sum
            if d2 <= r**2:
                apsum += pixval
                apcount += 1      
    
    # Determine the mean and the median inclusiding Poissonian error estimators
    mean, med = apsum/apcount, np.median(pixvals)
    mean_err, med_err = np.sqrt(abs(mean)/apcount), 1/(4 * apcount * poisson.pmf(med, mean)**2)
    
    #print("DEBIG [apsum, apcount, mean]:\t\t", apsum, apcount, mean)
    #print("DEBUG [med, mean, poisson(med,mean)]:\t\t", med, mean, poisson.pmf(med,mean))
    
    return [apsum, mean, med], [np.sqrt(abs(apsum)), mean_err, med_err]
    
    

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
    # List for storing the pixel values
    pixvals = []
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
            # Append pixel value
            pixvals.append(pixval)
            
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
        #print("\n\n\t\t\t\titeration:\t{} \n\t\t\t\t sigma_an:\t{}".format(n, sigma_an))
        #TODO UNCOMMENT
        
    # Compute and return calibrated aperture flux
    apscal = apsum - apcount*av_an          
    return [apscal, apscal/apcount, np.median(pixvals)], [ansum, av_an, np.median(pixvals_an)]
                       


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
        print("\n\n\t\t\t\titeration:\t{} \n\t\t\t\t sigma_an:\t{}".format(n, sigma_an))
        #TODO UNCOMMENT
        
    
    # Compute the Poissonian sample mean and median as well as the corresponding errors
    mean, med = av_an, np.median(pixvals_an)
    mean_err, med_err = np.sqrt(abs(mean)/ancount), 1/(4 * ancount * poisson.pmf(med, mean)**2)
    
    #print("DEBIG [ansum, ancount, av_an]:\t\t", ansum, ancount, av_an)
    #print("DEBUG [med, mean, poisson(med,mean)]:\t\t", med, mean, poisson.pmf(med,mean))
    
    return [ansum, av_an, np.median(pixvals_an)], [np.sqrt(abs(ansum)), mean_err, med_err]
        
    

# Function which computes normalized flux differences as well as the ordinary and extraordinary counts for a preselection regions in various images defined by 'loc_lsts'. 
def compute_fluxlsts(data_dirs, bias, masterflat_norm, loc_lsts, r_range):

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
        savedir = data_dir.rsplit("/sorted")[0] + "/sorted/loadfiles/" + data_dir.rsplit("/",2)[1]
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        np.save(savedir + "/O_0lst.npy", O_0lst), np.save(savedir + "/sigmaO_0lst.npy", sigmaO_0lst)
        np.save(savedir + "/E_0lst.npy", E_0lst), np.save(savedir + "/sigmaE_0lst.npy", sigmaE_0lst)
        np.save(savedir + "/F_0lst.npy", F_0lst), np.save(savedir + "/sigmaF_0lst.npy", sigmaF_0lst)
        # Save filt_lst and pos0_lst
        np.save(savedir + "/filter_lst.npy", filter_lst), np.save(savedir + "/pos0_lst.npy", pos0_lst)
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
    
    return np.array([Q_jqr, U_jqr, P_jqr, Phi_jqr]), np.array([sigmaQ_jqr, sigmaU_jqr, sigmaP_jqr, sigmaPhi_jqr])



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



# Function for showing and saving png image
def saveim_png(data, savedir, fname, 
               colmap='coolwarm', orig=None, datextent=None, interp='None',
               xtag='X', ytag='Y', title=None):
    
    # Show and save image    
    plt.figure()
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.imshow(data, cmap=colmap, origin=orig, 
               norm=norm, interpolation=interp, extent=datextent)
    plt.colorbar()
    plt.xlabel(xtag, fontsize=20), plt.ylabel(ytag, fontsize=20)
    if not title == None:
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
                 fit=False, fitZdata=None, fitdataplttype='surface',
                 colour='r', colmap='coolwarm', rowstride=1, colstride=1, lw=0,
                 xtag='X', ytag='Y', ztag='Z', title=None):
    
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
            fitplt = ax.scatter(Xgrid, Ygrid, fitZdata, color=colour)
        elif fitdataplttype == 'surface':
            fitplt = ax.plot_surface(Xgrid, Ygrid, fitZdata, cmap=colmap, 
                                     rstride=rowstride, cstride=colstride, 
                                     linewidth=lw, antialiased=False)
            fig.colorbar(fitplt, shrink=1.0, aspect=20)
    
    # Save plot
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    ax.set_xlabel(xtag, fontsize=20), ax.set_ylabel(ytag, fontsize=20)
    ax.set_zlabel(ztag, fontsize=20)
    if not title == None:
        plt.title(title, fontsize=24)
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
    print(checkP, checkphi)
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
def embed(data, frameshape, offset=np.zeros(2), cornerpix=[0,0]):
    
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
def mask2d(data, mask, compute_uplcorn=False):
    
    # Create flattened masked array
    idx = mask.sum(1).cumsum()[:-1]
    flatmasked = np.split(data[mask], idx)
    
    # Reshape to form 2d array and find the upper left corner of the cropped region
    lst2d, firstrow = [], None
    for rowno, row in enumerate(flatmasked):
        possrow = list(row)
        if len(possrow) != 0:
            lst2d.append(possrow)
            # Determine the index of the first row containing a True value
            if compute_uplcorn and (firstrow == None):
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
    if pixoffs == None:
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
def offsetopt_well(ims, dxrange, dyrange, center, R, anRmin, anRmax, cutoutR=10,
                   saveims=False, pltsavedir=None, imsavedir=None, starno=27):
    
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
            plt.imshow(interim, origin='lower')
            plt.scatter(newcent[0], newcent[1], c='k', s=40)
            plt.show()
            plt.close() 
            '''           
            
            
            # Determine normalized flux
            F = apersum(interim, center[0], center[1], R, anRmin, anRmax)
            
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
    plt.imshow(offset_arr, origin='lower')
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
    ''' # Diagnostic plot    
    
    
    '''
    var = [5,5,5,10,10,5] # Allowed sigma's for x0, y0, z0, sigx, sigy and A
    if ( np.diag(pcov_gauss) == var[i] for i in range(len(var)) ):
        print("\nPossible problem in fit!\n")
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
        offsetopt_well(ims, newdxrange, newdyrange, center, R, anRmin, anRmax, 
                       saveims=saveims, pltsavedir=pltsavedir, imsavedir=imsavedir)
    '''# Diagnostic plot
    
    
    # Create final slit image using guassian fit
    offsetopt_fitgauss = np.rint(popt_gauss[0:2]).astype(int)
    finalim_gauss = align_slits(ims, np.tile((offsetopt_fitgauss),[len(ims),1]), detoffs=False)
    
    
    
    # PLOTS
    if saveims:
        saveim_png(offset_arr, pltsavedir+"/", 
                   "offsetopt{}tpl8_dx{}dy{}".format(starno+1, 
                                                     offsetopt_fitgauss[0],
                                                     offsetopt_fitgauss[1]), 
                   colmap='coolwarm', orig='lower',
                   datextent=[np.min(dxrange), np.max(dxrange), np.min(dyrange), np.max(dyrange)],
                   xtag=r"$\delta_X$", ytag=r"$\delta_Y$")
        # Save offset_arr and paraboloid fit as 3D png
        save3Dim_png(dxgrid, dygrid, offset_arr, pltsavedir, 
                     "offsetopt{}tpl8_gauss3Dv2".format(starno+1), 
                     fit=True, fitZdata=fitgauss2d, xtag=r"$\delta_X$", ytag=r"$\delta_Y$",
                     ztag=r"$\left( (O-E)/(O+E) \right)^2$")
        
        
        # Save Gaussian finalim to fits file
        savefits(finalim_gauss, imsavedir, "finalim{}dx{}dy{}".format(starno+1,
                                                                      offsetopt_fitgauss[0],
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
def offsetopt_cd(O, E, crange, drange, center, R, savetofits=False,
                 savedir=None, gradoptname=None):
    
    # Determination of c and d (gradient method)
    slitdiff, slitsum = O-E, O+E    
    grady, gradx = np.gradient(O)
    
    # Diagnostic plot
    '''
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    dat1 = ax1.imshow(slitdiff, origin='lower', cmap="afmhot")
    plt.colorbar(dat1)
    dat2 = ax2.imshow(grady, origin='lower', cmap="afmhot")
    plt.colorbar(dat2)
    plt.show()
    plt.close()
    '''
    
    Qmin_cd, Qmin = np.array([0,0]), np.inf
    for c in crange:
        for d in drange:
            
            temp = (slitdiff - c*gradx - d*grady)**2
            Qlst, Qlst_err = apersum_old(temp, center[0], center[1], R)
            Q = Qlst[0]
            
            savefits(slitdiff-c*gradx-d*grady, savedir+"/NONABS", 
                     "testc{}d{}".format(np.round([c,d],3)[0], np.round([c,d],3)))
            
            if Q <= Qmin:
                Qmin, Qmin_cd = Q, np.array([c,d])
                gradoptabs = np.sqrt(temp)
                gradopt = slitdiff - Qmin_cd[0]*gradx - Qmin_cd[1]*grady
    
    if savetofits:
        # Diagnostic image
        '''
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        dat1 = ax1.imshow(slitdiff, origin='lower', cmap="afmhot")
        plt.colorbar(dat1)
        dat2 = ax2.imshow(gradopt, origin='lower', cmap="afmhot")
        plt.colorbar(dat2)
        plt.show()
        plt.close()
        '''
        # Save the gradient corrected slit
        savefits(gradopt, savedir, gradoptname+"c{}d{}".format(np.round(Qmin_cd[0],3), 
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
    

        
        


#################### END FUNCTIONS FOR SLIT APPENDAGE ####################
#################### END FUNCTIONS #############################################





# Specify data and filename
datadir = "/home/bjung/Documents/Leiden_University/brp/data_red/calib_data"
scidatadir = datadir + "/sorted/NGC4696,IPOL"
sci_dirs = [scidatadir + "/CHIP1"]
testdata = sci_dirs[0] + "/tpl8/corrected2/FORS2.2011-05-04T01:31:46.334_COR.fits" # j=7, k=1
# Load testdata
header, data = extract_data(testdata)
# Directory for saving plots
plotdir = "/home/bjung/Documents/Leiden_University/brp/data_red/plots/NGC4696,IPOL"
imdir = "/home/bjung/Documents/Leiden_University/brp/data_red/images"
tabledir = "/home/bjung/Documents/Leiden_University/brp/data_red/tables"

# Specify bias and masterflat
header, Mbias = extract_data(datadir + "/masterbias.fits")
header, Mflat_norm = extract_data(datadir + "/masterflats/masterflat_norm_FLAT,LAM_IPOL_CHIP1.fits")



# Aproximate coordinates of selection of stars within CHIP1 of 'Vela1_95' and 'WD1615_154'. Axis 0 specifiec the different sci_dirs; axis 1 specifies the different stars within the sci_dirs; axis 2 specifies the x, y1, y2 coordinate of the specific star (with y1 specifying the y coordinate on the upper slit and y2 indicating the y coordinate on the lower slit) and the aproximate stellar radius. NOTE: THE LAST LIST WITHIN AXIS1 IS A SKY APERTURE!!!
star_lsts = [[[335, 904, 807, 5], [514, 869, 773, 7], [1169, 907, 811, 5], [1383, 878, 782, 7], 
              [341, 694, 599, 10], [370, 702, 607, 11], [362, 724, 630, 5], [898, 709, 609, 8], 
              [1836, 707, 611, 6], [227, 523, 429, 6], [354, 498, 404, 10], [376, 512, 418, 8], 
              [419, 525, 431, 7], [537, 491, 392, 7], [571, 541, 446, 8], [1096, 510, 416, 5], 
              [1179, 530, 436, 8], [487, 320, 226, 7], [637, 331, 238, 6], [1214, 345, 252, 6], 
              [1248, 326, 233, 6], [1663, 308, 217, 9], [326, 132, 40, 5], [613, 186, 94, 10], 
              [634, 184, 91, 9], [642, 134, 41, 7], [838, 175, 82, 8], [990, 140, 48, 11], 
              [1033, 157, 65, 9], [1172, 147, 55, 7], [1315, 164, 71, 8], [1549, 164, 72, 13]]] 
star_lsts = np.array(star_lsts) # 32 stars in total (42-10)

# List specifying the (axis1) indices of the first stars on each slit
slit_divide = np.array([1, 5, 10, 18, 23])



# Range of aperture radii
r_range = np.arange(1, 16) #[pixels]

# Pixel scale
pixscale = 0.126 #[arcsec/pixel]

# Boolean variable for switchin on polarization computations of selected stars
compute_anew = False
calc_cd, calc_well = True, False

# ESO given polarizations
VelaBV_PlPhi = [[0., 0.],[0., 0.]] # [-], [deg]
VelaBV_sigmaPlPhi = [[0., 0.],[0., 0.]] # [-], [deg]
WD1615BV_PlPhi, WD1615BV_sigmaPlPhi = [[0., None], [0., None]], [[0., None], [0., None]]
ESObvPLPHI = np.array([VelaBV_PlPhi, WD1615BV_PlPhi]) 
sigmaESObvPLPHI = np.array([VelaBV_sigmaPlPhi, WD1615BV_sigmaPlPhi])



# Compute fluxes and polarizations for selected stars in testdata and carry out slit appenditure
if compute_anew == True:
    compute_fluxlsts(sci_dirs, Mbias, Mflat_norm, star_lsts, r_range)


# Define the x- and y-ranges corresponding to the chip
chip_xyranges = [[183,1868],[10,934]]


# Cut and append slits
slits, upedges, lowedges, gapw, slitshapes = cut_to_slits(data)
# Make slits same shape
slits = [slits[i][0:np.min(slitshapes[:,0]),
                  0:np.min(slitshapes[:,1])] for i in range(0,10,1)]


# Determine slitwidths
upedges, lowedges, gapw = [np.array(temp) for temp in [upedges, lowedges, gapw]]
slitwidths = upedges-lowedges








# Create template aligned image
slitshapes = np.array([np.array(slits[i].shape) for i in range(0,10,1)])
aligntemp = np.concatenate([slits[i+1] for i in range(0,10,2)], axis = 0)
# Recall previous c- and dscapes
header, cscape_prev = extract_data(imdir+"/offsetopt/old/cdscapes/cscape_tpl8.fits")
header, dscape_prev = extract_data(imdir+"/offsetopt/old/cdscapes/dscape_tpl8.fits")



# Load list with the filenames of the non-interpolated corrected slits
aligneddirs, alignedfiles = mk_lsts(imdir+"/offsetopt/noninterp4")
aligneddatadict = {}
# Read out the offsets for each file
offsets = np.zeros([32,2])
for f in alignedfiles:
    print(f)
    # Store the aligned images in a dictionary
    header, aligneddatadict[f.split("dx")[0]] = extract_data(imdir+"/offsetopt/noninterp4/"+f)
    # Retrieve star number
    starno = f.split("finalim")[1][0:2]
    try: starno = int(starno)
    except ValueError: starno = int(starno[0])
    
    # Retrieve x offset
    temp = f.split("dx")[1][0]
    if temp == "-": dx = int(f.split("dx")[1][0:2])
    else: dx = int(temp)
    
    # Retrieve y offset
    dy = int(f.split("dy")[1][0:2])
    offsets[starno-1] = [dx,dy]
# Change into integer array
offsets = np.array(offsets).astype(int)
    
    

# Apply gradient method on all stars and create cdscapes
Qopts, opts_cd = [], []
n, cscape, dscape = len(slits)-2, np.zeros(aligntemp.shape), np.zeros(aligntemp.shape)
for starno, starpar in enumerate(star_lsts[0]):
    
    # Check whether to compute cdscapes
    if calc_cd == False:
        break
    print("\n\nComputing c- and d-scapes...")
    print("\n\nStarno:\t\t{}".format(starno+1))
    
    # Check whether current star is the first one appearing on a slitpair
    if (starno+1) in slit_divide:
        # Extract ordinary and extraordinary slit
        slitE, slitO = slits[n], slits[n+1]
        # Adjust shape so that O and E have equal size
        '''
        Nx, Ny = min(slitE.shape[1], slitO.shape[1]), min(slitE.shape[0], slitO.shape[0])
        slitE, slitO = slitE[0:Ny,0:Nx], slitO[0:Ny,0:Nx]
        '''
        # Determine the upper and lower edges of the slits
        upedgeE, upedgeO = upedges[n], upedges[n+1]
        lowedgeE, lowedgeO = lowedges[n], lowedges[n+1]
        print("Slit pair {}".format(n/2))
        n -= 2
        
    
    # Only evaluate test stars
    if starno+1 in [1]:
        backgrcent = backgrcents[teststarno]
        teststarno += 1
    else: 
        continue
        
    
    # Compute stellar location on O slit and on the template appended slit
    slitOcent = find_center([starpar[0]-chip_xyranges[0][0], starpar[1]-lowedgeO],
                             slitO, 15)
    appendedOcent = find_center([slitOcent[0], 
                                 slitOcent[1]+np.sum(slitwidths[[m+1 for m in np.arange(0,n+2,2)]])],
                                 aligntemp, 25) #TODO SAVE TO NP FILE
    print("appendedOcent:\t\t", appendedOcent)
    
    
    # Create cutout
    cutxmin, cutxmax = max(0, slitOcent[0]-35), min(slitO.shape[1]-1, slitOcent[0]+35)
    cutymin, cutymax = max(0, slitOcent[1]-35), min(slitO.shape[0]-1, slitOcent[1]+35)
    cutoutO = slitO[cutymin:cutymax, cutxmin:cutxmax]
    cutoutE = slitE[cutymin:cutymax, cutxmin:cutxmax]
    cutoutOcent = (slitOcent - np.rint([cutxmin,cutymin])).astype(int)
    # Apply whole pixel-accuracy offset
    framesize = 2*np.array(cutoutO.shape)
    lowlcorn = (0.25*framesize).astype(int)
    embedE = embed(cutoutE, framesize, offset=offsets[starno], cornerpix=lowlcorn)
    embedO = embed(cutoutO, framesize, offset=[0,0], cornerpix=lowlcorn)
    embedOcent = np.rint(0.25*framesize[[1,0]]).astype(int) + cutoutOcent
        
    
    # Diagnostic plot
    '''
    plt.figure()
    plt.imshow(embedO, origin='lower', cmap='rainbow')
    plt.scatter(embedOcent[0], embedOcent[1], s=30, c='k')
    plt.scatter(60, 45, s=30, c='k')
    plt.show()
    plt.close()
    
    plt.figure()
    plt.imshow(embedE, origin='lower', cmap='rainbow')
    plt.scatter(embedOcent[0], embedOcent[1], s=30, c='k')
    plt.show()
    plt.close()
    '''
    
    
    
    
    # Save original O-E
    savefits(embedO-embedE, imdir+"/background_tests/backgrtest_star{}".format(starno+1),
             "slitdiff_star{}".format(starno+1)) 
    savefits(embedO, imdir+"/background_tests/backgrtest_star{}".format(starno+1), "O_star{}".format(starno+1))       
    savefits(embedE, imdir+"/background_tests/backgrtest_star{}".format(starno+1), "E_star{}".format(starno+1)) 
    #TODO The allocation of the stellar center on the aligned image is OK
    
    
    
    
    # Determine stellar and background statistics in O and E
    Ostar, Ostar_err = apersum_old(embedO, embedOcent[0], embedOcent[1], starpar[3])
    Estar, Estar_err = apersum_old(embedE, embedOcent[0]+offsets[starno][0], 
                                           embedOcent[1]+offsets[starno][1], starpar[3])
    bckgO_reg, bckgO_reg_err = apersum_old(embedO, 60, 41, 15)
    bckgE_reg, bckgE_reg_err = apersum_old(embedE, 60, 41, 15)
    bckgO_an, bckgO_an_err = ansum(embedO, embedOcent[0], embedOcent[1], starpar[3], 30)
    bckgE_an, bckgE_an_err = ansum(embedE, embedOcent[0], embedOcent[1], starpar[3], 30)
    
    # Subtract median background from the O and E frames
    Ocorr_reg, Ecorr_reg = embedO - bckgO_reg[2], embedE - bckgE_reg[2]
    Ocorr_an, Ecorr_an = embedO - bckgO_an[2], embedE - bckgE_an[2]
    savefits(Ocorr_reg, imdir+"/background_tests/backgrtest_star{}".format(starno+1), "Ocorr_reg_star{}".format(starno+1))
    savefits(Ecorr_reg, imdir+"/background_tests/backgrtest_star{}".format(starno+1), "Ecorr_reg_star{}".format(starno+1))
    savefits(Ocorr_an, imdir+"/background_tests/backgrtest_star{}".format(starno+1), "Ocorr_an_star{}".format(starno+1))
    savefits(Ecorr_an, imdir+"/background_tests/backgrtest_star{}".format(starno+1), "Ecorr_an_star{}".format(starno+1))
    
    # Determine stellar and background statistics in O and E - BG_reg
    Ocorr_star, Ocorr_star_err = apersum_old(Ocorr_reg, embedOcent[0], embedOcent[1], starpar[3])
    Ecorr_star, Ecorr_star_err = apersum_old(Ecorr_reg, embedOcent[0], embedOcent[1], starpar[3])
    bckgOcorr_reg, bckgOcorr_reg_err = apersum_old(Ocorr_reg, 60, 41, 15)
    bckgEcorr_reg, bckgEcorr_reg_err = apersum_old(Ecorr_reg, 60, 41, 15)
    bckgOcorr_an, bckgOcorr_an_err = ansum(Ocorr_reg, 
                                           embedOcent[0], embedOcent[1], starpar[3], 30)
    bckgEcorr_an, bckgEcorr_an_err = ansum(Ecorr_reg, 
                                           embedOcent[0], embedOcent[1], starpar[3], 30)
    
    
    
    # Determine slitdifferences for background corrected images
    slitdiff_regcorr = Ocorr_reg - Ecorr_reg
    slitdiff_ancorr = Ocorr_an - Ecorr_an
    savefits(slitdiff_regcorr, imdir+"/background_tests/backgrtest_star{}".format(starno+1), "slitdiff_regcorr_star{}".format(starno+1))
    savefits(slitdiff_ancorr, imdir+"/background_tests/backgrtest_star{}".format(starno+1), "slitdiff_anncorr_star{}".format(starno+1))    
    # Compute left-over background signals
    bckgdiff_regcorr, bckgdiff_regcorr_err = apersum_old(slitdiff_regcorr, 60, 41, 15)
    bckgdiff_ancorr, bckgdiff_ancorr_err = apersum_old(slitdiff_ancorr, 60, 41, 15)
    bckgdiff_regcorrABS, bckgdiff_regcorr_errABS = apersum_old(np.abs(slitdiff_regcorr), 60, 41, 15)
    bckgdiff_ancorrABS, bckgdiff_ancorr_errABS = apersum_old(np.abs(slitdiff_ancorr), 60, 41, 15)
    
    
    
    
    
    # Recall previous c and d values for current star
    cval_prev = cscape_prev[appendedOcent[1], appendedOcent[0]]
    dval_prev = dscape_prev[appendedOcent[1], appendedOcent[0]]
    # Determine c and d values to use for evaluation
    crange = np.arange(-1.2, 1.25, 0.05) #TODO adjust to cval_prev
    drange = np.arange(-1.2, 1.25, 0.05) #TODO adjust to dval_prev   
    
    
    
    # Compute the c and d parameters which optimize overlap using gradient method
    gradopt_old, Qopt, opt_cd = offsetopt_cd(embedO, embedE, crange, drange,
                                embedOcent, starpar[3], 
                                savetofits=True, savedir=imdir+"/background_tests/backgrtest_star{}".format(starno+1), 
                                gradoptname="gradoptOLD_star{}".format(starno+1))
    Qopts.append(Qopt), opts_cd.append(opt_cd)
    print("Qopt, opt_cd:\t\t", Qopt, opt_cd)
    
    # New c and d parameters
    newcrange = np.arange(opt_cd[0]-0.5, opt_cd[0]+0.52, 0.02)
    newdrange = np.arange(opt_cd[1]-0.5, opt_cd[1]+0.52, 0.02)
    # Compute the c and d parameters anew with higher accuracy
    gradopt_old, Qopt, opt_cd = offsetopt_cd(embedO, embedE, newcrange, newdrange,
                                embedOcent, starpar[3], 
                                savetofits=True, savedir=imdir+"/background_tests/backgrtest_star{}".format(starno+1), 
                                gradoptname="gradoptOLDbetter_star{}".format(starno+1))
    Qopts.append(Qopt), opts_cd.append(opt_cd)
    print("Qopt, opt_cd:\t\t", Qopt, opt_cd)
    
    # Compute the c and d parameters for sky region corrected images
    gradopt_reg, Qopt, opt_cd = offsetopt_cd(Ocorr_reg, Ecorr_reg, crange, drange,
                                embedOcent, starpar[3],
                                savetofits=True, savedir=imdir+"/background_tests/backgrtest_star{}".format(starno+1), 
                                gradoptname="gradoptREG_star{}".format(starno+1))
    Qopts.append(Qopt), opts_cd.append(opt_cd)
    print("Qopt, opt_cd:\t\t", Qopt, opt_cd)
    # Compute the c and d parameters for sky annulus corrected images
    gradopt_an, Qopt, opt_cd = offsetopt_cd(Ocorr_an, Ecorr_an, crange, drange,
                                embedOcent, starpar[3], 
                                savetofits=True, savedir=imdir+"/background_tests/backgrtest_star{}".format(starno+1), 
                                gradoptname="gradoptANN_star{}".format(starno+1))
    Qopts.append(Qopt), opts_cd.append(opt_cd)
    print("Qopt, opt_cd:\t\t", Qopt, opt_cd)
    
    
    
    
    
    # Compute the sum over the old and new residuals    
    oldres, oldres_err = apersum_old(np.abs(gradopt_old), 
                                     embedOcent[0], embedOcent[1], starpar[3])
    newres_reg, newres_reg_err = apersum_old(np.abs(gradopt_reg), 
                                             embedOcent[0], embedOcent[1], starpar[3]) 
    newres_an, newres_an_err = apersum_old(np.abs(gradopt_an), 
                                           embedOcent[0], embedOcent[1], starpar[3]) 
    # Determine the background flux in derived gradient method optima images
    # NORMAL SCALE
    bckgOPT_old, bckgOPT_old_err = apersum_old(gradopt_old, 60, 41, 15) 
    bckgOPT_reg, bckgOPT_reg_err = apersum_old(gradopt_reg, 60, 41, 15) 
    bckgOPT_an, bckgOPT_an_err = apersum_old(gradopt_an, 60, 41, 15)    
    # ABSOLUTE SCALE 
    bckgOPT_oldABS, bckgOPT_old_errABS = apersum_old(np.abs(gradopt_old), 60, 41, 15) 
    bckgOPT_regABS, bckgOPT_reg_errABS = apersum_old(np.abs(gradopt_reg), 60, 41, 15) 
    bckgOPT_anABS, bckgOPT_an_errABS = apersum_old(np.abs(gradopt_an), 60, 41, 15) 







    # Create tables
    savedir = tabledir
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    
    '''
    # Go to table directory and open new file
    os.chdir(savedir)                
    savefile1 = open("tables1_star1",'w')
    savefile1.write("\\textbf{Type} \t&\t \\textbf{R [pixels]} \t&\t \\textbf{$\\Sigma$ [ADU]} \t&\t \\textbf{$\\mu$ [ADU]} \t&\t \\textbf{MD [ADU]} \\\\ \n")
    savefile1.write("\\hline \hline \n")
    types = ["$O$", "$E$", "$O-E$", 
             "$O-E, optm.$", "$O-E - bg_{sky,reg}, optm.$", "$O-E - bg_{sky,ann}, optm.$"]
    R = [str(starpar[3]), str(2*starpar[3]), 15, str(starpar[3]), str(2*starpar[3]), 15]
    aperdata = np.round([Ostar, Estar, slitdiff, Estar, bckgE_reg, bckgE_an])
    aperdata_err = np.round([Ostar_err, bckgO_reg_err, bckgO_an_err, 
                             Estar_err, bckgE_reg_err, bckgE_an_err], 2)
    for i in range(6):
        print(types[i])
        #savefile1.write("   \t&\t {A1} \t&\t {A2} \t&\t {B} \t&\t {C} \t&\t {D}\\\\ \n".format(A1=types[i], A2=R[i], B=aperdata[i][0], C=aperdata[i][1],D=aperdata[i][2]))
        savefile1.write("   \t&\t {A1} \t&\t {A2} \t&\t ${B}\\pm{Berr}$ \t&\t ${C}\\pm{Cerr}$ \t&\t ${D}$\\\\ \n".format(A1=types[i], A2=R[i], B=aperdata[i][0], Berr=aperdata_err[i][0], 
                                    C=aperdata[i][1], Cerr=aperdata_err[i][1], D=aperdata[i][2]))
    '''
                                            
        

    # Go to table directory and open new file
    os.chdir(savedir)                
    savefile1 = open("tables1_star1",'w')
    # star aper, sky aper, sky annul for O and for E
    savefile1.write("star aper, sky aper, sky annul; O-E\n")  
    savefile1.write(" \t&\t \\textbf{type} \t&\t \\textbf{R [pixels]} \t&\t \\textbf{$\\Sigma$ [ADU]} \t&\t \\textbf{$\\mu$ [ADU]} \t&\t \\textbf{MD [ADU]} \\\\ \n")
    savefile1.write("\\hline \hline \n")
    types = ["star aper", "sky aper", "sky annul", "star aper", "sky aper", "sky annul"]
    R = np.tile([starpar[3], 15, str(starpar[3])+"-"+str(2*starpar[3])],2)
    aperdata = np.round([Ostar, bckgO_reg, bckgO_an, 
                         Estar, bckgE_reg, bckgE_an], 2)
    aperdata_err = np.round([Ostar_err, bckgO_reg_err, bckgO_an_err, 
                             Estar_err, bckgE_reg_err, bckgE_an_err], 2)
    for i in range(6):
        print(types[i])
        savefile1.write("   \t&\t {A1} \t&\t {A2} \t&\t ${B}\\pm{Berr}$ \t&\t ${C}\\pm{Cerr}$ \t&\t ${D}$\\\\ \n".format(A1=types[i], A2=R[i], B=aperdata[i][0], Berr=aperdata_err[i][0], 
                                    C=aperdata[i][1], Cerr=aperdata_err[i][1], D=aperdata[i][2]))
    
    
    
    # star aper, sky aper, sky annul for O and for E - BG  
    savefile1.write("\n\n")   
    savefile1.write("star aper, sky aper, sky annul; O-E-BG_reg\n")  
    savefile1.write(" \t&\t \\textbf{type} \t&\t \\textbf{R [pixels]} \t&\t \\textbf{$\\Sigma$ [ADU]} \t&\t \\textbf{$\\mu$ [ADU]} \t&\t \\textbf{MD [ADU]} \\\\ \n")
    savefile1.write("\\hline \hline \n")
    types = ["star aper", "sky aper", "sky annul", "star aper", "sky aper", "sky annul"]
    R = np.tile([starpar[3], 15, str(starpar[3])+"-"+str(2*starpar[3])],2)
    aperdata = np.round([Ocorr_star, bckgOcorr_reg, bckgOcorr_an, 
                         Ecorr_star, bckgEcorr_reg, bckgEcorr_an], 2)
    aperdata_err = np.round([Ocorr_star_err, bckgOcorr_reg_err, bckgOcorr_an_err, 
                             Ecorr_star_err, bckgEcorr_reg_err, bckgEcorr_an_err], 2)
    for i in range(6):
        print(types[i])
        savefile1.write("   \t&\t {A1} \t&\t {A2} \t&\t ${B}\\pm{Berr}$ \t&\t ${C}\\pm{Cerr}$ \t&\t ${D}$\\\\ \n".format(A1=types[i], A2=R[i], B=aperdata[i][0], Berr=aperdata_err[i][0], 
                                    C=aperdata[i][1], Cerr=aperdata_err[i][1], D=aperdata[i][2]))
    
    
    
    
    
    # Write background corrected slit difference tables
    savefile1.write("\n\n\n\n\n")
    #savefile1.write(" \t&\t \\textbf{$\\Sigma$ [ADU]} \t&\t \\textbf{$\\hat{se}_{\\Sigma}$ [ADU]} \t&\t \\textbf{$\\mu$ [ADU]} \t&\t \\textbf{$\\hat{se}_{\\mu}$ [ADU]} \t&\t \\textbf{MD [ADU]} \t&\t \\textbf{$\\hat{se}_{MD}$ [ADU]} \\\\ \n")
    savefile1.write("\\textbf{Image Type} \t&\t \\textbf{R [pixels]} \t&\t \\textbf{$\\Sigma_{|backgr.|}$ [ADU]} \t&\t \\textbf{$\\mu_{|backgr.|}$ [ADU]} \t&\t \\textbf{$MD_{|backgr.|}$ [ADU]} \\\\ \n")
    savefile1.write("\\hline \n")   
    types = ["regional sky corr.", "$|\\text{regional sky corr.}|$",
             "annulus sky corr.", "$|\\text{annulus sky corr.}|$"]
    aperdata = np.round([bckgdiff_regcorr, bckgdiff_regcorrABS, 
                         bckgdiff_ancorr, bckgdiff_ancorrABS], 2)
    aperdata_err = np.round([bckgdiff_regcorr_err, bckgdiff_regcorr_errABS, 
                             bckgdiff_ancorr_err, bckgdiff_ancorr_errABS], 2)
    R = [15, 15, 15, 15]
    for i in range(4):
        savefile1.write("{A1} \t&\t {A2} \t&\t ${B}\\pm{Berr}$ \t&\t ${C}\\pm{Cerr}$ \t&\t ${D}$\\\\ \n".format(A1=types[i], A2=R[i], B=aperdata[i][0], Berr=aperdata_err[i][0], 
                                 C=aperdata[i][1], Cerr=aperdata_err[i][1],
                                 D=aperdata[i][2])) #Derr=aperdata_err[i][2]))       
    
    
    
    
    
    # Write residual tables
    savefile1.write("\n\n\n\n\n")
    #savefile1.write(" \t&\t \\textbf{$\\Sigma$ [ADU]} \t&\t \\textbf{$\\hat{se}_{\\Sigma}$ [ADU]} \t&\t \\textbf{$\\mu$ [ADU]} \t&\t \\textbf{$\\hat{se}_{\\mu}$ [ADU]} \t&\t \\textbf{MD [ADU]} \t&\t \\textbf{$\\hat{se}_{MD}$ [ADU]} \\\\ \n")
    savefile1.write("\\textbf{Image Type} \t&\t \\textbf{$\\Sigma$ [ADU]} \t&\t \\textbf{$\\mu$ [ADU]} \t&\t \\textbf{MD [ADU]} \\\\ \n")
    savefile1.write("\\hline \n")   
    types = ["uncorrected", "regional sky corr.", "annul. sky corr."]
    aperdata = np.round([oldres, newres_reg, newres_an], 2)
    aperdata_err = np.round([oldres_err, newres_reg_err, newres_an_err], 2)
    for i in range(3):
        savefile1.write("{A} \t&\t ${B}\\pm{Berr}$ \t&\t ${C}\\pm{Cerr}$ \t&\t ${D}$\\\\ \n".format(A=types[i], B=aperdata[i][0], Berr=aperdata_err[i][0], 
     C=aperdata[i][1], Cerr=aperdata_err[i][1],
     D=aperdata[i][2])) #Derr=aperdata_err[i][2]))

    # Jump back to original directory
    os.chdir(scidatadir)


    # Write background corrected optima background tables
    savefile1.write("\n\n\n\n\n")
    #savefile1.write(" \t&\t \\textbf{$\\Sigma$ [ADU]} \t&\t \\textbf{$\\hat{se}_{\\Sigma}$ [ADU]} \t&\t \\textbf{$\\mu$ [ADU]} \t&\t \\textbf{$\\hat{se}_{\\mu}$ [ADU]} \t&\t \\textbf{MD [ADU]} \t&\t \\textbf{$\\hat{se}_{MD}$ [ADU]} \\\\ \n")
    savefile1.write("\\textbf{Image Type} \t&\t \\textbf{$\\Sigma_{backgr.}$ [ADU]} \t&\t \\textbf{$\\mu_{backgr.}$ [ADU]} \t&\t \\textbf{$MD_{backgr.}$ [ADU]} \\\\ \n")
    savefile1.write("\\hline \n")   
    types = ["uncorrected", "regional sky corr.", "annul. sky corr.",
             "$|\\text{uncorrected}|$", "$|\\text{regional sky corr.}|$", 
             "$|\\text{annul. sky corr.}|$"]
    aperdata = np.round([bckgOPT_old, bckgOPT_reg, bckgOPT_an, 
                         bckgOPT_oldABS, bckgOPT_regABS, bckgOPT_anABS], 2)
    aperdata_err = np.round([bckgOPT_old_err, bckgOPT_reg_err, bckgOPT_an_err,
                             bckgOPT_old_errABS, bckgOPT_reg_errABS, bckgOPT_an_errABS], 2)
    for i in range(6):
        savefile1.write("{A} \t&\t ${B}\\pm{Berr}$ \t&\t ${C}\\pm{Cerr}$ \t&\t ${D}$\\\\ \n".format(A=types[i], B=aperdata[i][0], Berr=aperdata_err[i][0], 
     C=aperdata[i][1], Cerr=aperdata_err[i][1],
     D=aperdata[i][2])) #Derr=aperdata_err[i][2]))   















