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
    '''# Diagnostic plot    
    
    
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
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    dat1 = ax1.imshow(slitdiff, origin='lower', cmap="afmhot")
    plt.colorbar(dat1)
    dat2 = ax2.imshow(grady, origin='lower', cmap="afmhot")
    plt.colorbar(dat2)
    plt.show()
    plt.close()
    
    plt.figure()
    #plt.plot(np.arange(0,O.shape[0]), E[:,center[0]], color="y", marker='s', label=r"$E$")
    #plt.plot(np.arange(0,O.shape[0]), O[:,center[0]], color="g", marker='o', label=r"$O$")
    #print(np.max((E-O)[:,center[0]], np.max(grady[:,center[0]])))
    O_E, Ogrady = (O-E)[:,center[0]], grady[:,center[0]]
    normval1 = np.max(np.abs(O_E[~np.isnan(O_E)]))
    normval2 = np.max(np.abs(Ogrady[~np.isnan(Ogrady)]))
    print(O_E[~np.isnan(O_E)])
    plt.plot(np.arange(0,O[~np.isnan(O_E)].shape[0]), O_E[~np.isnan(O_E)]/normval1, color="r", marker='x', linestyle='--', label=r"$O-E$")
    plt.plot(np.arange(0,O[~np.isnan(Ogrady)].shape[0]), Ogrady[~np.isnan(Ogrady)]/normval2, color="b", marker='v', label=r"$\nabla_y (O)$")
    plt.legend(loc='best')
    plt.xlabel("Pixel", fontsize=20), plt.ylabel("Normalized Counts [---]", fontsize=20)
    plt.title("Stellar y-profile", fontsize=20)
    plt.savefig(pltsavedir+"grady_profile2")
    plt.show()
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
        plt.show()
        plt.close()
        
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
    


'''
# Function which computes O - E for all slits created via cut_to_slits and inserts the results into the array 'frame' in order to recreate a single image of the sky
def align_slits_wrong(slits, pixoffs):

    # Compute number of slits
    nrofslits = len(slits)
    ny = min(slitwidths)
    
    # Determine the centers of the stars of each slit
    for n in np.arange(0, nrofslits, 2):
        if n==8:
            slitOEstarpos_qca = starpos_qca[slit_div[n/2]::]
        else:    
            slitOEstarpos_qca = starpos_qca[slit_div[n/2]:slit_div[n/2+1]]
            
        centers_qca = np.zeros([len(slitOEstarpos_qca),2,2])
        for q, approxX in enumerate(slitOEstarpos_qca[:,0,0]):
            
            approxY_O, approxY_E = np.argmax(slits[n][:,approxX]), np.argmax(slits[n+1][:,approxX])
            centers_qca[q,0] = find_center([approxX,approxY_O], slits[n], 20)
            centers_qca[q,1] = find_center([approxX,approxY_E], slits[n+1], 20)
            
            print( centers_qca[q,0] , centers_qca[q,1] )
            
        print("centers_qca.shape\t\t", centers_qca.shape)
            
        # Determine stellar position offsets compared to slit O
        offsets_qa = np.diff(centers_qca, axis=1)
        print(offsets_qa.shape)
        offsets = np.mean(offsets_qa, axis=0).astype(int)[0]
        offsety = offsets[1]
        print(ny, offsety)
'''    
        
        


#################### END FUNCTIONS FOR SLIT APPENDAGE ####################
#################### END FUNCTIONS #############################################





# Specify data and filename
datadir = "/home/bjung/Documents/Leiden_University/brp/data_red/calib_data"
# Specify standard star directories
stddatadir = datadir + "/sorted/STD,IPOL"
std_dirs = [stddatadir + "/Vela1_95/CHIP1", stddatadir + "/WD1615_154/CHIP1"]
teststddata = [std_dirs[0] + "/tpl3/FORS2.2011-05-04T00:05:36.569.fits", # RETA POSANG 45 deg
               std_dirs[1] + "/tpl2/FORS2.2011-05-04T05:33:58.533.fits"] # RETA POSANG 45 deg
# Specify science data directories
scidatadir = datadir + "/sorted/NGC4696,IPOL"
sci_dirs = [scidatadir + "/CHIP1"]
testscidata = sci_dirs[0] + "/tpl8/corrected2/FORS2.2011-05-04T01:31:46.334_COR.fits" # RETA POSANG 45 deg # j=7, k=1
# Combine data dirs in list
testdata_fnames = [teststddata[0], teststddata[1], testscidata]
# Load testdata
headerVELA, dataVELA = extract_data(teststddata[0])
headerWD, dataWD = extract_data(teststddata[1])
headerNGC, dataNGC = extract_data(testscidata)
# Directory for saving plots
plotdir = "/home/bjung/Documents/Leiden_University/brp/data_red/plots"
imdir = "/home/bjung/Documents/Leiden_University/brp/data_red/images"
npsavedir = "/home/bjung/Documents/Leiden_University/brp/data_red/npsaves"

# Specify bias and masterflat
header, Mbias = extract_data(datadir + "/masterbias.fits")
header, Mflat_norm = extract_data(datadir + "/masterflats/masterflat_norm_FLAT,LAM_IPOL_CHIP1.fits")



# Aproximate coordinates of selection of stars within CHIP1 of 'Vela1_95' and 'WD1615_154'. Axis 1 specifies the different stars within the std_dir; axis 2 specifies the x, y1, y2 coordinate of the specific star (with y1 specifying the y coordinate on the upper slit and y2 indicating the y coordinate on the lower slit) as well as the aproximate stellar radius and the slit pair number (numbered 1 to 5 from lower to upper slit pair) which the star lies on. NOTE: THE LAST LIST WITHIN AXIS1 IS A SKY APERTURE!!!
star_lst_sci = [[335, 904, 807, 5, 5], [514, 869, 773, 7, 5], [1169, 907, 811, 5, 5], 
                [1383, 878, 782, 7, 5], [341, 694, 599, 10, 4], [370, 702, 607, 11, 4], 
                [362, 724, 630, 5, 4], [898, 709, 609, 8, 4], [1836, 707, 611, 6, 4], 
                [227, 523, 429, 6, 3], [354, 498, 404, 10, 3], [376, 512, 418, 8, 3], 
                [419, 525, 431, 7, 3], [537, 491, 392, 7, 3], [571, 541, 446, 8, 3], 
                [1096, 510, 416, 5, 3], [1179, 530, 436, 8, 3], [487, 320, 226, 7, 2], 
                [637, 331, 238, 6, 2], [1214, 345, 252, 6, 2], [1248, 326, 233, 6, 2], 
                [1663, 308, 217, 9, 2], [326, 132, 40, 5, 1], [613, 186, 94, 10, 1], 
                [634, 184, 91, 9, 1], [642, 134, 41, 7, 1], [838, 175, 82, 8, 1], 
                [990, 140, 48, 11, 1], [1033, 157, 65, 9, 1], [1172, 147, 55, 7, 1], 
                [1315, 164, 71, 8, 1], [1549, 164, 72, 13, 1]] #[pixel] # 32 stars
star_lst_stdVELA = [[1034, 347, 251, 15, 2], [1177, 368, 273, 8, 2], [319, 345, 250, 5, 2], [281, 499, 403, 6, 3], [414, 139, 45, 12, 1], [531, 706, 609, 5, 4], [1583, 322, 229, 3, 2], [1779, 321, 224, 4, 2], [1294, 725, 627, 4, 4], [1501, 719, 622, 7, 4]] #[pixel] # 10 stars
star_lst_stdWD = [[1039, 347, 253, 12, 2], [599, 541, 446, 5, 2], [365, 700, 604, 5, 4], [702, 903, 806, 6, 5], [801, 136, 43, 4, 1], [1055, 133, 43, 4, 1], [1186, 130, 37, 4, 1], [1132, 685, 592, 3, 4], [1222, 685, 592, 4, 4], [1395, 679, 587, 4, 4], [1413, 912, 816, 5, 5], [1655, 542, 449, 5, 3], [1643, 512, 417, 5, 3], [1632, 190, 97, 6, 1], [1608, 178, 85, 4, 1]] #[pixel] # 17 stars           
# Combine star lists
star_lsts = [star_lst_stdVELA, star_lst_stdWD, star_lst_sci]


# Define the x- and y-ranges corresponding to the chip
chip_xyranges = [[183,1868],[25,934]]
# Range of aperture radii
r_range = np.arange(1, 16) #[pixels]
# Pixel scale
pixscale = 0.126 #[arcsec/pixel]
# Boolean variable for switchin on polarization computations of selected stars
compute_anew = False
calc_cd, calc_well = True, False



# Compute fluxes and polarizations for selected stars in testdata and carry out slit appenditure
if compute_anew == True:
    compute_fluxlsts(sci_dirs, Mbias, Mflat_norm, star_lsts, r_range)



# Cut and append slits
slitsNGC, upedges, lowedges, gapw, slitshapes = cut_to_slits(dataNGC)
slitsNGC = [slitsNGC[i][0:np.min(slitshapes[:,0]),
                        0:np.min(slitshapes[:,1])] for i in range(0,10,1)] # Enforce same x-shape
slitsVELA = [dataVELA[lowedges[i]:upedges[i],
                      chip_xyranges[0][0]:chip_xyranges[0][0]+np.min(slitshapes[:,1])] 
             for i in range(len(lowedges))]
slitsWD = [dataWD[lowedges[i]:upedges[i],
                  chip_xyranges[0][0]:chip_xyranges[0][0]+np.min(slitshapes[:,1])]
           for i in range(len(lowedges))]
# Append to list
slits_lst = [slitsVELA, slitsWD, slitsNGC]



# Determine slitwidths
upedges, lowedges, gapw = [np.array(temp) for temp in [upedges, lowedges, gapw]]
slitwidths = upedges-lowedges





'''
####################### EVALUATION OF DIFFERENT INTERPOLATION METHODS #######################
# 1d spline #TODO
tck_1d = interpolate.splrep(xpoints, testslitO[27], s=0)
interpO3_1d = interpolate.splev(interpx, tck_1d, der=0)
# Compute the relative interpolation errors
diff1_norm = (interpO[270,xpoints*10] - testslitO[27]) / testslitO[27]
diff3_norm = (interpO3_1d[xpoints*10] - testslitO[27]) / testslitO[27] 
# PLOT
plt.figure()
plt.plot(xpoints, diff1_norm, 'r', label='2D cubic spline')
plt.plot(xpoints, diff3_norm, 'g', label='1D cubic spline')
plt.legend(loc='best')
plt.xlabel(r"X [pixels]", fontsize=20), plt.ylabel(r"Relative Error [--]", fontsize=20)
plt.title("Interpolation Errors")
plt.savefig(plotdir + "/interp/interpErrs.png")
plt.show()
plt.close()
####################### END evaluation of different interpolation methods #######################
''' # TODO EVALUATE MORE FORMS OF INTERPOLATION





# Initialize lists for storing results as well as the non-interpolated offset ranges
dxrange, dyrange, interpf = np.arange(-6,7,1), np.arange(-6,7,1), 5
optpixoffsets, wells, interpslits = [[],[]], [[],[]], []
# Lists for storing the optimal Q and c-/d-values
Qopts, opts_cd = [], []
cscape = np.tile(np.nan, np.array(Mbias.shape))
dscape = np.tile(np.nan, np.array(Mbias.shape))

# Compute the offset wells for all stars
for testdata_fname, slits, star_lst in zip(testdata_fnames, slits_lst, star_lsts):
    
    
    # Define plot save directories
    temp = testdata_fname.split(datadir+"/sorted/")[1]
    pltsavedir = plotdir +"/"+ temp.split("/")[0]
    imsavedir = imdir +"/"+ temp.split("/")[0]
    if pltsavedir.split("/")[-1] == "STD,IPOL":
        pltsavedir = plotdir +"/"+ temp.split("/")[1]
        imsavedir = imdir +"/"+ temp.split("/")[1]
            
    
    # Extract data
    header, data = extract_data(testdata_fname)
    # De-biasing and flat-fielding corrections
    data = (data - Mbias) / Mflat_norm
    

    # Define aperture and anulus radii
    R, anRmin, anRmax = starpar[3], 1.2*starpar[3], 2*starpar[3]

    
    # Determine c- and d-values of all stars
    for starno, starpar in enumerate(star_lst):
            
        
        # Determine
        if testdata_fname == testscidata:
            # Load list with the filenames of the non-interpolated corrected slits
            aligneddirs, alignedfiles = mk_lsts(imdir+"/offsetopt/noninterp4")
            aligneddatadict = {}
            # Read out the offsets for each file
            optpixoffsets = np.zeros([32,2])
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
                optpixoffsets[starno-1] = [dx,dy]   
            np.save(npsavedir+"/"+imsavedir.split("/")[-1]+"/optpixoffsets", optpixoffsets[0])
            
        
        
        # Check whether to compute cdscapes
        elif calc_well == False:
            break
        print("\n\nComputing offset wells...")
        print("\n\nStarno:\t\t{}".format(starno+1))
        
        
        
        # Extract ordinary and extraordinary slit
        slitEnr = 2*(starpar[4]-1)
        slitE, slitO = slits[slitEnr], slits[slitEnr + 1]
        '''
        # Adjust shape so that O and E have equal size
        Nx, Ny = min(slitE.shape[1], slitO.shape[1]), min(slitE.shape[0], slitO.shape[0])
        slitE, slitO = slitE[0:Ny,0:Nx], slitO[0:Ny,0:Nx]
        ''' #TODO REMOVE IF POSSIBLE
        '''
        # Determine slit interpolations
        interpE = interp(slitE, interpf*slitE.shape[1], interpf*slitE.shape[0])
        interpO = interp(slitO, interpf*slitO.shape[1], interpf*slitO.shape[0])
        interpslits.append(interpE), interpslits.append(interpO)
        '''
        # Determine the upper and lower edges of the slits
        upedgeE, upedgeO = upedges[slitEnr], upedges[slitEnr+1]
        lowedgeE, lowedgeO = lowedges[slitEnr], lowedges[slitEnr+1]
        print("Slit pair {}".format(slitEnr/2))
        
        
        # Compute stellar location on O slit
        slitOcent = find_center([starpar[0]-chip_xyranges[0][0], starpar[1]-lowedgeO],
                                 slitO, 15)
        
        
        # Diagnostic plots
        '''
        plt.figure()
        plt.imshow(np.log(slitO), origin='lower')
        plt.colorbar()
        plt.scatter(slitOcent[0], slitOcent[1], c='k', s=50)
        plt.scatter(starpar[0]-chip_xyranges[0][0], starpar[1]-lowedgeO, c='k', s=50)
        plt.show()
        plt.close()
        '''
        # IS OK #TODO POSSIBLY REMOVE
        
        
        # Compute wells for original slits
        offsetopt, well, alignedim_well = offsetopt_well([slitE,slitO], dxrange, dyrange, 
                                                         slitOcent, R, anRmin, anRmax,
                                                         cutoutR=20, saveims=True, 
                                                         pltsavedir=pltsavedir,
                                                         imsavedir=imsavedir,
                                                         starno=starno)
        optpixoffsets[0].append(offsetopt), wells[0].append(well)
        
        
        '''
        plt.figure()
        plt.imshow(alignedim_well, origin='lower')
        plt.colorbar()
        plt.scatter(slitOcent[0], slitOcent[1], c='k', s=50)
        plt.show()
        plt.close()
        '''
        
    
    # Save the stellar pixel-accurate offsets and the stellar offset wells to np savefiles
    if calc_well:
        savenp(wells[0], npsavedir+"/"+imsavedir.split("/")[-1], "wells")
        savenp(optpixoffsets[0], npsavedir+"/"+imsavedir.split("/")[-1], "optpixoffsets")     
    
    
    
    '''
    # Set interpolated offset parameters
    dxrange_interp = np.arange(offsetopt[0]-20, offsetopt[0]+20, 2)
    dyrange_interp = np.arange(offsetopt[1]-20, offsetopt[1]+20, 2)
    # Compute wells for interpolated slits
    [INTERPoffsetopt, 
    INTERPwell, 
    INTERPalignedim_well] = offsetopt_well([interpE,interpO],
                                           dxrange_interp, dyrange_interp, 
                                           interpOcent, interpf*R, interpf*anRmin,
                                           interpf*anRmax,
                                           cutoutR=interpf*20, saveims=True, 
                                           pltsavedir=pltsavedir,
                                           imsavedir=imsavedir,
                                           starno=starno)
    optoffsets[1].append(INTERPoffsetopt), wells[1].append(INTERPwell)
    '''
    '''
    # Align interpolated slits using the mean of calculated offsetopts for the stars on each slit
    if calc_well:
        
        # Average over the computed offsets for each slit
        meanoffs, prevdiv = [], 0
        for n, slitdiv in enumerate(slit_divide): #TODO TODO CHANGE slit_divide to starpar[4]
        
            # Compute mean offset for slit
            meanoff = np.mean( np.array(optoffsets[1])[prevdiv:slitdiv-1], axis=0)
            meanoffs.append(meanoff)
        
        # Align the slits using the mean offsets over each slit
        totim_interp = align_slits(slits, meanoffs, detoffs=False)
        
        # Save result
        savefits(totim_interp, imsavedir, "aligntot_allslits")
    '''
    
        
          
    # Determine c- and d-values of all stars
    for starno, starpar in enumerate(star_lst):
        
        # Check whether to compute cdscapes
        if calc_cd == False:
            break
        print("\n\nComputing c- and d-scapes...")
        print("\n\nStarno:\t\t{}".format(starno+1))
        
        
        # Load the pixel-accurate offsets
        offsets = np.load(npsavedir+"/"+imsavedir.split("/")[-1]+"/optpixoffsets.npy")
        
        
        # Extract ordinary and extraordinary slit
        slitEnr = 2*(starpar[4]-1)
        slitE, slitO = slits[slitEnr], slits[slitEnr + 1]
        '''
        # Adjust shape so that O and E have equal size
        Nx, Ny = min(slitE.shape[1], slitO.shape[1]), min(slitE.shape[0], slitO.shape[0])
        slitE, slitO = slitE[0:Ny,0:Nx], slitO[0:Ny,0:Nx]
        ''' #TODO REMOVE IF POSSIBLE
        '''
        # Determine slit interpolations
        interpE = interp(slitE, interpf*slitE.shape[1], interpf*slitE.shape[0])
        interpO = interp(slitO, interpf*slitO.shape[1], interpf*slitO.shape[0])
        interpslits.append(interpE), interpslits.append(interpO)
        '''
        # Determine the upper and lower edges of the slits
        upedgeE, upedgeO = upedges[slitEnr], upedges[slitEnr+1]
        lowedgeE, lowedgeO = lowedges[slitEnr], lowedges[slitEnr+1]
        print("Slit pair {}".format(slitEnr/2))
        
            
        
        # Compute stellar location on O slit and on the template appended slit
        slitOcent = find_center([starpar[0]-chip_xyranges[0][0], starpar[1]-lowedgeO],
                                 slitO, 15)
        dataOcent = find_center([starpar[0], starpar[1]],
                                 data, 15)
        
        '''
        appendedOcent = find_center([slitOcent[0], 
                                     slitOcent[1]+np.sum(slitwidths[[m for m in np.arange(0,n)]])],
                                     aligntemp, 25) #TODO SAVE TO NP FILE
        print("appendedOcent:\t\t", appendedOcent)
        
        plt.figure()
        plt.imshow(data, origin='lower', cmap='rainbow')
        plt.colorbar()
        plt.scatter(appendedOcent[0], appendedOce[1], s=30, c='k')
        plt.show()
        plt.close()
        '''
        
        
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
        
        
        # Determine the background fluxes in E and O
        backgrE, backgrEerr = ansum(embedE, embedOcent[0], embedOcent[1], 
                                    minAn_r=anRmin, maxAn_r=anRmax)
        backgrO, backgrOerr = ansum(embedO, embedOcent[0], embedOcent[1], 
                                    minAn_r=anRmin, maxAn_r=anRmax)
        # Correct the embeded images for the background flux
        embedOcorr = embedO - backgrO[2]
        embedEcorr = embedE - backgrE[2]
        
        
        # Diagnostic plot    
        '''
        plt.figure()
        plt.imshow(embedOcorr, origin='lower', cmap='rainbow')
        plt.colorbar()
        plt.scatter(embedOcent[0], embedOcent[1], s=30, c='k')
        plt.show()
        plt.close()
        
        plt.figure()
        plt.imshow(embedEcorr, origin='lower', cmap='rainbow')
        plt.colorbar()
        plt.scatter(embedOcent[0], embedOcent[1], s=30, c='k')
        plt.show()
        plt.close()
        '''
        
        
        # Save O-E
        savefits(embedOcorr-embedEcorr, imdir+"/offsetopt/cdscapes15/star{}".format(starno+1), 
                 "slitdiff_star{}".format(starno+1)) 
        savefits(cutoutO, imdir+"/offsetopt/cdscapes15/star{}".format(starno+1), 
                 "O_star{}".format(starno+1))       
        savefits(cutoutE, imdir+"/offsetopt/cdscapes15/star{}".format(starno+1), 
                 "E_star{}".format(starno+1)) 
        #TODO The allocation of the stellar center on the aligned image is OK
        
        
        # Recall previous c and d values for current star
        cval_prev, dval_prev = 0, 0
        '''
        dval_prev = dscape_prev[appendedOcent[1], appendedOcent[0]]
        ''' # TODO REMOVE
        for itno in range(10):
            
            # Define c-ranges
            cstart, cend = np.round([cval_prev - 2/(itno+1), cval_prev + 2/(itno+1)], 2)
            crange = np.linspace(cstart, cend, 49)
            cstep = np.round(crange[1]-crange[0], 3)
            print("\tcrange:\t", cstart, cend, cstep)
            # Define d-ranges
            dstart, dend = np.round([dval_prev - 2/(itno+1), dval_prev + 2/(itno+1)], 2)
            drange = np.linspace(dstart, dend, 49)
            dstep = np.round(drange[1]-drange[0], 3)
            print("\tdrange:\t", dstart, dend, dstep)
            
            
            # Compute the c and d parameters which optimize overlap using gradient method
            gradopt, Qopt, opt_cd = offsetopt_cd(embedOcorr, embedEcorr, crange, drange,
                                                 embedOcent, starpar[3],
                                                 savetofits=True, 
                                savedir=imsavedir+"/offsetopt/cdscapes15/star{}".format(starno+1), 
                                gradoptname="gradopt{}cstep{}dstep{}".format(itno, cstep, dstep))
            cval_prev, dval_prev = opt_cd
            print("Qopt, opt_cd:\t\t", Qopt, opt_cd)
            
            
        # Update c- and dscape
        cscape[dataOcent[1],dataOcent[0]] = opt_cd[0]+offsets[starno][0]
        dscape[dataOcent[1],dataOcent[0]] = opt_cd[1]+offsets[starno][1]
        
        # Append the best (i.e. last) optima parameters to list
        Qopts.append(Qopt), opts_cd.append(opt_cd)
    




# Determine bivariate third order polynomial fit to c- and dscapes if calc_cd==True
if calc_cd:
    # Determine x and y coordinates as well as the values of evaluated points
    c_xycoord, d_xycoord = np.argwhere(~np.isnan(cscape)), np.argwhere(~np.isnan(dscape))
    cpoints, dpoints = np.dstack(c_xycoord)[0], np.dstack(d_xycoord)[0]
    c_x, c_y, d_x, d_y = cpoints[1,:], cpoints[0,:], dpoints[1,:], dpoints[0,:]
    cval, dval = cscape[c_y, c_x], dscape[d_y, d_x]
    
    
    # Compute gridpoints for evaluation
    scapex, scapey = np.arange(0,cscape.shape[1],1), np.arange(0,cscape.shape[0],1)
    scape_xgrid, scape_ygrid = np.meshgrid(scapex, scapey)
    # Compute coordinates in arcseconds
    c_xarcs, c_yarcs = np.array([c_x - np.median(scapex), c_y]) * .126
    d_xarcs, d_yarcs = np.array([d_x - np.median(scapex), d_y]) * .126
    scapexarcs = (np.arange(0,cscape.shape[1],1) - np.median(scapex))*.126
    scapeyarcs = np.arange(0,cscape.shape[0],1)*.126
    scapexarcs_grid, scapeyarcs_grid = np.meshgrid(scapexarcs, scapeyarcs)
    # Third order univariate polynomial fit
    polynom_c = np.polyfit(c_x[cval>-3.], cval[cval>-3.], 2)
    polynom_d = np.polyfit(d_y[cval>-3.], dval[cval>-3.], 2)
    # Evaluate the derived polynomials
    polyval_c, polyval_d = np.polyval(polynom_c, ), np.polyval(polynom_d, d_x)
    polyfitdata_c = np.tile(polyval_c, [len(scapey),len(scapex)])
    polyfitdata_d = np.tile(polyval_d, [1,len(scapex)])
    
    
    # Save results
    savefits(polyfitdata_c, imdir+"/offsetopt/cdscapes16/cscapes", "cscapefitted_tpl8")
    savefits(polyfitdata_d, imdir+"/offsetopt/cdscapes16/dscapes", "dscapefitted_tpl8")
    
    
    # Save c and d polynomial fits
    if not os.path.exists(plotdir+"/cdscapes16/cscape"):
        os.makedirs(plotdir+"/cdscapes16/cscape")
    elif os.path.exists(plotdir+"/cdscapes16/cscape"):
        shutil.rmtree(plotdir+"/cdscapes16")
        os.makedirs(plotdir+"/cdscapes16")
    plt.imshow(polyfitdata_c, origin='lower', 
               extent=[scapexarcs[0],scapexarcs[-1],scapeyarcs[0],scapeyarcs[-1]])
    plt.scatter(c_xarcs, c_yarcs, c=cval)
    plt.colorbar()
    plt.title(r"c-scape")
    plt.savefig(plotdir+"/cdscapes16/cscape")
    plt.show()
    
    if not os.path.exists(plotdir+"/cdscapes16/dscape"):
        os.makedirs(plotdir+"/cdscapes16/dscape")    
    elif os.path.exists(plotdir+"/cdscapes16/dscape"):
        shutil.rmtree(plotdir+"/cdscapes16")
        os.makedirs(plotdir+"/cdscapes16/star{}")    
    plt.imshow(polyfitdata_d, origin='lower',
               extent=[scapexarcs[0],scapexarcs[-1],scapeyarcs[0],scapeyarcs[-1]])
    plt.scatter(d_xarcs, d_yarcs, c=dval)
    plt.colorbar()
    plt.title(r"d-scape")
    plt.savefig(plotdir+"/cdscapes16/dscape")
    plt.show()
        




# Save to fits
savefits(cscape, imdir+"/offsetopt/cdscapes16/cscapes","cscape_tpl8")
savefits(dscape, imdir+"/offsetopt/cdscapes16/dscapes", "dscape_tpl8")
# Save wells as numpy files

np.save(npsavedir+"/cscape", cscape), np.save(npsavedir+"/dscape", dscape)








