import numpy as np
from astropy.io import fits
import shutil
import os

from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
            if pix_val > countmax:
                countmax = pix_val
                center = [x,y]
                
    return center
                
            
    
# Function which calculates the aperture count rate for a star centered at pixel coordinates [px, py] for an aperture radius r (sky annulus subtraction not included).
def apersum_old(image, px, py, r):
    
    
    # Determine the aperture limits
    ny, nx = image.shape
    apx_min = max(1, px - r)
    apx_max = min(nx, px + r)
    apy_min = max(1, py - r)
    apy_max = min(ny, py + r)

    
    # Compute the total count rate within the aperture
    apsum = 0.0
    for i in range(apx_min, apx_max+1):
        for j in range(apy_min, apy_max+1):
            
            # Calculate squared distance to central pixel
            dx = i - px
            dy = j - py
            d2 = dx**2 + dy**2
            
            # Store the current pixel's count value
            pixval = image[j-1,i-1]
            
            if d2 <= r**2:
                apsum += pixval
                
    return apsum
    
    

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
    itnos = 8
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
        
    # Compute and return calibrated aperture flux
    apscal = apsum - apcount*av_an          
    return apscal
                       
    

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



# Function which embeds a data array in a larger (nrow,ncol)-frame, in order to correctly overlap images. Frameshape should be specified with its first element indicating the number of rows and its second element indicating the number of columns! Conversely, offset should be specified with its first element (X) indicating the column-wise offset and its second element (Y) indicating the row-wise offset! 'Cornerpix' define the pixel indices corresponding to the frame position where the upper left corner of the data array should be placed.
def embed(data, frameshape, offset=np.zeros(2), cornerpix=[0,0]):
    
    # Determine frame limits and create frame accordingly
    nrow, ncol = frameshape[0], frameshape[1]
    frame = np.zeros([nrow,ncol])
    
    # Determine data shape
    nrow_dat, ncol_dat = data.shape[0], data.shape[1]
    
    # Embed the data array in the frame, using the specified offset. NOTE: x=columns and y=rows!
    frame[cornerpix[0]+offset[1]:cornerpix[0]+offset[1]+nrow_dat, 
          cornerpix[1]+offset[0]:cornerpix[1]+offset[0]+ncol_dat] = data
    
    return frame



# Function which cuts each image into approximate slits using the first derivative along Y
def cut_to_slits(slitdata):
    chipdata = slitdata[10:934:,183:1868]
    rowsno = chipdata.shape[0]
    
    
    # Read out chipdata row for row and cut out slits
    derivs, slits, slitwidths = [], [], []
    agap, cutend, gapwidths = True, 0, []
    itnos = np.arange(1, rowsno-2, 1)
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
                    slitwidths.append(slit.shape[0])
                    gapwidths.append( cutstart - prevcutend ) # [pixels]
                        
                    
                    # Diagnostic plot
                    '''
                    plt.figure(0)
                    norm = ImageNormalize(stretch=SqrtStretch())
                    plt.imshow(slit, cmap='afmhot', origin='lower', norm=norm)
                    plt.colorbar()
                    plt.show()
                    plt.close()
                    '''
            
            except NameError:
                print("first max")
    
    return slits, slitwidths, gapwidths
    # Diagnostic plot
    '''
    plt.figure(1)
    plt.scatter(itnos, derivs)
    plt.show()
    plt.close()
    '''
    

    
# Function which computes O - E for all slits created via cut_to_slits and inserts the results into the array 'frame' in order to recreate a single image of the sky
def append_slits(slits, slitwidths, gapwidths, starpos_qca):

    # Compute number of slits
    nrofslits = len(slits)

    for n in np.arange(0, nrofslits, 2):
        
        '''
        # Find the star positions on each slit
        yO_qa = ( starpos_qca[:,0,1] - np.sum(slitwidths[n+1::]) 
                                     - np.sum(gapwidths[n+1::]) )
        yE_qa = ( starpos_qca[:,1,1] - np.sum(slitwidths[n+2::]) 
                                     - np.sum(gapwidths[n+2::]) )
        '''
                                     
        centers_qca = np.zeros([len(starpos_qca),2,2])
        for q, approxX in enumerate(starpos_qca[0,:,0]):
            
            approxY_O, approxY_E = np.argmax(slits[n][:,approxX]), np.argmax(slits[n+1][:,approxX])
            centers_qca[q,0] = find_center([approxX,approxY_O], slits[n], 15)
            centers_qca[q,1] = find_center([approxX,approxY_E], slits[n+1], 15)
            
            '''
            centers_qca[q,0] = find_center([starpos_qca[q,0,0],approxY_c[0]], slits[n], 15)
            centers_qca[q,1] = find_center([starpos_qca[q,1,0],approxY_c[1]], slits[n+1], 15)
            '''
        # Determine stellar position offsets compared to slit O
        offsets_qa = np.diff(centers_qca, axis=1)
        offsets = np.mean(offsets_qa, axis=0).astype(int)[0]
        
        
        
        # Embed 
        framesize = 1.5*np.array(slits[0].shape)
        cornerO = ((framesize - slits[n].shape) / 2).astype(int)
        cornerE = ((framesize - slits[n].shape) / 2).astype(int)
        frameO = embed(slits[n], framesize, cornerpix=cornerO)
        frameE = embed(slits[n+1], framesize, offset=offsets+[0,3], cornerpix=cornerE) 
        
        

        # Compute the normalized difference between the ordinary and extroardinary slit (or the other way arround?)
        slit_diff = frameO - frameE
        slit_sum = frameO + frameE
        cal_slit =  slit_diff / slit_sum
        if n == 0:
            cal_slits = cal_slit
        else:
            cal_slits = np.concatenate((cal_slits, cal_slit), axis=0)
        
             
        '''
        # Check that each slit contains the same number of pixel rows
        if slits[n+1].shape[0] < slits[n].shape[0]:
            newxlen = slits[n+1].shape[0]
            slits[n] = slits[n][0:newxlen, :]
        
        elif slits[n+1].shape[0] > slits[n].shape[0]:
            newxlen = slits[n].shape[0]
            slits[n+1] = slits[n+1][0:newxlen, :]
        
        print("newxlen:\t\t{}".format(newxlen))
        
        
        
        # Select a region of E which is to be subtracted from O taking into account the offset
        sel_slitO, sel_slitE = slits[n], slits[n+1]
        print(sel_slitO.shape, sel_slitE.shape)
        for a, offset in enumerate(offsets[0]):
            print(offsets)
            if a==0 and offset >= 0:
                sel_slitE = sel_slitE[offset::, :]
                print(sel_slitE.shape)
            elif a==0 and offset < 0:
                sel_slitE = sel_slitE[0:(sel_slitE.shape[1]+offset+1) , :]
                print(sel_slitE.shape)
            elif a==1 and offset >= 0:
                sel_slitE = sel_slitE[:, offset::]
                print(sel_slitE.shape)
            elif a==1 and offset < 0:
                sel_slitE = sel_slitE[:, 0:(sel_slitE.shape[1]+offset+1)]
                print(sel_slitE.shape)
        
        print("POEPIEEEEK\t\t", sel_slitE.shape, sel_slitO.shape)
        # Compute the normalized difference between the ordinary and extroardinary slit (or the other way arround?)
        sel_shape = sel_slitE.shape
        sel_slitO = sel_slitO[0:sel_shape[0], 0:sel_shape[1]]
        slit_diff = sel_slitO - sel_slitE
        slit_sum = sel_slitO + sel_slitE
        print(slit_diff.shape, slit_sum.shape)
        cal_slit =  slit_diff / slit_sum
        if n == 0:
            cal_slit0 = cal_slit
            cal_slits = cal_slit0
        else:
            cal_slit = cal_slit[0:cal_slit0.shape[0]+1, :] # TODO TODO PERHAPS CHANGE TO EMBED AFTER ALL
            cal_slit = cal_slit[:, 0:cal_slit0.shape[1]+1]
            print("POEPIEK2\t\t:", cal_slit0.shape, cal_slit.shape)
            cal_slits = np.concatenate((cal_slits, cal_slit), axis=0)
        '''
        
        
    return cal_slits



#################### END FUNCTIONS FOR SLIT APPENDAGE ####################
#################### END FUNCTIONS #############################################





# Specify data and filename
datadir = "/home/bjung/Documents/Leiden_University/brp/data_red/calib_data"
scidatadir = datadir + "/sorted/NGC4696,IPOL"
sci_dirs = [scidatadir + "/CHIP1"]
testdata = datadir[0] + "/tpl8/FORS2.2011-05-04T01:31:46.334.fits"
# Directory for saving plots
plotdir = "/home/bjung/Documents/Leiden_University/brp/data_red/plots"

# Specify bias and masterflat
header, Mbias = extract_data(datadir + "/masterbias.fits")
header, Mflat_norm = extract_data(datadir + "/masterflats/masterflat_norm_FLAT,LAM_IPOL_CHIP1.fits")



# Aproximate coordinates of selection of stars within CHIP1 of 'Vela1_95' and 'WD1615_154'. Axis 0 specifiec the different sci_dirs; axis 1 specifies the different stars within the sci_dirs; axis 2 specifies the x, y1, y2 coordinate of the specific star (with y1 specifying the y coordinate on the upper slit and y2 indicating the y coordinate on the lower slit) and the aproximate stellar radius. NOTE: THE LAST LIST WITHIN AXIS1 IS A SKY APERTURE!!!
star_lsts = [[[335, 904, 807, 5], [514, 869, 773, 7], [1169, 907, 811, 5], [1383, 878, 782, 7], 
              [341, 694, 599, 10], [370, 702, 607, 11], [362, 724, 630, 5], [898, 709, 609, 8], 
              [1836, 707, 611, 6], [227, 523, 429, 6], [343, 492, 399, 5], [354, 494, 400, 12], 
              [373, 520, 413, 8], [537, 491, 392, 7], [571, 541, 446, 8], [1096, 510, 416, 5], 
              [1179, 530, 436, 8], [487, 320, 226, 7], [637, 331, 238, 6], [1214, 345, 252, 6], 
              [1248, 326, 233, 6], [1663, 308, 217, 9], [326, 132, 40, 5], [613, 186, 94, 10], 
              [634, 184, 91, 9], [642, 134, 41, 7], [838, 175, 82, 8], [990, 140, 48, 11], 
              [1033, 157, 65, 9], [1172, 147, 55, 7], [1315, 164, 71, 8], [1549, 164, 72, 13]]] 
star_lsts = np.array(star_lsts) # 35 stars in total (42-7)
# List specifying the (axis1) indices of the first stars on each slit
slit_divide = np.array([0, 4, 9, 17, 22])



# Range of aperture radii
r_range = np.arange(1, 16) #[pixels]

# Pixel scale
pixscale = 0.126 #[arcsec/pixel]

# Boolean variable for switchin on polarization computations of selected stars
compute_anew = False

# ESO given polarizations
VelaBV_PlPhi = [[0., 0.],[0., 0.]] # [-], [deg]
VelaBV_sigmaPlPhi = [[0., 0.],[0., 0.]] # [-], [deg]
WD1615BV_PlPhi, WD1615BV_sigmaPlPhi = [[0., None], [0., None]], [[0., None], [0., None]]
ESObvPLPHI = np.array([VelaBV_PlPhi, WD1615BV_PlPhi]) 
sigmaESObvPLPHI = np.array([VelaBV_sigmaPlPhi, WD1615BV_sigmaPlPhi])



# Compute fluxes and polarizations for selected stars in testdata and carry out slit appenditure
if compute_anew == True:
    compute_fluxlsts(sci_dirs, Mbias, Mflat_norm, star_lsts, r_range)



# Create plots
for i, sci_dir in enumerate(sci_dirs):
    print(sci_dir.split("/")[-2])
    
    loaddir = sci_dir.rsplit("/",2)[0] + "/loadfiles/" + sci_dir.rsplit("/",2)[1]
    regions = np.array(star_lsts[i])
    ESO_BV_PlPhi, ESO_BV_sigmaPlPhi = ESObvPLPHI[i], sigmaESObvPLPHI[i]
    
    
    
    # Indices for selecting the aperture radii corresponding to each star
    reg_ind, rad_ind = np.arange(0, len(regions), 1), regions[:,3]-1
    std_rad = rad_ind[0]+1
    # List of colors for distinguishing the B and V filter in the plots
    Bcolor_lst, b = np.tile('b', 14), 0
    Vcolor_lst, v = np.tile('v', 14), 0
    # Initiate indices for first encountered B and V exposures
    B1ind, V1ind = None, None
    
    
    
    # Load flux lists and filter list for plots
    OEF_jkqr, sigmaOEF_jkqr, filter_lst, pos_jkqca = load_lsts(loaddir)
    #OEF_jkqr, sigmaOEF_jkqr = np.nan_to_num(OEF_jkqr), np.nan_to_num(sigmaOEF_jkqr)
    '''
    # Set NAN values to 0
    sigmaOEF_jkqr = np.where(sigmaOEF_jkqr >= 0., sigmaOEF_jkqr, 0.)
    '''
    # Select correct aperture radii for each region
    OEF_jkq = np.array([ jkqr[:,:,reg_ind,rad_ind] for jkqr in OEF_jkqr])
    sigmaOEF_jkq = np.array([ sigma_jkqr[:,:,reg_ind,rad_ind] for sigma_jkqr in sigmaOEF_jkqr])
    
    
    # Compute Stokes Q and U as well as the linear polarization degree and angle and all corresponding error margins
    QUPphi_jqr, sigmaQUPphi_jqr = compute_pol(OEF_jkqr[2], sigmaOEF_jkqr[2])
    # Select correct aperture radii for each region
    QUPphi_jq = np.array([ jkqr[:,reg_ind,rad_ind] for jkqr in QUPphi_jqr])
    sigmaQUPphi_jq = np.array([ sigma_jq[:,reg_ind,rad_ind] for sigma_jq in sigmaQUPphi_jqr])
    
    
    
    # Initiate QUPphi0 where phi is going to be corrected for instrumental offsets
    QUPphi0_jqr, sigmaQUPphi0_jqr = QUPphi_jqr, sigmaQUPphi_jqr
    QUPphi0_jq, sigmaQUPphi0_jq = QUPphi_jq, sigmaQUPphi_jq
    
    
    
    # Initiation of the FOV distances for figure 4
    listshape = OEF_jkqr[0].shape
    regsdistOE, normfluxOE_jkq, normfluxerrOE_jkq = [], [], []
    
    
    # Tracks the number of skipped templates
    skips = 0
    # Boolean for checking when the first v_HIGH and b_HIGH filter are iterated through
    seenB, seenV = False, False
    
    
    # Create a list with all the template directories within sci_dirs
    [tpl_dirlst, tpl_flst] = mk_lsts(sci_dir)
    tplnamemask = np.array([(len(tpl_name) < 5) for tpl_name in tpl_dirlst])
    tpl_dirlst = np.concatenate( (np.sort(tpl_dirlst[tplnamemask]), 
                                  np.sort(tpl_dirlst[np.logical_not(tplnamemask)])) )
    tpl_dirlst = np.delete(tpl_dirlst, np.argwhere(tpl_dirlst=="appended"))
    
    
    # Compute the offsets which should be used for overlapping the templates
    offsets_jkqa = np.average(pos_jkqca - pos_jkqca[0], axis=3) # [pixels]
    offsets_jka = np.average(offsets_jkqa, axis=2) # [pixels]
    # Initiate list of dataframes
    framesize = (1084, 2098) # [pixels]
    frames_jk = np.zeros([len(tpl_dirlst), 4, framesize[0], framesize[1]]) # [pixels]
    
    
    
    #### Plots for all templates ####  
    for J, tpl_name in enumerate(tpl_dirlst): 
        print("\t", tpl_name)
        tpl_dir = sci_dir + '/' + tpl_name
        

        # Create a list with filenames of files stored within tpldir
        expdir_lst, expfile_lst = mk_lsts(tpl_dir)
            
                    
        # Skip 'appended' subdirectory
        if (tpl_name == "appended") or (len(expfile_lst) != 4):
            print("\t\t skipped")
            skips += 1
            continue
        j = J-skips
        
        
        # Skip tpl5 #TODO TODO TODO TODO
        if j == 4:
            continue        
        
        
        # Define plot parameters 
        filtermask = (filter_lst == filter_lst[j])  
        mark_lst = ['o','v']
        
        
        # Select the right plot colour and correct the polarization angle for instrumental offset
        if filter_lst[j] == "b_HIGH":
            # Check whether a label should be included in the plots
            if seenB == False:
                B1ind = j
                lbl = "b_HIGH"
            else:
                lbl = None
            # Select plot colour
            plotcolour = 'b'
            colour2 = Bcolor_lst[b]
            # Instrumental offset angle
            if i==0:
                offset = 1.54*2. # deg
            elif i==1:
                offset = 0. # deg
            # Select correct ESO given polarizations
            ESO_PL, ESO_PHI = ESO_BV_PlPhi[0,0], ESO_BV_PlPhi[0,1]
            ESO_sigmaPL, ESO_sigmaPHI = ESO_BV_sigmaPlPhi[0,0], ESO_BV_sigmaPlPhi[0,1]
            
            seenB = True
            b += 1
        
        elif filter_lst[j] == "v_HIGH":
            # Check whether a label should be included in the plots
            if seenV == False:
                V1ind = j
                lbl = "v_HIGH"
            else:
                lbl = None
            # Select plot colour
            plotcolour = 'g'
            colour2 = Vcolor_lst[v]
            # Instrumental offset angle
            if i==0:
                offset = 1.8*2. # deg
            elif i==1:
                offset = 0. #deg
            # Select correct ESO given polarizations
            # Select correct ESO given polarizations
            ESO_PL, ESO_PHI = ESO_BV_PlPhi[1,0], ESO_BV_PlPhi[1,1]
            ESO_sigmaPL, ESO_sigmaPHI = ESO_BV_sigmaPlPhi[1,0], ESO_BV_sigmaPlPhi[1,1]
            
            seenV = True
            v += 1
        
        
        
        # Correct for instrumental offset
        QUPphi0_jq = np.where(QUPphi_jq != QUPphi_jq[3,j,0], QUPphi0_jq, QUPphi0_jq - offset)
        QUPphi0_jqr = np.where(QUPphi_jqr != QUPphi_jqr[3,j,0,:], QUPphi0_jqr, QUPphi0_jqr - offset)
        
        
        
        # Create a list with filenames of files stored within tpldir
        expdir_lst, expfile_lst = mk_lsts(tpl_dir + '/corrected2')
        expfile_lst = np.sort(expfile_lst)    
        
        
        
        # Append slits for all exposures # TODO TODO TODO TODO TODO TODO TODO
        for k, f in enumerate(expfile_lst):
            print("\t\t", f)
            header, datacor = extract_data(tpl_dir + '/corrected2/' + f)  
            slits, slitw, gapw = cut_to_slits(datacor)
            cal_slits = append_slits(slits, slitw, gapw, pos_jkqca[j,k]) 
            
            
            fig1 = plt.figure(1)
            ax1 = fig1.add_subplot(111)
            norm = ImageNormalize(stretch=SqrtStretch())
            image1 = ax1.imshow(cal_slits, cmap='afmhot', origin='lower', norm=norm)        
            plt.colorbar(image1)       
            plt.show()     
            
            
        
        
        
        ############## PLOT 0 ##############     
        # Plot of aperture radius vs degree of linear polarization for the standard star
        fig0 = plt.figure(0)
        ax0 = fig0.add_subplot(111)
        print(np.mean(QUPphi0_jqr[2,j,30]))
        if np.mean(np.nan_to_num(QUPphi0_jqr[2,j,30])) > 0.2:
            print("POEPIEEK:\t\t{}".format(tpl_name))
            plotcolour2 = 'r' #TODO TODO TODO SKIPPED TEMPLATE 5
            apRvsPl(ax0, r_range, pixscale, 
                QUPphi0_jqr[2,j,30], sigmaQUPphi0_jqr[2,j,30], 
                colour=plotcolour2, tag=" tpl{}".format(j))
        else:
            plotcolour2 = plotcolour
            apRvsPl(ax0, r_range, pixscale, 
                QUPphi0_jqr[2,j,30], sigmaQUPphi0_jqr[2,j,30], 
                colour=plotcolour2, tag=lbl)
        ############## PLOT 0 ##############
        
        
        
        
        ############## PLOT 1 ##############     
        # Plot of aperture radius vs degree of linear polarization for the standard star
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        norm = ImageNormalize(stretch=SqrtStretch())
        image1 = ax1.imshow(np.log(frametot), cmap='afmhot', origin='lower', norm=norm)        
        plt.colorbar(image1)
        ############## PLOT 1 ##############
        
        
        
        
        ############## PLOT 5 ##############     
        # Vector plot showing polarization degrees of all regions (Only executed for the first exposure of the last template)
        
        #with sns.axes_style("whitegrid",{'axes.grid' : False}):
        fig5 = plt.figure(5)
        ax5 = fig5.add_subplot(111)
        # Load image
        vecplotdir = sci_dir + '/' + tpl_name + '/corrected2/'
        datadirs, datafiles = mk_lsts(vecplotdir)
        header, datacor = extract_data(vecplotdir + datafiles[0])
        # Compute vectors
        vectorX, vectorY = regions[:,0], regions[:,1]
        vectorU = QUPphi0_jq[2,j,:] * np.cos(QUPphi0_jq[3,j,:]*np.pi/180.)
        vectorV = QUPphi0_jq[2,j,:] * np.sin(QUPphi0_jq[3,j,:]*np.pi/180.)
        M = np.hypot(vectorU, vectorV)
        # Plot image
        norm = ImageNormalize(stretch=SqrtStretch())
        image5 = ax5.imshow(np.log(datacor), cmap='afmhot', origin='lower', norm=norm)
        ##########plt.colorbar(image)
        # Plot vectors
        Q = ax5.quiver(vectorX, vectorY, 
                       vectorU, vectorV, 
                       M, units='xy', pivot='mid')
                       #color = '#666699')
    qk = ax5.quiverkey(Q, 0.15, 0.9, 5e-3, r'$0.5\%$ polarization', labelpos='N',
                       coordinates='figure', fontproperties={"size": 15})
    plt.colorbar(image5)
    ############## PLOT 5 ##############  
    
    
       

    
    # Check if the savefile directory is present
    savedir = plotdir + '/' + sci_dir.split("/")[-2]
    if not os.path.exists(savedir):
        os.chdir(plotdir)
        os.makedirs(savedir)
        os.chdir(scidatadir)
    
    
    plt.figure(0)
    plt.grid()
    plt.xlabel(r'$\mathrm{Radius \ [arcsec]}$', fontsize=20)
    plt.ylabel(r'$\mathrm{Degree \ of \ linear \ polarization \ [-]}$', fontsize=20)
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.savefig(savedir + '/' + 'RvsPl')
    
    
    '''
    plt.figure(1)
    x_tickrange, y_tickrange = np.arange(0,2038,200), np.arange(1000,-1,-100)
    plt.title("Median image")
    plt.xticks(x_tickrange, (x_tickrange-1000)*pixscale), plt.yticks(y_tickrange,  (y_tickrange-500)*pixscale)
    plt.xlabel(r'X [arcsec]', fontsize=20)
    plt.ylabel(r'Y [arcsec]', fontsize=20)
    plt.tight_layout()
    plt.savefig(savedir + '/' + 'totimage') 
    '''
        
    
    plt.figure(5)
    x_tickrange, y_tickrange = np.arange(0,2038,200), np.arange(1000,-1,-100)
    plt.xticks(x_tickrange, (x_tickrange-1000)*pixscale), plt.yticks(y_tickrange,  (y_tickrange-500)*pixscale)
    plt.xlabel(r'X [arcsec]', fontsize=20)
    plt.ylabel(r'Y [arcsec]', fontsize=20)
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.savefig(savedir + '/' + 'polprofile')
    
    
    # Show and close figures
    plt.show()
    plt.close()
    



'''
# Read in files
tpl_dlst, tpl_flst = mk_lsts(datapath)
for tpl in tpl_dlst:
    print("\t{}".format(tpl))
    

    
    for exp in exp_flst:
        print("\t\t{}".format(exp))
    
        
        
        header, data = extract_data(datapath + '/' + tpl + '/' + exp)
        datacal = (data - Mbias) / Mflat_norm # Calibrated data
        # Append slits
        cal_slits = append_slits(datacal, pixoffs=np.tile(2, 5))
'''
        
'''
        # Save to fits file
        savedir = datapath + "/appended/" + tpl
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        hdu = fits.PrimaryHDU(cal_slits)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(savedir + "/{}.fits".format(exp.split(".fits")[0] + "_APPENDED"))
'''



############### PLOTS #################


'''
plt.figure()
ax = plt.gca()
norm = ImageNormalize(stretch=SqrtStretch())  
im = ax.imshow(cal_slits, cmap='Greys', origin='lower', norm = norm)
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)

plt.tight_layout()
plt.show()    
'''









