import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
import os
'''
from sets import Set
import aper
from astropy.stats import sigma_clipped_stats
'''
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
'''
from photutils import aperture_photometry
from photutils import DAOStarFinder
from photutils import CircularAperture
'''
import matplotlib.pyplot as plt





#################### FUNCTIONS ####################



# Returns a list will all subdirectories in a specific folder as well as the
# contained files
def mk_lsts(dir_name):

    #dir_lst = [dirs[0] for dirs in os.walk(dir_name)]
    dir_lst = next(os.walk(dir_name))[1]
    file_lst = next(os.walk(dir_name))[2]
    return dir_lst, file_lst



# Checks whether fname is a fits or reg file and extracts the corresonding header and data or coordinates and radius respectively
def extract_data(fname):

    # Put data and header files into two separate lists
    if fname.endswith("fit") or fname.endswith("fits"):
        hdul = fits.open(fname)
        header = hdul[0].header             
        data = hdul[0].data
        hdul.close()
        
    return header, data



# Define ratio G for calculating P_Q, P_U or P_V (see Bagnulo2009 p.997)
def G(counts_b0, counts_b90):
    
    r = counts_b0 / counts_b90
    G = (r -1) / (r + 1)
    
    return G
    
    

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
                
            
    
# Function which calculates the aperture count rate for a star centered at pixel coordinates [px, py] for an aperture radius r
def apersum(image, px, py, r):
    
    
    # Determine the aperture limits
    ny, nx = image.shape
    apx_min = max(1, px - r)
    apx_max = min(nx, px + r)
    apy_min = max(1, py - r)
    apy_max = min(ny, py + r)

    
    # Compute the total count rate within the aperture
    apsum = 0.0
    for i in range(apx_min, apx_max):
        for j in range(apy_min, apy_max):
            
            # Calculate squared distance to central pixel
            dx = i - px
            dy = j - py
            d2 = dx**2 + dy**2
            
            # Store the current pixel's count value
            pixval = image[j,i]
            
            if d2 <= r**2:
                apsum += pixval
                
    return apsum
    


# Function for fitting a sine wave
def sine_fit(t, w, phi, amp, b):

    sine = amp*np.sin(w*t + phi) + b
    
    return sine
    
    
    
# Function for fitting a parabola
def parabola_fit(x, a, b):
    
    y = a * x**2 + b
    
    return y

    
    
    


#################### FUNCTIONS ####################





# Specify necessary directories
datadir = "/home/bjung/Documents/Leiden_University/brp/data_red/calib_data"
stddatadir = datadir + "/sorted/STD,IPOL"
veladir = stddatadir + "/Vela1_95_coord1/CHIP1"
plotim1 = stddatadir + "/Vela1_95_coord1/CHIP1/tpl8/FORS2.2011-05-04T00:24:56.664.fits"
plotim2 = stddatadir + "/WD1615_154/CHIP1/tpl3/FORS2.2011-05-04T05:37:44.543.fits"
plotims = [plotim1, plotim2]
plotdir = "/home/bjung/Documents/Leiden_University/brp/data_red/plots"
# Create list of directories and files within veladir
std_dirs = [stddatadir + "/Vela1_95_coord1/CHIP1", stddatadir + "/WD1615_154/CHIP1"]

# Pixel scale (same for all exposures)
pixscale = 0.126 #[arcsec/pixel]
# Counters for the number of skipped templates
skip_lst = [2, 0]
# Range of aperture radii for plotting polarisation degree against aperture radius
r_range = np.arange(1, 17, 1) #[pixel]
# Range of retarder waveplate angles
ret_angles = np.arange(0.0, 90.0, 22.5) #[degrees]
# Load bias frame
bias_header, bias = extract_data(datadir + "/masterbias.fits")
# Boolean for determining whether the list structures have to be computed anew
compute_anew = False
# Aproximate coordinates of selection of stars within CHIP1 of 'Vela1_95_coord1' and 'WD1615_154'. Axis 0 specifiec the different std_dirs; axis 1 specifies the different stars within the std_dir; axis 2 specifies the x, y1, y2 coordinate of the specific star (with y1 specifying the y coordinate on the upper slit and y2 indicating the y coordinate on the lower slit) and the aproximate stellar radius. NOTE: THE LAST LIST WITHIN AXIS1 IS A SKY APERTURE!!!
star_lsts = [[[1034, 347, 251, 15], [1177, 368, 273, 8], [319, 345, 250, 5], [281, 499, 403, 6], [414, 139, 45, 12], [253, 376, 281, 4], [531, 706, 609, 5], [1583, 322, 229, 3], [1779, 321, 224, 4], [1294, 725, 627, 4], [1501, 719, 622, 7], [1040, 890, 791, 15], [1679, 150, 58, 15], [923, 513, 423, 15], [259, 157, 63, 15]],

            [[1039, 347, 253, 12], [248, 195, 103, 8], [240, 380, 286, 8], [362, 351, 258, 3], [599, 541, 446, 5], [365, 700, 604, 5], [702, 903, 806, 6], [756, 374, 279, 4], [801, 136, 43, 4], [1055, 133, 43, 4], [1186, 130, 37, 4], [1330, 139, 46, 3], [1132, 685, 592, 3], [1222, 685, 592, 4], [1395, 679, 587, 4], [1413, 912, 816, 5], [1517, 915, 816, 3], [1618, 894, 799, 3], [1649, 709, 613, 3], [1655, 542, 449, 5], [1643, 512, 417, 5], [1632, 190, 97, 6], [1608, 178, 85, 4], [1437,336,240, 15], [602, 700, 608, 15], [502, 152, 60, 15], [1303, 886, 790, 15]]] #[pixel]



# Compute the linear polarization degrees for each template of exposures taken of Vela1_95 and WD1615_154
for i, std_dir in enumerate(std_dirs): 
    print("\n\n\n{}".format(std_dir))
    
    
    
    # Asks whether the list structures have to be computed anew
    if not compute_anew:
        print("SKIP COMPUTATION OF LIST STRUCTURES")
        break
    # Skip 'loadfiles' directory      
    if std_dir == "loadfiles":
        continue
        
        
        
    # Create a list with all the template directories within std_dirs
    tpl_dirlst, tpl_flst = mk_lsts(std_dir)
    tpl_dirlst = np.sort(tpl_dirlst)

    # A 1D lst for storing the exposures containing the least amount of stars per template
    least_expnos = np.zeros(len(tpl_dirlst), dtype=np.int64) 
    
       
    
    # Initiate lists containing the ordinary flux, the extraordinary flux and the normalized flux differences, for each template (axis=0), each exposure file (axis 1), each selected star within the exposure (axis2), and each aperture radius (axis 3)
    O_0lst, sigmaO_0lst = [], []
    E_0lst, sigmaE_0lst = [], []
    F_0lst, sigmaF_0lst = [], []
    # List for storing the filters of each exposure as well as the q-index and xcoordinate of the STD star within each template (axis1) and each exposure (axis2)
    filter_lst = []
    for j, tpl_name in enumerate(tpl_dirlst):
        tpl_dir = std_dir + '/' + tpl_name
        
        # Create a list with filenames of files stored within tpldir
        expdir_lst, expfile_lst = mk_lsts(tpl_dir)
        expfile_lst = np.sort(expfile_lst)
        
        # Initial setting for the least amount of detected stars within the template
        N_stars = 1e18   
        
        # Skips the first template taken of Vela1_95, since the star is on the edge of a slit within this template
        if (tpl_dir == veladir + "/tpl1") or (len(expfile_lst) < 4):
            print("\n\n\tSKIP {}!!!".format(tpl_name))
            continue
        else:
            tpl_name = tpl_dirlst[j]
            print("\n\n\t{}".format(tpl_name))      
        
        
        
        # Initiate first sublists for distinguishing different exposures
        O_1lst, sigmaO_1lst = [], []
        E_1lst, sigmaE_1lst = [], []
        F_1lst, sigmaF_1lst = [], []
        for k, f in enumerate(expfile_lst):
            print("\n\t\t {}".format(f))

            if (f.endswith("fits")) and (len(expfile_lst) == 4):
                header, data = extract_data(tpl_dir + '/' + f)
                # Subtract bias
                data = data - bias
                # Save corrected image
                savedir = tpl_dir + "/corrected" 
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                    hdu = fits.PrimaryHDU(data)
                    hdulist = fits.HDUList([hdu])
                    hdulist.writeto(savedir + '/' + f.split(".fits")[0] + "_COR.fits")


       
                # Specify observation parameters
                expno = header["HIERARCH ESO TPL EXPNO"]
                filt_name = header["HIERARCH ESO INS FILT1 NAME"]
                filt_id = header["HIERARCH ESO INS FILT1 ID"]
                ret_angle = header["HIERARCH ESO INS RETA2 POSANG"] * np.pi / 180.
                woll_angle = header["HIERARCH ESO INS WOLL POSANG"] * np.pi / 180.
                print("\t\t\t\tFILTER_ID: {A}; \t FILTER_NAME: {B}".format(A=filt_id, B = filt_name))
                print("\t\t\t\tWollangle: {A}; \t Retangle: {B}".format(A=woll_angle, B = np.round(ret_angle, 2)))                
                  

   
                # Initiate second sublist of F for distinguishing between different stars within the current exposure
                O_2lst, sigmaO_2lst = [], []
                E_2lst, sigmaE_2lst = [], []
                F_2lst, sigmaF_2lst = [], []
                for q, coord in enumerate(star_lsts[i]):

                    # Finds the central pixel of the selected stars within the specific exposure                    
                    coord1, coord2 = coord[0:2], [coord[0],coord[2]]
                    if q != len(star_lsts[i]):
                        center1 = find_center(coord1, data, 15)
                        center2 = find_center(coord2, data, 15)
                        centers = [center1, center2]
                    if q == len(star_lsts[i]):
                        centers = [coord1, coord2] # Sky aperture
                    

                    
                    # Initiate third sublist of F for distinguishing between different aperture radii
                    O_3lst, sigmaO_3lst = [], []
                    E_3lst, sigmaE_3lst = [], []
                    F_3lst, sigmaF_3lst = [], [] 
                    for l, R in enumerate(r_range):  

                        # Lists for temporary storage of aperture sum values and corresponding shotnoise levels
                        apsum_lst, shotnoise_lst = [], []
                        for center in centers:
                        
                            # Compute cumulative counts within aperture
                            apsum = apersum(data, center[0], center[1], R)
                            apsum_lst.append(apsum)
                            # Compute photon shot noise within aperture 
                            shotnoise = np.sqrt(apsum)
                            shotnoise_lst.append(shotnoise)
                        
                        # Compute normalised flux differences for current aperture size
                        F, sigmaF = fluxdiff_norm(apsum_lst[1], apsum_lst[0], shotnoise_lst[1], shotnoise_lst[0]) 
                        
                        
                        
                        # Append results to third sublist
                        O_3lst.append(apsum_lst[1]), sigmaO_3lst.append(shotnoise_lst[1])
                        E_3lst.append(apsum_lst[0]), sigmaE_3lst.append(shotnoise_lst[0])
                        F_3lst.append(F), sigmaF_3lst.append(sigmaF)
                    # Append the third sublist to the second sublist
                    O_2lst.append(O_3lst), sigmaO_2lst.append(sigmaO_3lst)
                    E_2lst.append(E_3lst), sigmaE_2lst.append(sigmaE_3lst)
                    F_2lst.append(F_3lst), sigmaF_2lst.append(sigmaF_3lst)
            # Append second sublist to first sublist
            O_1lst.append(O_2lst), sigmaO_1lst.append(sigmaO_2lst)
            E_1lst.append(E_2lst), sigmaE_1lst.append(sigmaE_2lst)
            F_1lst.append(F_2lst), sigmaF_1lst.append(sigmaF_2lst)
        # Append first sublist to main list 
        O_0lst.append(O_1lst), sigmaO_0lst.append(sigmaO_1lst)
        E_0lst.append(E_1lst), sigmaE_0lst.append(sigmaE_1lst)
        F_0lst.append(F_1lst), sigmaF_0lst.append(sigmaF_1lst)      
        # Append filter name to filter_lst
        filter_lst.append(filt_name)
    # Transform into arrays for future computations
    O_0lst, sigmaO_0lst = np.array(O_0lst), np.array(sigmaO_0lst)
    E_0lst, sigmaE_0lst = np.array(E_0lst), np.array(sigmaE_0lst)
    F_0lst, sigmaF_0lst = np.array(F_0lst), np.array(sigmaF_0lst) 
    
    # Save the flux arrays
    savedir = stddatadir + "/loadfiles/" + std_dir.split("/")[-2]
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    np.save(savedir + "/O_0lst.npy", O_0lst), np.save(savedir + "/sigmaO_0lst.npy", sigmaO_0lst)
    np.save(savedir + "/E_0lst.npy", E_0lst), np.save(savedir + "/sigmaE_0lst.npy", sigmaE_0lst)
    np.save(savedir + "/F_0lst.npy", F_0lst), np.save(savedir + "/sigmaF_0lst.npy", sigmaF_0lst)
    # Save filt_lst
    np.save(savedir + "/filter_lst.npy", filter_lst)
 
    



for i, std_dir in enumerate(std_dirs): 
    print("\n\n\n{}".format(std_dir)) 
    
    # Load list structures
    loaddir = stddatadir + "/loadfiles/" + std_dir.split("/")[-2]
    os.chdir(loaddir)
    O_0lst, sigmaO_0lst = np.load("O_0lst.npy"), np.load("sigmaO_0lst.npy")
    E_0lst, sigmaE_0lst = np.load("E_0lst.npy"), np.load("sigmaE_0lst.npy")
    F_0lst, sigmaF_0lst = np.load("F_0lst.npy"), np.load("sigmaF_0lst.npy")
    print("List structures loaded...")
    # Load filter_lst
    filter_lst = np.load("filter_lst.npy")
    os.chdir(stddatadir)

    
    
    # Stellar radius of the STD star
    star_lst = np.array(star_lsts[i])
    std_rad = star_lst[0,3]
    

    # List of colors for distinguishing the B and V filter in the plots
    Bcolor_lst, b = ["#3333ff", "#3399ff", "#33ffff", "#e6ac00"], 0
    Vcolor_lst, v = ["#336600", "#66cc00", "#cccc00", "#ccff33"], 0


    # Create a list with all the template directories within std_dirs
    tpl_dirlst, tpl_flst = mk_lsts(std_dir)
    tpl_dirlst = np.sort(tpl_dirlst)
        
    
    # Compute Stokes variables
    Q_jqr = 0.5 * F_0lst[:,0,:,:] - 0.5 * F_0lst[:,2,:,:]
    U_jqr = 0.5 * F_0lst[:,1,:,:] - 0.5 * F_0lst[:,3,:,:]
    # Compute standard deviations
    sigmaQ_jqr = 0.5 * np.sqrt(sigmaF_0lst[:,0,:,:]**2 + sigmaF_0lst[:,2,:,:]**2)
    sigmaU_jqr = 0.5 * np.sqrt(sigmaF_0lst[:,1,:,:]**2 + sigmaF_0lst[:,3,:,:]**2)

    # Compute degree of linear polarization and polarization angle
    P_jqr = np.sqrt(Q_jqr**2 + U_jqr**2)            
    phi_jqr = (1/2.) * np.arctan(np.divide(Q_jqr, U_jqr)) * (180. / np.pi) 
    # Compute standard deviations 
    temp = np.sqrt( (Q_jqr * sigmaQ_jqr)**2 + (U_jqr * sigmaU_jqr)**2 )
    sigmaP_jqr = np.divide(temp, P_jqr)
    # Compute zero angle
    phi0_jqr = phi_jqr - 1.54 #TODO CHECK INSTRUMENTAL ROTATION ANGLES
    
    
    # Select the aperture radii corresponding to each star
    star_ind, rad_ind = np.arange(0, len(star_lst), 1), star_lst[:,3] - 1
    Q_jq, sigmaQ_jq = Q_jqr[:,star_ind, rad_ind], sigmaQ_jqr[:,star_ind, rad_ind]
    U_jq, sigmaU_jq = U_jqr[:,star_ind, rad_ind], sigmaU_jqr[:,star_ind, rad_ind]
    P_jq, sigmaP_jq = P_jqr[:,star_ind,rad_ind], sigmaP_jqr[:,star_ind,rad_ind]
    phi0_jq = phi0_jqr[:,star_ind,rad_ind]
    # Compute stellar distances to FOV center
    starsO, starsE = star_lst[:,[0,1]], star_lst[:,[0,2]]
    starsO_dist = np.sqrt((starsO[:,0]-1034)**2 + starsO[:,1]**2)
    starsE_dist = np.sqrt((starsE[:,0]-1034)**2 + starsE[:,1]**2)
    starsAll_dist = np.concatenate((starsO_dist, starsE_dist), axis=0)
    
    
    '''
    r_diffO, r_diffE = np.diff(O_0lst, axis = 3), np.diff(E_0lst, axis = 3)
    surface_diffs = np.diff(np.pi * r_range**2)
    shape = list(r_diffO.shape)
    #print(shape)
    shape[3] = 1
    surface_diffs = np.tile(surface_diffs, shape)
    r_normdiffO = r_diffO / surface_diffs
    r_normdiffE = r_diffE / surface_diffs
    
    print("min r_diffO:\t\t\t{}".format(np.argmin(r_normdiffO, axis=3)))
    print("min r_diffO:\t\t\t{}".format(np.argmin(r_normdiffO, axis=3)))
    '''
    
    
    
    #### Plots ####  
    for j, tpl_name in enumerate(tpl_dirlst[skip_lst[i]::]):
        
        if filter_lst[j] == "b_HIGH":
            # Aperture radius vs linear polarization
            plt.figure(0)
            tempr = np.linspace(0., r_range[-1]*pixscale, 1000)
            if i == 0:
                tempPl1, tempPl2 = np.tile(0.0750, len(tempr)), np.tile(0.076, len(tempr))
                plt.plot(tempr, tempPl1, color='0.1', linestyle = '--')
                plt.plot(tempr, tempPl2, color='0.1', linestyle = '--')
            elif i == 1:
                tempP = np.tile(0.0, len(tempr))
                plt.plot(tempr, tempP, color='0.1', linestyle = '--')
            plt.errorbar(r_range*pixscale, P_jqr[j,0,:], yerr = sigmaP_jqr[j,0,:], marker='o', color=Bcolor_lst[b], label=tpl_name)
            


            # The normalized ordinary and extraordinary flux rates as function of retarder waveplate angle for the standard star
            plt.figure(1)
            normO = np.average(O_0lst[j,:,0,std_rad-1], weights = 1. / sigmaO_0lst[j,:,0,std_rad-1]**2)
            normO_err = np.sqrt(1. / np.sum(1./sigmaO_0lst[j,:,0,std_rad-1]**2))
            normE = np.average(E_0lst[j,:,0,std_rad-1], weights = 1. / sigmaE_0lst[j,:,0,std_rad-1]**2)
            normE_err = np.sqrt(1. / np.sum(1./sigmaE_0lst[j,:,0,std_rad-1]**2))
            
            plotO = O_0lst[j,:,0,std_rad-1]/normO
            tempO = sigmaO_0lst[j,:,0,std_rad-1]**2 + (O_0lst[j,:,0,std_rad-1]**2 * normO_err**2) / normO**2
            plotO_err = 1/normO * np.sqrt(tempO)
            
            plotE = E_0lst[j,:,0,std_rad-1]/normE
            tempE = sigmaE_0lst[j,:,0,std_rad-1]**2 + (E_0lst[j,:,0,std_rad-1]**2 * normE_err**2) / normE**2
            plotE_err = 1/normE * np.sqrt(tempE)
            
            plt.errorbar(ret_angles, plotO, yerr = plotO_err, marker='o', color = Bcolor_lst[b], label = tpl_name)
            plt.errorbar(ret_angles, plotE, yerr = plotE_err, marker='D', color = Bcolor_lst[b], label = tpl_name)  
            
            '''
            # Computing a fit
            b_guessO, A_guessO = plotO[0], plotO[2]
            b_guessE, A_guessE = plotE[0], plotE[2]
            omega_guess, phi_guessO, phi_guessE = 2*np.pi / 180., 0.0, 90.0
            p0_O = [omega_guess, phi_guessO, A_guessO, b_guessO]
            p0_E = [omega_guess, phi_guessE, A_guessE, b_guessE]
            t = np.linspace(0.0, 90., len(plotO))
            fitO, fitE = curve_fit(sine_fit, t, plotO, p0=p0_O), curve_fit(sine_fit, t, plotE, p0=p0_E)
            data_fitO, data_fitE = sine_fit(np.linspace(0., 90., 1000), *fitO[0]), sine_fit(np.linspace(0., 90., 1000), *fitE[0])

            plt.plot(np.linspace(0., 90., 1000), data_fitO, color = Bcolor_lst[v], label = "fit {}".format(tpl_name))
            plt.plot(np.linspace(0., 90., 1000), data_fitE, color = Bcolor_lst[v], label = "fit {}".format(tpl_name))
            ''' 

            if j != 0: #TODO REMOVE!!!!!
                continue
            
            
            
            # U vs Q for all stars and all templates
            plt.figure(2)
            plt.errorbar(U_jqr[j,1:-1,std_rad-1], Q_jqr[j,1:-1,std_rad-1], xerr = sigmaU_jqr[j,1:-1,std_rad-1], yerr = sigmaQ_jqr[j,1:-1,std_rad-1], fmt='o', color = Bcolor_lst[b], label = tpl_name)
            plt.errorbar(U_jqr[j,0,std_rad-1], Q_jqr[j,0,std_rad-1], xerr = sigmaU_jqr[j,0,std_rad-1], yerr = sigmaQ_jqr[j,0,std_rad-1], fmt='*', markersize = 16., color = Bcolor_lst[b])
            plt.errorbar(U_jqr[j,-1,std_rad-1], Q_jqr[j,-1,std_rad-1], xerr = sigmaU_jqr[j,-1,std_rad-1], yerr = sigmaQ_jqr[j,-1,std_rad-1], fmt='s', markersize = 10., color = Bcolor_lst[b])
            



            #TODO CODE THIS!!!
            # Plot linear polarization degree versus distance from FOV center for B-filter
            plt.figure(3)
            plt.errorbar(starsO_dist*pixscale, P_jq[j,:], yerr = sigmaP_jq[j,:], fmt='o', color = Bcolor_lst[b], label = tpl_name)
            for q, coords in enumerate(star_lst):
                loc = np.array([starsO_dist[q]*pixscale, P_jq[j,q]])
                textloc = loc - np.array([10., 0.0025])
                plt.annotate('[{},{}]'.format(coords[0], coords[1]), xy = loc, xytext = textloc)
            #plt.errorbar(starsE_dist*pixscale, P_jq[j,:], yerr = sigmaP_jq[j,:], fmt='D', color = Bcolor_lst[b], label = tpl_name)
            # Parabolic fit
            p0 = [0.057, 0.0]
            dists = np.linspace(0.0, np.max(starsAll_dist*pixscale)+20*pixscale, 1000)
            fit_init = curve_fit(parabola_fit, starsO_dist, P_jq[j,:], p0=p0)
            fit_P = parabola_fit(dists, *fit_init[0])
            #plt.plot(dists, fit_P, color = Bcolor_lst[b]) #TODO UNCOMMENT IF WORKS
            
            b += 1  
                       



        if filter_lst[j] == "v_HIGH":
            # Aperture radius vs linear polarization
            plt.figure(0)
            tempr = np.linspace(0., r_range[-1]*pixscale, 1000)
            if i == 0:
                tempPl1, tempPl2 = np.tile(0.0821, len(tempr)), np.tile(0.0827, len(tempr))
                plt.plot(tempr, tempPl1, color='0.1', linestyle = '--')
                plt.plot(tempr, tempPl2, color='0.1', linestyle = '--')
            elif i == 1:
                tempP = np.tile(0.0, len(tempr))
                plt.plot(tempr, tempP, color='0.1', linestyle = '--')
            plt.errorbar(r_range*pixscale, P_jqr[j,0,:], yerr = sigmaP_jqr[j,0,:], marker='o', color=Vcolor_lst[v], label=tpl_name)
            


            # The normalized ordinary and extraordinary flux rates as function of retarder waveplate angle for the standard star
            plt.figure(1)
            # Calculating the normalized flux values and the corresponding errors
            normO = np.average(O_0lst[j,:,0,std_rad-1], weights = 1. / sigmaO_0lst[j,:,0,std_rad-1]**2)
            normO_err = np.sqrt(1. / np.sum(1./sigmaO_0lst[j,:,0,std_rad-1]**2))
            normE = np.average(E_0lst[j,:,0,std_rad-1], weights = 1. / sigmaE_0lst[j,:,0,std_rad-1]**2)
            normE_err = np.sqrt(1. / np.sum(1./sigmaE_0lst[j,:,0,std_rad-1]**2))
            
            plotO = O_0lst[j,:,0,std_rad-1]/normO
            tempO = sigmaO_0lst[j,:,0,std_rad-1]**2 + (O_0lst[j,:,0,std_rad-1]**2 * normO_err**2) / normO**2
            plotO_err = 1/normO * np.sqrt(tempO)
            
            plotE = E_0lst[j,:,0,std_rad-1]/normE
            tempE = sigmaE_0lst[j,:,0,std_rad-1]**2 + (E_0lst[j,:,0,std_rad-1]**2 * normE_err**2) / normE**2
            plotE_err = 1/normE * np.sqrt(tempE)

            plt.errorbar(ret_angles, plotO, yerr = plotO_err, marker='o', color = Vcolor_lst[v], label = tpl_name)
            plt.errorbar(ret_angles, plotE, yerr = plotE_err, marker='D', color = Vcolor_lst[v], label = tpl_name)
                 
            '''
            # Computing a fit
            b_guessO, A_guessO = plotO[0], plotO[2]
            b_guessE, A_guessE = plotE[0], plotE[2]
            omega_guess, phi_guessO, phi_guessE = 2*np.pi / 180., 0.0, 90.0
            p0_O = [omega_guess, phi_guessO, A_guessO, b_guessO]
            p0_E = [omega_guess, phi_guessE, A_guessE, b_guessE]
            t = np.linspace(0.0, 90., len(plotE))
            fitO, fitE = curve_fit(sine_fit, t, plotO, p0=p0_O), curve_fit(sine_fit, t, plotE, p0=p0_E)
            data_fitO, data_fitE = sine_fit(np.linspace(0., 90., 1000), *fitO[0]), sine_fit(np.linspace(0., 90., 1000), *fitE[0])
            '''


            '''
            plt.plot(np.linspace(0., 90., 1000), data_fitO, color = Vcolor_lst[v], label = "fit {}".format(tpl_name))
            plt.plot(np.linspace(0., 90., 1000), data_fitE, color = Vcolor_lst[v], label = "fit {}".format(tpl_name))
            '''
            


            if (j != 2): #TODO REMOVE!!!!!
                continue
                


            # U vs Q for all stars and all templates            
            plt.figure(2)
            plt.errorbar(U_jqr[j,1:-1,std_rad-1], Q_jqr[j,1:-1,std_rad-1], xerr = sigmaU_jqr[j,1:-1,std_rad-1], yerr = sigmaQ_jqr[j,1:-1,std_rad-1], fmt='o', color = Vcolor_lst[v], label = tpl_name)
            plt.errorbar(U_jqr[j,0,std_rad-1], Q_jqr[j,0,std_rad-1], xerr = sigmaU_jqr[j,0,std_rad-1], yerr = sigmaQ_jqr[j,0,std_rad-1], fmt='*', markersize = 16., color = Vcolor_lst[v])
            plt.errorbar(U_jqr[j,-1,std_rad-1], Q_jqr[j,-1,std_rad-1], xerr = sigmaU_jqr[j,-1,std_rad-1], yerr = sigmaQ_jqr[j,-1,std_rad-1], fmt='s', markersize = 10., color = Vcolor_lst[v])
            
            

            # Plot linear polarization degree versus distance from FOV center for V-filter            
            plt.figure(4)
            plt.errorbar(starsO_dist*pixscale, P_jq[j,:], yerr = sigmaP_jq[j,:], fmt='o', color = Vcolor_lst[v], label = tpl_name)
            for q, coords in enumerate(star_lst):
                loc = np.array([starsO_dist[q]*pixscale, P_jq[j,q]])
                textloc = loc - np.array([10., 0.0025])
                plt.annotate('[{},{}]'.format(coords[0], coords[2]), xy = loc, xytext = textloc)
            #plt.errorbar(starsE_dist*pixscale, P_jq[j,:], yerr = sigmaP_jq[j,:], fmt='D', color = Vcolor_lst[v], label = tpl_name)
            
            '''
            rs = np.linspace(0.0, np.max(starsAll_dist*pixscale)+20*pixscale, 1000)
            model_P = 0.057 * rs**2
            plt.plot(rs, model_P, color='g', label = tpl_name) #TODO UNCOMMENT WHEN WORKING
            '''
            
            v += 1
            
            
    '''   
    # Vector plot showing polarization degrees of stars
    plt.figure(7)
    header, data = extract_data(plotims[i])
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.imshow(data, cmap='Greys', origin='lower', norm=norm)
    '''

    
    
    # Check if the savefile directory is present
    savedir = plotdir + '/' + std_dir.split("/")[-2]
    if not os.path.exists(savedir):
        os.chdir(plotdir)
        os.makedirs(savedir)
        os.chdir(stddatadir)
    


    plt.figure(0)
    plt.title(r"Polarization Profile {A}".format(A = std_dir.split("/")[-2]))
    plt.xlabel(r'$\mathrm{Radius \ [arcsec]}$', size=24)
    plt.ylabel(r'$\mathrm{Degree \ of \ linear \ polarization \ [-]}$', size=24)
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.savefig(savedir + '/' + 'RvsPl_alltplsV2')
    
    
    plt.figure(1)     
    plt.axis(xmin=0, xmax=91)                 
    plt.title(r"$\alpha$ vs f_norm {A}".format(A = std_dir.split("/")[-2]))
    plt.xlabel(r'$\alpha \mathrm{\ [^{\circ}]}$', size=24)
    plt.ylabel(r'$F \mathrm{\ [-]}$', size=24)
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.savefig(savedir + '/' + 'alpha_F')  
    
    
    plt.figure(2)                      
    plt.title(r"U/I vs Q/I {A}".format(A = std_dir.split("/")[-2]))
    plt.xlabel(r'$\frac{U}{I} \mathrm{\ [-]}$', size=24)
    plt.ylabel(r'$\frac{Q}{I} \mathrm{\ [-]}$', size=24)
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.savefig(savedir + '/' + 'UvsQ')

    
    plt.figure(3)                    
    plt.title(r"Radial LinPol Profile B-band".format(A = std_dir.split("/")[-2]))
    plt.xlabel(r'Radial distance [arcsec]', size=24)
    plt.ylabel(r'Degree of linear polarization [-]', size=24)
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.savefig(savedir + '/' + 'radialprofileB')


    plt.figure(4)                   
    plt.title(r"Radial LinPol Profile V-band".format(A = std_dir.split("/")[-2]))
    plt.xlabel(r'Radial distance [arcsec]', size=24)
    plt.ylabel(r'Degree of linear polarization [-]', size=24)
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.savefig(savedir + '/' + 'radialprofileV')
        
    # Show and close figures
    plt.show()  
    plt.close(0), plt.close(1), plt.close(2), plt.close(3), plt.close(4)




              
                    













