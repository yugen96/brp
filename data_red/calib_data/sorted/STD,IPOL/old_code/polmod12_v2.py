import numpy as np
import os
import aper
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import aperture_photometry
from photutils import DAOStarFinder
from photutils import CircularAperture
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
    
    
    
    


#################### FUNCTIONS ####################





# Specify necessary directories
datadir = "/home/bjung/Documents/Leiden_University/brp/data_red/calib_data/sorted"
stddatadir = datadir + "/STD,IPOL"
veladir = stddatadir + "/Vela1_95_coord1/CHIP1"
imdir = veladir + "/tpl4"
plotdir = "/home/bjung/Documents/Leiden_University/brp/data_red/plots"

# Create list of directories and files within veladir
std_dirs = [stddatadir + "/Vela1_95_coord1/CHIP1", stddatadir + "/WD1615_154/CHIP1"]

# Specify Vela1_95's and WD1615-154's physical X-coordinates within the fits files (in pixels) 
aprox_xc = [1033, 1038]

# Range of aperture radii for plotting polarisation degree against aperture radius
r_range = np.arange(1, 20, 1) 





#TODO automize for all folders in STD_sorted2,IPOL
# Compute the linear polarization degrees for each template of exposures taken of Vela1_95 and WD1615_154
for i, std_dir in enumerate(std_dirs): 
    print("\n\n\n{}".format(std_dir)) 
    
    # Create a list with all the template directories within std_dirs
    tpl_dirlst, tpl_flst = mk_lsts(std_dir)
    tpl_dirlst = np.sort(tpl_dirlst)
    
    # List of colors for distinguishing the B and V filter in the plots
    Bcolor_lst, b = ["#001233", "#003399", "#1a66ff", "#99bbff"], 0
    Vcolor_lst, v = ["#660066", "#cc00cc", "#ff66ff", "#ffb3ff"], 0

    # Initiate lists containing the normalized flux differences, for each template (axis=0), each exposure file (axis 1), each aperture radius (axis 2) and each star within the exposure (dictionary using x-coordinates as key-words)
    F_0lst, sigmaF_0lst = [], []

    # Counter for skipped templates
    skips = 0
    
    for j, tpl_name in enumerate(tpl_dirlst):
        tpl_dir = std_dir + '/' + tpl_name
        
        # Create a list with filenames of files stored within tpldir
        expdir_lst, expfile_lst = mk_lsts(tpl_dir)
        expfile_lst = np.sort(expfile_lst)
        
        # Skips the first template taken of Vela1_95, since the star is on the edge of a slit within this template
        if (tpl_dir == veladir + "/tpl1") or (len(expfile_lst) < 4):
            print("\n\n\tSKIP {}!!!".format(tpl_name))
            skips += 1
            continue
        else:
            tpl_name = tpl_dirlst[j]
            print("\n\n\t{}".format(tpl_name)) 
            
            
        # Initiate first sublist of F for distinguishing different exposures
        F_1lst, sigmaF_1lst = [], []
        for k, f in enumerate(expfile_lst):
            print("\n\t\t {}".format(f))
            
         
            
            if (f.endswith("fits")) and (len(expfile_lst) == 4):
                header, data = extract_data(tpl_dir + '/' + f)
        
                                
                # Specify observation parameters
                expno = header["HIERARCH ESO TPL EXPNO"]
                pixscale = header["HIERARCH ESO INS PIXSCALE"]
                filt_name = header["HIERARCH ESO INS FILT1 NAME"]
                filt_id = header["HIERARCH ESO INS FILT1 ID"]
                ret_angle = header["HIERARCH ESO INS RETA2 POSANG"] * np.pi / 180.
                woll_angle = header["HIERARCH ESO INS WOLL POSANG"] * np.pi / 180.
                print("\t\t\t\tFILTER_ID: {A}; \t FILTER_NAME: {B}".format(A=filt_id, B = filt_name))
                print("\t\t\t\tWollangle: {A}; \t Retangle: {B}".format(A=woll_angle, B = np.round(ret_angle, 2)))                

             
             
                #TODO REWRITE YOURSELF??? / READ THROUGH DAOSTARFINDER SOURCE CODE???
                # Finds relevant stars in image
                mean, median, std = sigma_clipped_stats(data, sigma=3.0, iters=5)
                daofind = DAOStarFinder(fwhm=3.0, threshold=1000.*std)
                sources = daofind(data - median)
                
                # Determine the x and y coordinates of the sources
                Xlst, Ylst = list(sources['xcentroid']), list(sources['ycentroid'])
                xcoords, ycoords = np.array([int(xc) for xc in Xlst]), np.array([int(yc) for yc in Ylst])
                # Save the first set of coordinates
                if k == 0:
                    print("YESS!!!", skips)
                    init_xcoords = xcoords
                                    
                

                #TODO Compute error margins    
                # Initiate second sublist of F for distinguishing between different aperture radii
                F_2lst, sigmaF_2lst = [], []   
                for q, xc in enumerate(list(set(xcoords))):    
                
                    # Skip computation of aperture sums for the star at 'xc' if not already detected in the first exposure
                    originalX = np.logical_and(init_xcoords >= xc-5, init_xcoords <= xc+5)
                    if np.sum(originalX) != 2:
                        print("\t\t\t\tCURRCOORD:\t{}".format(xcoords))
                        print("\t\t\t\tORIGCOORD:\t{}".format(init_xcoords))
                        print("\t\t\t\tSKIPSTAR:\t{}".format(xc))
                        print("\t\t\t\toriginalX:\t{}".format(originalX))
                        continue       
                    
                    
                    
                    # Initiate third sublist of F for distinguishing between different stars within the current exposure
                    F_3lst, sigmaF_3lst = [], []
                    for l, R in enumerate(r_range):                      
                    
                        # Look up the indices of identical stars imaged on different slits
                        indices = np.where(np.logical_and(xcoords >= xc-5, xcoords <= xc+5))
                                                
                        # Skip false sources (i.e. bac pixels or sources imaged on only one slit)
                        # TODO THERE MIGHT STILL BE THE POSSIBILITY THAT FALSE SOURCES ARE TAKEN INTO ACCOUNT, IF THERE ARE TWO WITHIN <=5 PIXELS PROXIMITY FROM EACH OTHER
                        if len(indices[0]) != 2:
                            continue

                        # Lists for temporary storage of aperture sum values and corresponding shotnoise levels
                        apsum_lst, shotnoise_lst = [], []
                        for ind in indices[0]:
                        
                            # Compute cumulative counts within aperture
                            apsum = aper.apersum(data, xcoords[ind], ycoords[ind], R)
                            apsum_lst.append(apsum)
                            # Compute photon shot noise within aperture 
                            shotnoise = np.sqrt(apsum)
                            shotnoise_lst.append(shotnoise)

                        # Compute normalised flux differences for current aperture size
                        F, sigmaF = fluxdiff_norm(apsum_lst[1], apsum_lst[0], shotnoise_lst[1], shotnoise_lst[0]) 


                        # Append results to third sublist
                        F_3lst.append(F), sigmaF_3lst.append(sigmaF)
                    print(len(F_3lst))
                    # Append the third sublist to the second sublist
                    F_2lst.append(F_3lst), sigmaF_2lst.append(sigmaF_3lst)     
            # Append second sublist to first sublist
            F_1lst.append(F_2lst), sigmaF_1lst.append(sigmaF_2lst)
            
        F_1lst = np.array(F_1lst)
        print(F_1lst.shape)




        
'''
        # Append first sublist to main list
        F_0lst.append(F_1lst), sigmaF_0lst.append(sigmaF_1lst)
        # TODO FOR A SINGLE TEMPLATE THE NUMBER OF SELECTED STARS OUGHT TO BE THE SAME, PROVIDED THAT THE INTEGRATION TIMES OF THE DIFFERENT EXPOSURES ARE EQUAL. CHECK WHETHER THIS IS THE CASE. ---> Exposure times seem to vary with 0.001s order magnitude. ---> Number of selected stars varies per exposure
         # TODO AND TRANSLATE THE LIST STRUCTURE ABOVE TO THE COMPUTATIONS BELOW
        

        print(j)
        print(F_0lst[j-2][0])

        # Compute Stokes variables
        Q_r = 0.5 * F_0lst[j,0,:] - 0.5 * F_0lst[j,2,:]
        U_r = 0.5 * F_0lst[j,1,:] - 0.5 * F_0lst[j,3,:]
        # Compute standard deviations
        sigmaQ_r = 0.5 * np.sqrt(sigmaF_0lst[j,0,:]**2 + sigmaF_0lst[j,2,:]**2)
        sigmaU_r = 0.5 * np.sqrt(sigmaF_0lst[j,1,:]**2 + sigmaF_0lst[j,3,:]**2)
        
        # Compute degree of linear polarization and polarization angle
        P_r = np.sqrt(Q_r**2 + U_r**2)            
        phi_r = (1/2.) * np.arctan(np.divide(Q_r, U_r)) * (180. / np.pi) 
        # Compute standard deviations 
        temp = np.sqrt( (Q_r * sigmaQ_r)**2 + (U_r * sigmaU_r)**2 )
        sigmaP_r = np.divide(temp, P_r)
        # Compute zero angle
        phi_r0 = phi_r - 1.54 #TODO CHECK INSTRUMENTAL ROTATION ANGLES
                    
            

    #### Plots ####
        savedir = plotdir + '/' + std_dir.split("/")[-2] +'V2'
        if not os.path.exists(savedir):
            os.chdir(plotdir)
            os.makedirs(savedir)
            os.chdir(stddatadir)
        
        
        # Linear polarization degree vs aperture radii
        if filt_name == "b_HIGH":
            plt.errorbar(r_range*pixscale, P_r, yerr = sigmaP_r, marker='o', color=Bcolor_lst[b], label=tpl_name)
            b += 1
        elif filt_name == "v_HIGH":
            plt.errorbar(r_range*pixscale, P_r, yerr = sigmaP_r, marker='o', color=Vcolor_lst[v], label=tpl_name)
            v += 1
    plt.title(r"Polarization Profile {A} {B}".format(A = std_dir.split("/")[-2], B = tpl_name))
    plt.xlabel(r'$\mathrm{Radius \ [arcsec]}$', size=24)
    plt.ylabel(r'$\mathrm{Degree \ of \ linear \ polarization \ [-]}$', size=24)
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.savefig(savedir + '/' + 'RvsPl_alltplsV2')
    plt.show()  
    plt.close()

    
    # Turn list structures into arrays for future computations
    F_0lst, sigmaF_0lst = np.array(F_0lst), np.array(sigmaF_0lst)
'''
              
                    













