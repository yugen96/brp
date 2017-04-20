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



#TODO automize for all folders in STD_sorted2,IPOL
# Compute the linear polarization degrees for each template of exposures taken of Vela1_95 and WD1615_154
for i in range(len(std_dirs)): 
    std_dir = std_dirs[i]
    print("\n\n\n{}".format(std_dir))
    
    # Create a list with all the template directories within std_dirs
    tpl_dirlst, tpl_flst = mk_lsts(std_dir)
    tpl_dirlst = np.sort(tpl_dirlst)
    
    # List of colors for distinguishing the B and V filter in the plots
    Bcolor_lst, b = ["#001233", "#003399", "#1a66ff", "#99bbff"], 0
    Vcolor_lst, v = ["#660066", "#cc00cc", "#ff33ff", "#ff80ff"], 0
    

    for j in range(len(tpl_dirlst)):
        tpl_name = tpl_dirlst[j]
        tpl_dir = std_dir + '/' + tpl_name
        
        # Create a list with filenames of files stored within tpldir
        expdir_lst, expfile_lst = mk_lsts(tpl_dir)
        expfile_lst = np.sort(expfile_lst)
        
        # Skips the first template taken of Vela1_95, since the star is on the edge of a slit within this template
        if (tpl_dir == veladir + "/tpl1") or (len(expfile_lst) < 4):
            print("\n\n\tSKIP {}!!!".format(tpl_name))
            continue
        else:
            tpl_name = tpl_dirlst[j]
            print("\n\n\t{}".format(tpl_name)) 
            

        # Range of aperture radii for plotting polarisation degree against aperture radius
        r_range = np.arange(0.1, 13.1, 0.1)
        # Lists for storing the different Q and U values of each exposure within each template for each aperture radius 
        F_lst = np.zeros([len(tpl_dirlst), len(expfile_lst), len(r_range)])
        sigmaF_lst = np.zeros([len(tpl_dirlst), len(expfile_lst), len(r_range)])   


        for k in range(len(expfile_lst)):
            f = expfile_lst[k]
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
                xcoords, ycoords = list(sources['xcentroid']), list(sources['ycentroid'])
                

                # Pick out Vela1_95
                posts = []
                for q in range(len(xcoords)):
                    # Checks whether there are xcoordinates which correspond to the x-position of the std star in (std_dir + '/' + tpl_dir)
                    if np.round(xcoords[q]) in np.arange(aprox_xc[i]-8., aprox_xc[i]+7., 1.):
                        posts.append((xcoords[q], ycoords[q]))
                
                
                #TODO Compute error margins
                # Calculate aperture sums for different aperture radii
                for l in range(len(r_range)):      
                    R = r_range[l]
                    # Compute cumulative counts within aperture
                    aps = CircularAperture(posts, r=R)
                    ap_sums = list(aperture_photometry(data, aps)['aperture_sum'])
                    # Compute photon shot noise within aperture
                    shotnoise = np.sqrt(ap_sums)
                 
                 
                    # Compute normalised flux differences for current aperture size
                    F, sigmaF = fluxdiff_norm(ap_sums[1], ap_sums[0], shotnoise[1], shotnoise[0]) 
                    
                    # Append results to lists
                    F_lst[j,k,l] = F
                    sigmaF_lst[j,k,l] = sigmaF
                    
                    
                
        # Compute Stokes variables
        Q_r = 0.5 * F_lst[j,0,:] - 0.5 * F_lst[j,2,:]
        U_r = 0.5 * F_lst[j,1,:] - 0.5 * F_lst[j,3,:]
        # Compute standard deviations
        sigmaQ_r = 0.5 * np.sqrt(sigmaF_lst[j,0,:]**2 + sigmaF_lst[j,2,:]**2)
        sigmaU_r = 0.5 * np.sqrt(sigmaF_lst[j,1,:]**2 + sigmaF_lst[j,3,:]**2)
        
        # Compute degree of linear polarization and polarization angle
        P_r = np.sqrt(Q_r**2 + U_r**2)            
        phi_r = (1/2.) * np.arctan(np.divide(Q_r, U_r)) * (180. / np.pi) 
        # Compute standard deviations 
        temp = np.sqrt( (Q_r * sigmaQ_r)**2 + (U_r * sigmaU_r)**2 )
        sigmaP_r = np.divide(temp, P_r)
        # Compute zero angle
        phi_r0 = phi_r - 1.54 #TODO CHECK INSTRUMENTAL ROTATION ANGLES
                
        

    #### Plots ####
        savedir = plotdir + '/' + std_dir.split("/")[-2]
        if not os.path.exists(savedir):
            os.chdir(plotdir)
            os.makedirs(savedir)
            os.chdir(stddatadir)
            
        if filt_name == "b_HIGH":
            plt.errorbar(r_range*pixscale, P_r, yerr = sigmaP_r, fmt='o', color=Bcolor_lst[b], label=tpl_name)
            b += 1
        elif filt_name == "v_HIGH":
            plt.errorbar(r_range*pixscale, P_r, yerr = sigmaP_r, fmt='o', color=Vcolor_lst[v], label=tpl_name)
            v += 1
    plt.title(r"Polarization Profile {A} {B}".format(A = std_dir.split("/")[-2], B = tpl_name))
    plt.xlabel(r'$\mathrm{Radius \ [arcsec]}$', size=24)
    plt.ylabel(r'$\mathrm{Degree \ of \ linear \ polarization \ [-]}$', size=24)
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.savefig(savedir + '/' + 'RvsPl_alltpls')
    #plt.show()  
    plt.clf()
                    













