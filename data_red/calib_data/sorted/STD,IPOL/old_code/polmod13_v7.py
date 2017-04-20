import numpy as np
from sets import Set
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
datadir = "/home/bjung/Documents/Leiden_University/brp/data_red/calib_data"
stddatadir = datadir + "/sorted/STD,IPOL"
veladir = stddatadir + "/Vela1_95_coord1/CHIP1"
imdir = veladir + "/tpl4"
plotdir = "/home/bjung/Documents/Leiden_University/brp/data_red/plots"

# Create list of directories and files within veladir
std_dirs = [stddatadir + "/Vela1_95_coord1/CHIP1", stddatadir + "/WD1615_154/CHIP1"]

# Pixel scale (same for all exposures)
pixscale = 0.126 #[arcsec/pixel]

# Counters for the number of skipped templates
skip_lst = [2, 0]

# Specify Vela1_95's and WD1615-154's physical X-coordinates within the fits files (in pixels) 
aprox_xc = [1033, 1038] #[pixel]

# Range of aperture radii for plotting polarisation degree against aperture radius
r_range = np.arange(1, 16, 1) #[pixel]

# Range of retarder waveplate angles
ret_angles = np.arange(0.0, 90.0, 22.5) #[degrees]

# Load bias frame
bias_header, bias = extract_data(datadir + "/masterbias.fits")

# Boolean for determining whether the list structures have to be computed anew
compute_anew = True






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
    
       
    
    # Initiate lists containing the ordinary flux, the extraordinary flux and the normalized flux differences, for each template (axis=0), each exposure file (axis 1), each aperture radius (axis 2) and each star within the exposure (dictionary using x-coordinates as key-words)
    O_0lst, sigmaO_0lst = [], []
    E_0lst, sigmaE_0lst = [], []
    F_0lst, sigmaF_0lst = [], []
    # List for storing the filters of each exposure
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
        
                # Specify observation parameters
                expno = header["HIERARCH ESO TPL EXPNO"]
                filt_name = header["HIERARCH ESO INS FILT1 NAME"]
                filt_id = header["HIERARCH ESO INS FILT1 ID"]
                ret_angle = header["HIERARCH ESO INS RETA2 POSANG"] * np.pi / 180.
                woll_angle = header["HIERARCH ESO INS WOLL POSANG"] * np.pi / 180.
                print("\t\t\t\tFILTER_ID: {A}; \t FILTER_NAME: {B}".format(A=filt_id, B = filt_name))
                print("\t\t\t\tWollangle: {A}; \t Retangle: {B}".format(A=woll_angle, B = np.round(ret_angle, 2)))                
                
                
                
                
                
                #TODO REWRITE YOURSELF??? / READ THROUGH DAOSTARFINDER SOURCE CODE???
                # Finds relevant stars in image
                mean, median, std = sigma_clipped_stats(data, sigma=3.0, iters=5)
                daofind = DAOStarFinder(fwhm=3.0, threshold=100.*std)
                sources = daofind(data - median)
                
                # Determine the x and y coordinates of the sources
                Xlst, Ylst = list(sources['xcentroid']), list(sources['ycentroid'])
                xcoords, ycoords = np.array([int(xc) for xc in Xlst]), np.array([int(yc) for yc in Ylst])  

                #TODO REMOVED THE REMOVAL OF SINGULAR COORDINATES (WAS COMMENTED OUT)
                

                # Update the set of initial x-coordinates for the current STD_dir
                if (j == 0+skip_lst[i]) and (k == 0):
                    xcs_lst = xcoords
                    ycs_lst = ycoords
                                        


                #TODO Compute error margins    
                # Initiate second sublist of F for distinguishing between different stars within the current exposure
                O_2lst, sigmaO_2lst = [], []
                E_2lst, sigmaE_2lst = [], []
                F_2lst, sigmaF_2lst = [], []
                for q, xc in enumerate(list(set(xcs_lst))):
                    
                    # Look up the indices of identical stars imaged on different slits
                    indices = np.where(np.logical_and(xcs_lst >= xc-5, xcs_lst <= xc+5))    
                    # Skip false sources (i.e. bac pixels or sources imaged on only one slit)
                    # TODO THERE MIGHT STILL BE THE POSSIBILITY THAT FALSE SOURCES ARE TAKEN INTO ACCOUNT, IF THERE ARE TWO WITHIN <=5 PIXELS PROXIMITY FROM EACH OTHER
                    if len(indices[0]) != 2:
                        print("Possible Artefact!")
                        xcs_lstDEL = np.delete(xcs_lst, q, 0)
                        continue
                    
                    # Check which index corresponds to the ESO STDs
                    if xc in np.arange(aprox_xc[i]-7, aprox_xc[i]+8, 1):
                        STD_ind = np.min(indices[0])
                        STD_xcoord = xcs_lst[STD_ind]

                    
                    # Initiate third sublist of F for distinguishing between different aperture radii
                    O_3lst, sigmaO_3lst = [], []
                    E_3lst, sigmaE_3lst = [], []
                    F_3lst, sigmaF_3lst = [], [] 
                    for l, R in enumerate(r_range):  

                        # Lists for temporary storage of aperture sum values and corresponding shotnoise levels
                        apsum_lst, shotnoise_lst = [], []
                        for ind in indices[0]:
                        
                            # Compute cumulative counts within aperture
                            apsum = aper.apersum(data, xcs_lst[ind], ycs_lst[ind], R)
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
    # Save STD_indices, STD_xcoord and filt_lst
    np.save(savedir + "/STD_ind.npy", STD_ind)
    np.save(savedir + "/STD_xcoord.npy", STD_xcoord)
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
    # Load STD_ind
    STD_ind = np.load("STD_ind.npy")
    STD_xcoord = np.load("STD_xcoord.npy")
    filter_lst = np.load("filter_lst.npy")
    os.chdir(stddatadir)
    print("STD index loaded...")
    print("STD xcoord:\t\t".format(STD_xcoord))



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
    
    
    
    
    
    #### Plots ####  
    for j, tpl_name in enumerate(tpl_dirlst[skip_lst[i]::]):
        if filter_lst[j] == "b_HIGH":
            # Aperture radius vs linear polarization
            plt.figure(0)
            plt.errorbar(r_range*pixscale, P_jqr[j,STD_ind,:], yerr = sigmaP_jqr[j,STD_ind,:], marker='o', color=Bcolor_lst[b], label=tpl_name)
        
            # phi0 vs Q for standard star in all templates
            plt.figure(1)
            plt.errorbar(phi0_jqr[j,STD_ind,-1], Q_jqr[j,STD_ind,-1], yerr = sigmaQ_jqr[j,STD_ind,-1], fmt='o', color = Bcolor_lst[b], label = tpl_name)
            
            # phi0 vs U for standard star in all templates
            plt.figure(2)
            plt.errorbar(phi0_jqr[j,STD_ind,-1], U_jqr[j,STD_ind,-1], yerr = sigmaU_jqr[j,STD_ind,-1], fmt='o', color = Bcolor_lst[b], label = tpl_name)
            
            # U vs Q for all stars and all templates
            plt.figure(3)
            plt.errorbar(U_jqr[j,:,-1], Q_jqr[j,:,-1], xerr = sigmaU_jqr[j,:,-1], yerr = sigmaQ_jqr[j,:,-1], fmt='o', color = Bcolor_lst[b], label = tpl_name)
            
            # The normalized ordinary and extraordinary flux rates as function of retarder waveplate angle
            plt.figure(4)
            normO = np.average(O_0lst[j,:,STD_ind,-9], weights = 1. / sigmaO_0lst[j,:,STD_ind,-1]**2)
            normE = np.average(E_0lst[j,:,STD_ind,-1], weights = 1. / sigmaE_0lst[j,:,STD_ind,-1]**2)
            plotO, plotE = O_0lst[j,:,STD_ind,-1]/normO, E_0lst[j,:,STD_ind,-1]/normE
            
            plt.errorbar(ret_angles, plotO, marker='o', color = Bcolor_lst[b], label = tpl_name)
            plt.errorbar(ret_angles, plotE, marker='D', color = Bcolor_lst[b], label = tpl_name)
            
            b += 1


        if filter_lst[j] == "v_HIGH":
            # Aperture radius vs linear polarization
            plt.figure(0)
            plt.errorbar(r_range*pixscale, P_jqr[j,STD_ind,:], yerr = sigmaP_jqr[j,STD_ind,:], marker='o', color=Vcolor_lst[v], label=tpl_name)
            
            # phi0 vs Q for standard star in all templates
            plt.figure(1)
            plt.errorbar(phi0_jqr[j,STD_ind,-1], Q_jqr[j,STD_ind,-1], yerr = sigmaQ_jqr[j,STD_ind,-1], fmt='o', color = Vcolor_lst[v], label = tpl_name)

            # phi0 vs U for standard star in all templates            
            plt.figure(2)
            plt.errorbar(phi0_jqr[j,STD_ind,-1], U_jqr[j,STD_ind,-1], yerr = sigmaU_jqr[j,STD_ind,-1], fmt='o', color = Vcolor_lst[v], label = tpl_name)

            # U vs Q for all stars and all templates            
            plt.figure(3)
            plt.errorbar(U_jqr[j,:,-1], Q_jqr[j,:,-1], xerr = sigmaU_jqr[j,:,-1], yerr = sigmaQ_jqr[j,:,-1], fmt='o', color = Vcolor_lst[v], label = tpl_name)

            # The normalized ordinary and extraordinary flux rates as function of retarder waveplate angle            
            plt.figure(4)
            normO = np.average(O_0lst[j,:,STD_ind,-1], weights = 1. / sigmaO_0lst[j,:,STD_ind,-1]**2)
            normE = np.average(E_0lst[j,:,STD_ind,-1], weights = 1. / sigmaE_0lst[j,:,STD_ind,-1]**2)
            plotO, plotE = O_0lst[j,:,STD_ind,-1]/normO, E_0lst[j,:,STD_ind,-1]/normE
            
            plt.errorbar(ret_angles, plotO, marker='o', color = Vcolor_lst[v], label = tpl_name)
            plt.errorbar(ret_angles, plotE, marker='D', color = Vcolor_lst[v], label = tpl_name)   
    
            v += 1   

    
    
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
    plt.title(r"phi0 vs Q/I {A}".format(A = std_dir.split("/")[-2]))
    plt.xlabel(r'$\phi_0 \mathrm{\ [^{\circ}]}$', size=24)
    plt.ylabel(r'$\frac{Q}{I} \mathrm{\ [-]}$', size=24)
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.savefig(savedir + '/' + 'phi0vsQ') 
    
    
    plt.figure(2)
    plt.title(r"phi0 vs U/I {A}".format(A = std_dir.split("/")[-2]))
    plt.xlabel(r'$\phi_0 \mathrm{\ [^{\circ}]}$', size=24)
    plt.ylabel(r'$\frac{U}{I} \mathrm{\ [-]}$', size=24)
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.savefig(savedir + '/' + 'phi0vsU')  

    
    plt.figure(3)                      
    plt.title(r"U/I vs Q/I {A}".format(A = std_dir.split("/")[-2]))
    plt.xlabel(r'$\frac{U}{I} \mathrm{\ [-]}$', size=24)
    plt.ylabel(r'$\frac{Q}{I} \mathrm{\ [-]}$', size=24)
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.savefig(savedir + '/' + 'UvsQ')

    
    plt.figure(4)     
    plt.axis(xmin=0, xmax=91)                 
    plt.title(r"$\alpha$ vs f_norm {A}".format(A = std_dir.split("/")[-2]))
    plt.xlabel(r'$\alpha \mathrm{\ [^{\circ}]}$', size=24)
    plt.ylabel(r'$F \mathrm{\ [-]}$', size=24)
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.savefig(savedir + '/' + 'alpha_F')
    
    # Show and close figures
    plt.show()  
    plt.close(0), plt.close(1), plt.close(2), plt.close(3), plt.close(4), plt.close(5)





              
                    













