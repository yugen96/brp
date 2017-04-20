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
def F(f_or, f_ex, sigma_or, sigma_ex):
    
    # Compute normalized flux difference
    flux_sum = f_or + f_ex
    fluxdiff_norm = (f_or - f_ex) / flux_sum
    
    # Compute standard deviation
    temp_or = (f_ex - f_or*f_ex)**2 / flux_sum**4
    temp_ex = (f_or - f_or*f_ex)**2 / flux_sum**4
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
for j in range(len(std_dirs)): 
    std_dir = std_dirs[j]
    print("\n\n\n{}".format(std_dir))
    
    # Create a list with all the template directories within std_dirs
    tpl_dirlst, tpl_flst = mk_lsts(std_dir)
    tpl_dirlst = np.sort(tpl_dirlst)
    
    
    for tpl_name in tpl_dirlst:
        print("\n\n\t{}".format(tpl_name)) 
        tpl_dir = std_dir + '/' + tpl_name
        
        if (tpl_dir != std_dirs[0] + "/tpl1"):
            
            # Create a list with filenames of files stored within tpldir
            expdir_lst, expfile_lst = mk_lsts(tpl_dir)
            expfile_lst = np.sort(expfile_lst)
            

            # Range of aperture radii for plotting polarisation degree against aperture radius
            r_range = np.arange(0.01, 6.01, 0.01)
            # Lists for storing the different Q and U values of each exposure within each template for each aperture radius
            Q_lst = np.zeros([len(expfile_lst), len(r_range)])
            sigmaQ_lst = np.zeros([len(expfile_lst), len(r_range)])
            U_lst = np.zeros([len(expfile_lst), len(r_range)])
            sigmaU_lst = np.zeros([len(expfile_lst), len(r_range)])


            for i in range(len(expfile_lst)):
                f = expfile_lst[i]
                print("\n\t\t {}".format(f))
                
                if f.endswith("fits"):
                    header, data = extract_data(tpl_dir + '/' + f)
                    
                    # Specify observation parameters
                    expno = header["HIERARCH ESO TPL EXPNO"]
                    filt_name = header["HIERARCH ESO INS FILT1 NAME"]
                    filt_id = header["HIERARCH ESO INS FILT1 ID"]
                    ret_angle = header["HIERARCH ESO INS RETA2 POSANG"]
                    woll_angle = header["HIERARCH ESO INS WOLL POSANG"] 
                    print("\t\t\t\tFILTER_ID: {A}; \t FILTER_NAME: {B}".format(A=filt_id, B = filt_name))
                    print("\t\t\t\Wollangle: {A}; \t Retangle: {B}".format(A=woll_angle, B = ret_angle))
                    
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
                        if np.round(xcoords[q]) in np.arange(aprox_xc[j]-8., aprox_xc[j]+7., 1.):
                            posts.append((xcoords[q], ycoords[q]))
                    
                    
                    #TODO Compute error margins
                    # Calculate aperture sums for different aperture radii
                    for k in range(len(r_range)):      
                        R = r_range[k]
                        # Compute cumulative counts within aperture
                        aps = CircularAperture(posts, r=R)
                        ap_sums = list(aperture_photometry(data, aps)['aperture_sum'])
                        # Compute photon shot noise within aperture
                        shotnoise = np.sqrt(ap_sums)
                        
                    
                        # Calculates Q, U and V using the ordinary and extraordinary fluxes at certain retangle for the selected stars + their standar deviations
                        Q, U = 0, 0
                        norm_fluxdiff, sigma_F = F(ap_sums[1], ap_sums[0], shotnoise[1], shotnoise[0]) 
                        N = float(len(expfile_lst))
                        Q = 2/N * norm_fluxdiff * np.cos(4*ret_angle)
                        sigmaQ = 4/(N**2) * (np.cos(4*ret_angle))**2 * sigma_F**2
                        U = 2/N * norm_fluxdiff * np.sin(4*ret_angle) 
                        sigmaU = 4/(N**2) * (np.sin(4*ret_angle))**2 * sigma_F**2
                
                        # Append to list
                        Q_lst[i,k] = Q
                        sigmaQ_lst[i,k] = sigmaQ
                        U_lst[i,k] = U             #TODO Check whether to compute Stokes parameter I
                        sigmaU_lst[i,k] = sigmaU



                # Compute the degree and angle of linear polarization at different values of r
                Q_r, U_r = np.sum(Q_lst, axis=0), np.sum(U_lst, axis=0)
                Q_r2, U_r2 = np.square(Q_r), np.square(U_r)
                QU2_r = np.vstack((Q_r2, U_r2))
                Q2plusU2_r = np.sum(QU2_r, axis=0)

                P_l__r = np.sqrt(Q2plusU2_r)            
                phi_l__r = (1/2.) * np.arctan(np.divide(Q_r, U_r)) * (180. / np.pi)   
                #TODO CHECK INSTRUMENTAL ROTATION ANGLES              
                
                
                # Compute errors
                varQ_r, varU_r = np.sum(sigmaQ_lst, axis=0), np.sum(sigmaU_lst, axis=0)
                varP_l__r = np.divide(Q_r2, Q2plusU2_r)*varQ_r + np.divide(U_r2, Q2plusU2_r)*varU_r
                sigmaP_l__r = np.sqrt(varP_l__r)
                
                
                


            #### Plots ####
            savedir = plotdir + "/errorbars/" + std_dir.split("/")[-2]
            if not os.path.exists(savedir):
                os.chdir(plotdir)
                os.makedirs(savedir)
                os.chdir(stddatadir)
            
            plt.plot(r_range, P_l__r)
            plt.title(r"Polarization Profile {A} {B}".format(A = std_dir.split("/")[-2], B = tpl_name))
            plt.xlabel(r'$\mathrm{Aperture \ radius \ [pixels]}$', size=24)
            plt.ylabel(r'$\mathrm{Degree \ of \ linear \ polarization \ [-]}$', size=24)
            plt.legend(loc = 'best')
            plt.tight_layout()
            plt.savefig(savedir + '/' + 'RvsPl_{B}'.format(B=tpl_name))
            plt.show()
            plt.clf()
                





            
'''


# Calculates Q, U and V using the ordinary and extraordinary fluxes at certain retangle for the selected stars
Q, U = 0, 0
for k in range(count_lst.shape[0]):
    norm_fluxdiff = F(count_lst[k,0], count_lst[k,1]) 
    Q += 2/float(count_lst.shape[0]) * norm_fluxdiff * np.cos(4*angle_lst[k,0])
    U += 2/float(count_lst.shape[0]) * norm_fluxdiff * np.sin(4*angle_lst[k,0])


# Compute the degree and angle of linear polarization
P_l = np.sqrt(Q**2 + U**2)            
phi_l = (1/2.) * np.arctan(U/Q) * (180. / np.pi)  
poldeg_lst.append(P_l)
polang_lst.append(phi_l)      
print("[P_l, phi_l] = {b}".format(b = [P_l, phi_l]))                  





        
linpol = np.mean(poldeg_lst)
polang = np.mean(polang_lst)
print("\n\n\n[linpol, polang] = {c}".format(c = [linpol, polang]))                  
'''









