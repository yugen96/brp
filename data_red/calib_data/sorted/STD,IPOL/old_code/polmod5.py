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
    
    
    
# Calculates normalized flux/count differences between the ordinary and extraordinary target spectra or fluxes as specified on p.39 of the FORS2 user manual. N.B.: The fluxes must have been derived for the same angle of the retarder waveplate!
def F(f_or, f_ex):
    
    fluxdiff_norm = (f_or - f_ex) / (f_or + f_ex)
    return fluxdiff_norm
    
    
    
    


#################### FUNCTIONS ####################





# Specify necessary directories
datadir = "/home/bjung/Documents/Leiden_University/brp/data_red/calib_data/sorted"
stddir = datadir + "/STD,IPOL"
veladir = stddir + "/Vela1_95_coord1/CHIP1"
imdir = veladir + "/tpl4"
savedir = "/home/bjung/Documents/Leiden_University/brp/data_red/plots"


# Specify Vela1_95's physical coordinates within the fits files as well as an array specifiying a window of approximate coordinates for the star. Window looks like [[ array(poss_xcoords), array(poss_ycoords) ], [ --- idem for the lower star --- ]]
Vela1_95_coords = [[1033, 355], [1034, 260]]


#TODO automize for all folders in STD_sorted2,IPOL
'''
# Create list of directories and files within veladir
stdstardir_lst, stdfile_lst = mk_lsts(stddir)
for m1 in range(len(stddir)):
    chipdir_lst, chipf_lst = mk_lsts(stdstardir_lst[m1])
    for m2 in chipdir_lst
'''



# Iterates through all the sets of four exposures (each exposure n's retarder waveplate rotated over n*22.5 deg) and calculates the corresponding degrees of polarization for vela1_95
poldeg_lst, polang_lst = [], []

# Create a list with filenames of files stored within tpldir
expdir_lst, expfile_lst = mk_lsts(imdir)
expfile_lst = np.sort(expfile_lst)


# TODO CHANGE COMMENTS!!!!!!!!!!!!!!!!!!
# Predefine lists in which to store wollangle and retangle and starcounts. The template exposure number is specified along axis0, whilst the two different angle types and the different ds9 stellar regions are specified along axis1

# Range of aperture radii for plotting polarisation degree against aperture radius
r_range = np.arange(0.01, 6.01, 0.01)
angle_lst = np.zeros([len(expfile_lst), 2])
count_lst = np.zeros([len(expfile_lst), 2])
Q_lst = np.zeros([len(expfile_lst), len(r_range)])
U_lst = np.zeros([len(expfile_lst), len(r_range)])



    for i in range(len(expfile_lst)):
        f = expfile_lst[i]
        print("\n {}".format(f))
        

        if f.endswith("fits"):
            header, data = extract_data(imdir + '/' + f)
            
            # Specify observation parameters
            expno = header["HIERARCH ESO TPL EXPNO"]
            # Append position angles of retarder waveplate and wollaston prism to angle_lst
            ret_angle = header["HIERARCH ESO INS RETA2 POSANG"]
            woll_angle = header["HIERARCH ESO INS WOLL POSANG"] 
            angle_lst[i, 0] = ret_angle
            angle_lst[i, 1] = woll_angle
            
            
            #TODO REWRITE YOURSELF??? / READ THROUGH DAOSTARFINDER SOURCE CODE???
            # Finds relevant stars in image
            mean, median, std = sigma_clipped_stats(data, sigma=3.0, iters=5)
            daofind = DAOStarFinder(fwhm=3.0, threshold=1000.*std)
            sources = daofind(data - median)
            xcoords, ycoords = list(sources['xcentroid']), list(sources['ycentroid'])
            

            # Pick out Vela1_95
            VelaPosts = []
            for q in range(len(xcoords)):
                # Checks whether there are xcoordinates which correspond to the x-position of Vela1_95
                if np.round(xcoords[q]) in np.arange(1030., 1040., 1.):
                    
                    VelaPosts.append((xcoords[q], ycoords[q]))
            
            
            # Calculate aperture sums for different aperture radii
            for k in range(len(r_range)):      
                R = r_range[k]
                aps = CircularAperture(VelaPosts, r=R)
                ap_sums = list(aperture_photometry(data, aps)['aperture_sum'])
                
          
                # Calculates Q, U and V using the ordinary and extraordinary fluxes at certain retangle for the selected stars
                Q, U = 0, 0
                norm_fluxdiff = F(ap_sums[1], ap_sums[0]) 
                Q = 2/float(len(expfile_lst)) * norm_fluxdiff * np.cos(4*ret_angle)
                U = 2/float(len(expfile_lst)) * norm_fluxdiff * np.sin(4*ret_angle)
        
                # Append to list
                Q_lst[i,k] = Q
                U_lst[i,k] = U



    # Compute the degree and angle of linear polarization at different values of r
    Q_r, U_r = np.sum(Q_lst, axis=0), np.sum(U_lst, axis=0)
    Q_r2, U_r2 = np.square(Q_r), np.square(U_r)
    QU2_r = np.vstack((Q_r2, U_r2))

    P_l__r = np.sqrt(np.sum(QU2_r, axis=0))            
    phi_l__r = (1/2.) * np.arctan(np.divide(Q_r, U_r)) * (180. / np.pi)  



    #### Plots ####
    plt.plot(r_range, P_l__r)
    plt.title(r"Linear polarization of Vela1_95 as function of distance")
    plt.xlabel(r'$\mathrm{Aperture \ radius \ [pixels]}$', size=24)
    plt.ylabel(r'$\mathrm{Degree \ of \ linear \ polarization \ [-]}$', size=24)
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.savefig(savedir + '/Vela1_95tpl4_RvsLinpol.png'.format(A=i))
    plt.show()
    plt.clf()
            





            
'''
        count_lst[i,:] = counts_r



        #### Plots ####
        print(len(counts_r[:,0]))
        print(len(r_range))
        plt.plot(r_range, counts_r[:,0])
        plt.xlabel(r'$\mathrm{r}$', size=24)
        plt.ylabel(r'$\mathrm{Counts \ [-]}$', size=24)
        plt.legend(loc = 'best')
        plt.tight_layout()
        plt.savefig(savedir + '/Vela1coord1_tpl4file{A}_RvsCounts.png'.format(A=i))
        #plt.show()
        plt.clf()



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









