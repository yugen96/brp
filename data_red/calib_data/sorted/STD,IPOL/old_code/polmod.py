import numpy as np
import fits_pixel_photometry as pixphot
import os
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
datadir = "/home/bjung/Documents/Leiden_University/brp/data_red/calib_data"
stddir = datadir + "/STD_sorted,IPOL"
veladir = stddir + "/Vela1_95_coord1/CHIP1"


# Specify Vela1_95's physical coordinates within the fits files as well as an array specifiying a window of approximate coordinates for the star. Window looks like [[ array(poss_xcoords), array(poss_ycoords) ], [ --- idem for the lower star --- ]]
Vela1_95_coords = [[1033, 355], [1034, 260]]
'''
poss_x = np.arange(1033-50, 1033+50, 1)
poss_y1, poss_y2 = np.arange(355-50, 355+50, 1), np.arange(260-50, 260+50, 1)
poss_y = np.append(poss_y1, poss_y2)
'''


#TODO automize for all folders in STD_sorted2,IPOL
# Create list of directories and files within veladir
tpldir_lst, tplfile_lst = mk_lsts(veladir)


# Iterates through all the sets of four exposures (each exposure n's retarder waveplate rotated over n*22.5 deg) and calculates the corresponding degrees of polarization for vela1_95
poldeg_lst, polang_lst = [], []
for tpldir in tpldir_lst:
    print("\n\n\n")
    print(tpldir)
    
    # Skip template 1: Vela1_95 is on the edge of a slit within this template
    if tpldir != "tpl1":
            

        # Create a list with filenames of files stored within tpldir
        expdir_lst, expfile_lst = mk_lsts(veladir + '/' + tpldir)
        expfile_lst = np.sort(expfile_lst)
        
        
        # Predefine lists in which to store wollangle and retangle and starcounts. The template exposure number is specified along axis0, whilst the two different angle types and the different ds9 stellar regions are specified along axis1
        angle_lst = np.zeros([4,2])
        count_lst = np.zeros([4,3])
        
        
        for i in range(len(expfile_lst)):
            f = expfile_lst[i]
            print("\n")
            print(f)
            

            if f.endswith("fits"):
                header, data = extract_data(veladir + '/' + tpldir + '/' + f)
                
                # Specify observation parameters
                expno = header["HIERARCH ESO TPL EXPNO"]
                # Append position angles of retarder waveplate and wollaston prism to angle_lst
                ret_angle = header["HIERARCH ESO INS RETA2 POSANG"]
                woll_angle = header["HIERARCH ESO INS WOLL POSANG"] 
                angle_lst[i, 0] = ret_angle
                angle_lst[i, 1] = woll_angle
                
                
                
                # Finds relevant stars in image 
                mean, median, std = sigma_clipped_stats(data, sigma=3.0, iters=5)
                
                print(std)
                
                daofind = DAOStarFinder(fwhm=3.0, threshold=1000.*std)
                sources = daofind(data - median)
                xcoords, ycoords = list(sources['xcentroid']), list(sources['ycentroid'])
                
                # Pick out Vela1_95
                VelaPosts = []
                for i in range(len(xcoords)):
                    # Checks whether there are xcoordinates which correspond to the x-position of Vela1_95
                    if np.round(xcoords[i]) in np.arange(1030., 1040., 1.):
                        
                        VelaPosts.append((xcoords[i], ycoords[i]))
                        
                
                aps = CircularAperture(VelaPosts, r=4.)
                ap_sums = list(aperture_photometry(data, aps)['aperture_sum'])
                
                
                
                
                
                
                
                
                
                #### Plots ####
                '''
                norm = ImageNormalize(stretch=SqrtStretch())           
                plt.imshow(data, cmap='Greys', origin='lower', norm=norm)
                aps.plot(color='blue', lw=1.5, alpha=0.5)
                plt.colorbar()
                plt.show()
                '''
                
                
                
                
                
                
    else:
        print("Skipped template 1!")
        


'''
                #TODO Create aperture around central pixel. Calculate counts. Subtract background.
                for coords in Vela1_95_coords:
                    Xmin, Xmax = coords[0]-100, coords[0] + 100
                    Ymin, Ymax = coords[1]-100, coords[1] + 100
                    
                    aproxloc = data[Ymin:Ymax, Xmin:Xmax]
                    Xcent = coords[0] + np.unravel_index(aproxloc.argmax(), aproxloc.shape)[1] - 100
                    Ycent = coords[1] + np.unravel_index(aproxloc.argmax(), aproxloc.shape)[0] - 100
                    print(Xcent, Ycent)
                    
                    centcount = data[Ycent, Xcent]
                    count = centcount
                    
                             
                    print(coords[1], coords[0])
                    print(centcount)
                    # Determine the stellar radius in the image based on the region in which the counts remain >0.022*centcount
                    rad = 0
                    while count > 0.022 * centcount:
                        print(rad)
                        count = data[coords[1]+rad, coords[0]]
                        rad += 1
                        
                    print(rad)
'''
            
         
# Calculates Q, U and V using the ordinary and extraordinary fluxes at certain retangle for the selected stars
Q, U = 0, 0
for k in range(count_lst.shape[0]):
norm_fluxdiff = F(count_lst[k,0], count_lst[k,1]) #TODO Check whether arguments of F should be switched around! ---> Find out which of the MOS stripes belongs to extraordinary/ordinary axis Wollaston prism!!!
Q += 2/count_lst.shape[0] * norm_fluxdiff * np.cos(4*angle_lst[k,0])
U += 2/count_lst.shape[0] * norm_fluxdiff * np.sin(4*angle_lst[k,0])

print("\n\n\n", Q, U)            


# Compute the degree and angle of linear polarization
P_l = np.sqrt(Q**2 + U**2)            
phi_l = (1/2) * np.arctan(U/Q)           
print("\n\n\n", P_l, phi_l)                 

















