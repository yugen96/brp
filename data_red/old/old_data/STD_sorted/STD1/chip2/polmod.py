import numpy as np
import sp
import fits_pixel_photometry as pixphot
import os
from astropy.io import fits





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
'''   
    elif fname.endswith("reg"):
        xc, yc, r = sp.regextract(fname)
        
        return xc, yc, r 
'''



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





# Specify current directory
currpath = os.getcwd()
# Create list of directories and files within current directory
dir_lst, file_lst = mk_lsts(currpath)


# Predefine lists in which to store wollangle and retangle and starcounts. The template exposure number is specified along axis0, whilst the two different angle types and the different ds9 stellar regions are specified along axis1
angle_lst = np.zeros([4,2])
count_lst = np.zeros([4,3])

data_lst, header_lst = [], []
for f in file_lst:
    print(f)

    if f.endswith("fit") or f.endswith("fits"):
        header, data = extract_data(f)
        data_lst.append(data)
        header_lst.append(header)
        
        # Specify observation parameters
        expno = header["HIERARCH ESO TPL EXPNO"]
        gain = header["HIERARCH ESO DET OUT1 GAIN"]
        t_exp = header["EXPTIME"]
        
        # Append position angles of retarder waveplate and wollaston prism to angle_lst
        ret_angle = header["HIERARCH ESO INS RETA2 POSANG"]
        woll_angle = header["HIERARCH ESO INS WOLL POSANG"] 
        angle_lst[expno-1, 0] = ret_angle
        angle_lst[expno-1, 1] = woll_angle
                
        
        # Calculates the counts for the selected star regions within the data of 'f' and appends the result to count_lst
        x_lst, y_lst, r_lst = sp.regextract("sel_stars{A}.reg".format(A = expno))
        for i in range(len(x_lst)):
        
            counts = pixphot.apphot(data, x_lst[i], y_lst[i], r_lst[i], r_lst[i]+5, r_lst[i]+15)
            count_lst[expno-1, i] = counts
            

# Calculates Q, U and V using the ordinary and extraordinary fluxes at certain retangle for the selected stars
print("\n\n\n", angle_lst) 
print("\n\n\n", count_lst) 

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



'''
        for i in range(len(r)):
            print(r[i])
            flux, eflux, sky, skyerr = sp.aper(data, xc=[x[i]], yc=[y[i]], phpadu=1., apr= [r[i]], skyrad=[35, 45], exact=True, flux=True, setskyval=0.)
            print(flux)

        print(data) 
        x, y, flux, sharpness, roundness = sp.find(data, 300, 15)
        print("poo")
        flux, eflux, sky, skyerr = sp.aper(data, x, y, gain, [5], [5, 10], flux=True)
      
        print(flux)
        

      
    elif f.endswith("reg"):
        nr = int(f[9])
        print(nr)
        xc, yc, r = extract_data(f)

        for i in range(np.ma.size(flux_lst, axis=0)):
            flux_lst[nr, i], eflux_lst[nr, i], sky, skyerr = sp.aper()
'''    
    



'''
# Extracts data and headers from the fits files in the current directory
data_lst, header_lst = [], []
for f in file_lst:
    header, data = extract_data(f)
    data_lst.append(data)
    header_lst.append(header)


# Extract data and header from file 
f = file_lst[0]
header, data = extract_data(f)   
# Specify the positioning angle of the retarder waveplate and the Wollaston prism
ret_angle = header["HIERARCH ESO INS RETA2 POSANG"]
woll_angle = header["HIERARCH ESO INS WOLL POSANG"]
'''

















