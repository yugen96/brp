import numpy as np
import os
import shutil
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import SqrtStretch
from mpl_toolkits.axes_grid1 import make_axes_locatable




#################### FUNCTIONS ####################



# Returns a list will all subdirectories in a specific folder as well as the
# contained files
def mk_lsts(dir_name):

    #dir_lst = [dirs[0] for dirs in os.walk(dir_name)]
    dir_lst = next(os.walk(dir_name))[1]
    file_lst = next(os.walk(dir_name))[2]
    return dir_lst, file_lst



# Checks whether fname is a fits file and extracts the corresonding header and data if this is the case
def extract_data(fname):

    # Put data and header files into two separate lists
    if fname.endswith("fit") or fname.endswith("fits"):
        hdul = fits.open(fname)
        header = hdul[0].header             
        data = hdul[0].data
        hdul.close()
        
    return header, data



# Function which computes O - E for all slits and appends the results to recreate a single image of the sky
def append_slits(slitdata):
    chipdata = slitdata[10:934:,183:1868]
    rowsno = chipdata.shape[0]


    # Read out chipdata row for row and cut out slits
    derivs = []
    slits = []
    itnos = np.arange(1, rowsno-2, 1)
    for i in itnos:
        print("pixelrow:\t", i)
        
        row, next1row, next2row = chipdata[i,:], chipdata[i+1,:], chipdata[i+2,:]    
        rowmed, next1rowmed, next2rowmed = np.median(row), np.median(next1row), np.median(next2row)   
        
        deriv = next1rowmed - rowmed
        nextderiv = next2rowmed - next1rowmed
        derivs.append(deriv)
                
        # Cut out slits and extraordinary and ordinary beams from the data array
        if np.abs(deriv) > 20. and np.abs(nextderiv) < 20.:          
            cutstart = i
                
        if np.abs(deriv) < 20. and np.abs(nextderiv) > 20.:
            cutend = i
            
            # Skips the first peak in the derivatives, so that slit can be cut out correctly
            try:
                slit = chipdata[ cutstart:cutend, : ]
                if slit.shape[0]>10:
                    slits.append(slit)
            except NameError:
                print("first max")
    
    
    '''
    plt.figure(0)
    plt.scatter(itnos, derivs)
    plt.show()
    plt.close()
    '''

    print("\n\n")
    for n in np.arange(0, len(slits), 2):
        
        # Check that each slit contains the same number of pixel rows
        if slits[n+1].shape[0] < slits[n].shape[0]:
            newxlen = slits[n+1].shape[0]
            slits[n] = slits[n][0:newxlen, :]
            
        elif slits[n+1].shape[0] > slits[n].shape[0]:
            newxlen = slits[n].shape[0]
            slits[n+1] = slits[n+1][0:newxlen, :]
        
        
        # Compute the normalized difference between the ordinary and extroardinary slit (or the other way arround?)
        slit_diff = slits[n] - slits[n+1]
        slit_sum = slits[n] + slits[n+1]
        cal_slit =  slit_diff / slit_sum
        if n == 0:
            cal_slits = cal_slit
        else:
            cal_slits = np.concatenate((cal_slits, cal_slit), axis=0)

    return cal_slits
       


#################### FUNCTIONS ####################



# Specify data and filename
currdir = os.getcwd()
datapath = currdir + "/NGC4696,IPOL/CHIP1"
expdir_lst, expdata_lst = mk_lsts(datapath)
# Specify bias and masterflat
os.chdir("..")
CALdir = os.getcwd()
header, Mbias = extract_data(CALdir + "/masterbias.fits")
header, Mflat_norm = extract_data(CALdir + "/masterflats/masterflat_norm_FLAT,LAM_IPOL_CHIP1.fits")
os.chdir(currdir)

# Read in files
for f in expdata_lst:
    print("\t\t{}".format(f))
    header, data = extract_data(datapath + '/' +  f)
    datacal = (data - Mbias) / Mflat_norm # Calibrated data
    # Append slits
    cal_slits = append_slits(datacal)


    # Save to fits file
    savedir = datapath + "/appended"
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    hdu = fits.PrimaryHDU(cal_slits)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(savedir + "/{}.fits".format(f.split(".fits")[0] + "_APPENDED"))


############### PLOTS #################



'''
for n in np.arange(0, len(slits), 1):   
    norm = ImageNormalize(stretch=SqrtStretch())  
    plt.imshow(slits[n], cmap='Greys', origin='lower', norm = norm)
    plt.colorbar()
    plt.show()
'''

plt.figure()
ax = plt.gca()
norm = ImageNormalize(stretch=SqrtStretch())  
im = ax.imshow(cal_slits, cmap='Greys', origin='lower', norm = norm)
ax.set_xlim([600,1100])
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)

plt.tight_layout()
plt.show()


        
























