import numpy as np
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



# Checks whether fname is a fits file and extracts the corresonding header and data if this is the case
def extract_data(fname):

    # Put data and header files into two separate lists
    if fname.endswith("fit") or fname.endswith("fits"):
        hdul = fits.open(fname)
        header = hdul[0].header             
        data = hdul[0].data
        hdul.close()
        
    return header, data



#################### FUNCTIONS ####################





# Specify current datapath
currpath = os.getcwd()
# Specify flat files directories
flat_directories = ["/sorted/FLAT,LAM_IPOL", "/sorted/FLAT,SKY_IMG"]
# Specify chip numbers
chipnos = ["/CHIP1", "/CHIP2"]
# Extracts masterbias from the BIAS folder
masterbias_header, masterbias_data = extract_data(currpath + "/masterbias.fits")



# Creates a masterflat for each of the directories specified in flat_directories
for flat_dir in flat_directories:
    for chipno in chipnos:
        # Go to flat images directory
        os.chdir(currpath + flat_dir + chipno)
        # Create list with files contained within the directory
        dlst, flst = mk_lsts(currpath + flat_dir + chipno)
        
        
        # Create a list containing all the data arrays of the flatfiles contained in flat_dir, calibrated for the bias. Resulting list looks like: [data1, data2, data3, ...] where datai stands for the data array of the i-th flat
        calflat_datalst = []
        flat_headerlst = []
        expt_lst = []
        for flat in flst:
            print(flat)
            flat_header, flat_data = extract_data(flat)
            
            # Read exposure time
            expt = flat_header["EXPTIME"] #TODO Correct for different exposure times? -> Probably not necessary since I already take a median along al images.
              
            # Subtract masterbias from each flat
            flat_data = flat_data - masterbias_data
            
            # Append calibrated flat image to list of flats and header to headerlist
            calflat_datalst.append(flat_data)
            flat_headerlst.append(flat_header)
            expt_lst.append(expt)
    
 
        # Compute masterflat
        calflat_datalst = np.array(calflat_datalst)
        masterflat = np.median(calflat_datalst, axis=0)
        
        # Normalization
        norm = np.median(masterflat)
        masterflat_norm = masterflat / norm
        
        print(norm)
        print(masterflat_norm)
     
        # Writes masterflat to separate fits files in the original directory
        os.chdir(currpath + "/masterflats")
        fits.writeto("masterflat_norm{A}.fits".format(A = '_' + flat_dir.split('/')[1] + '_' + chipno.split('/')[1]), masterflat_norm, clobber=True)

 
    



    
            



    
    
    
    
'''
# Creates an individual masterflat for each of the different filters
for x in flat_dlst:
    print("Subdirectory: \t", x)
    
    # Goes to subdirectory x
    print(os.getcwd() + "/" + x)
    os.chdir(os.getcwd() + "/" + x)
    
    
    # Creates a list containing all the flat files within subdirectory x
    flat_sub_dlst, flat_sub_flst = mk_lsts(os.getcwd())
    
    
    # Extracts data from the list of files
    flat_data = extract_data(flat_sub_flst)
    
    
    # Goes back to main FLAT directory
    os.chdir("..")
    
    
    # Computes the normalised masterflat
    
    for i in range(len(flat_data)):
        flat_data[i] = flat_data[i] - masterbias[0]

    flat_data = np.array(flat_data) 
    masterflat = np.median(flat_data, axis = 0) 
    norm = np.median(masterflat)
    masterflat_norm = masterflat/norm
        
    
    print("\n\n", flat_data)
    print("\n\n", masterflat)
    print("\n\n", norm)
    print("\n\n", masterflat_norm)


    # Writes masterflat to separate fits files
    fits.writeto("masterflat_norm{A}.fits".format(A=x[4::]), masterflat_norm, clobber=True)
'''












