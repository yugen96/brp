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
    


# Extracts the data from a list containing fits files and puts these into
# a list data_lst. # The function needs to be provided with this list of files as well as the
# subdirectory where these files are located in the case they are not stored
# in the same directory as the script. 
def extract_data(file_lst, files_dir = ""):
    data_lst, header_lst = [], []
    
    for x in file_lst:
        print(x)
        
        # Put data and info files into two separate lists
        if x.endswith("fit") or x.endswith("fits"):
            hdul = fits.open(x)
            header = hdul[0].header
            header_lst.append(header)
            data = hdul[0].data
            data_lst.append(data)
            
    return header_lst, data_lst





#################### END FUNCTIONS ####################   



# Specify current folder
currpath = os.getcwd()
# Go to datapath folder
os.chdir(currpath + '/' + 'BIAS')

# Creates a list containing all the bias files
bias_dlst, bias_flst = mk_lsts(os.getcwd())
print(bias_flst)

# Separates list into data files and info files
bias_headers, bias_datas = extract_data(bias_flst)

# Computes the masterbias
bias_datas = np.array(bias_datas)
masterbias = np.median(bias_datas, axis = 0)

print("\n\n", bias_datas)
print("\n\n", masterbias)

# Return to original folder
os.chdir(currpath)
# Writes masterbias to separate fits file
fits.writeto("masterbias.fits", masterbias)










