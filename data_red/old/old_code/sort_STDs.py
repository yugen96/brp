import numpy as np
import os
import shutil
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
    
   
    
# Checks whether file 'fname' is contained within the specified folder 'filedir' and, if so, copies them to a new folder 'movedir' corresponding to the file's object type
def move_file(fname, datadir, movedir):   
    initdir = os.getcwd()
    os.chdir(datadir)
    
    if os.path.isfile(fname):
        
        if os.path.exists(movedir):
            shutil.copy2(os.getcwd() + '/' + fname     ,       movedir + '/' + fname)
        else:
            os.makedirs(movedir)
            shutil.copy2(os.getcwd() + '/' + fname     ,       movedir + '/' + fname)
    
    os.chdir(initdir)



#################### FUNCTIONS ####################





# Specify current datapath
currpath = os.getcwd()
# Extracts masterbias from the BIAS folder
masterbias_header, masterbias_data = extract_data(currpath + "/masterbias.fits")


# Go to STD directory
os.chdir(currpath + '/STD')
# Create list with files contained within the directory
dlst, flst = mk_lsts(currpath + '/STD')


# Create a dictionary containing all STD files sorted according to celestial coordinates. Resulting dictionary looks as follows: {'[RA1, DEC1]': [STD1.1, STD1.2, STD1.3, ...]  ,   '[RA2, DEC2]': [STD2.1, STD2.2, STD2.3, ...],   etc.}
STD_dict = {}
for std in flst:

    print(std)
    STD_header, STD_data = extract_data(std)
    coord = [STD_header["RA"], STD_header["DEC"]]
    coord_str = str(coord)

    if coord_str in STD_dict.keys():
        STD_dict[coord_str].append(STD_data)
    else:
        STD_dict[coord_str] = [STD_data]
       
   

# Writes each category images to a separate folder
i = 0
for catg in STD_dict.keys():
    movedir = currpath + "/STD_sorted/STD" + str(i)
    print(movedir)
   
    if os.path.exists(movedir):
        for STD_img in STD_dict[catg]:
            shutil.copy2(STD_img   ,   movedir + '/' + STD_img)
            
    else:
        os.makedirs(movedir)
        for STD_img in STD_dict[catg]:
            shutil.copy2(STD_img    ,   movedir + '/' + STD_img)





########################## COADDING CODE ################################
'''    
# Coadd STD images with same celestial coordinates  
i = 0
for catg in STD_dict.keys():

    stacked_img = np.median(STD_dict[catg], axis=0)
    
    # write stacked image to new file in subdirectory
    #os.chd("")
    fits.writeto("STD{A}.fits".format(A=i), stacked_img, clobber=True)
    os.rename("STD{A}.fits".format(A=i), currpath + "/STD_sorted/STD{A}.fits".format(A=i))
    print("written")
    
    i += 1
'''
########################## COADDING CODE ################################


    



    
            



    
    











