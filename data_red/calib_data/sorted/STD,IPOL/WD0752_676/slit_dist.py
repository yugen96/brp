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

# Create list with chip directories within currpath
chipdir, flst = mk_lsts(currpath)

for chipno in chipdir:
    print("\n\n\n")
    print(chipno)
    dlst, flst = mk_lsts(currpath + '/' + chipno)
    

    for f in flst:
        mospos_lst = []
        header, data = extract_data(currpath + '/' + chipno + '/' + f)
        
        for i in np.arange(1, 20, 1):
            mospos = header["HIERARCH ESO INS MOS{a} POS".format(a = i)]
            mospos_lst.append(mospos)
            
        mospos_lst = list(set(mospos_lst))                
        mospos_lst.sort()
        diff = np.diff(mospos_lst)
        print("")
        print(list(set(diff)))
        print(len(mospos_lst))
        #print(diff)

    

'''
# Sorts the STD files of both types(lamp or sky) according to chip number and celestial coordinates. Also checks whether the image corresponds to the 'IMG     ' instrument mode and the 'IMAGE   ' observation technique.
# Go to flat images directory
os.chdir(currpath + "/STD")
# Create list with files contained within the directory
dlst, flst = mk_lsts(currpath + "/STD")


# Sorts each flat in the directory according the chip number
for f in flst:
    print(f)
    header, data = extract_data(f)
    chipno = header['EXTNAME']
    obstech = header["HIERARCH ESO DPR TECH"]
    coord = [header["RA"], header["DEC"]]
    coord_str = str(coord)
    
    # Checks whether the flat is taken in the IPOL mode
    if obstech == 'POLARIMETRY':   
        
        movedir = currpath + "/STD_sorted3,IPOL/STD" + coord_str + '/' + chipno
        if os.path.exists(movedir):
            shutil.copy2(f   ,   movedir + '/' + f)
            
        else:
            os.makedirs(movedir)
            shutil.copy2(f    ,   movedir + '/' + f)
    
    elif obstech == 'IMAGE':

        movedir = currpath + "/STD_sorted3,IMG/STD" + coord_str + '/' + chipno
        if os.path.exists(movedir):
            shutil.copy2(f   ,   movedir + '/' + f)
            
        else:
            os.makedirs(movedir)
            shutil.copy2(f    ,   movedir + '/' + f)
'''














