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





# Specify the main datapath and the science data directory
os.chdir("..")
currpath = os.getcwd()
scipath = currpath + "/calib_data/unsorted/NGC4696"
sortpath = currpath + "/calib_data/sorted"


# Sorts the SCIENCE files according to chip number and celestial coordinates. Also checks whether the image corresponds to the 'IMG     ' instrument mode and the 'IMAGE   ' observation technique.
dlst, flst = mk_lsts(scipath)
# Sorts each image in the directory according to its chip number
tplstart_lst = []
for f in flst:
    print(f)
    
    os.chdir(scipath)
    header, data = extract_data(f)
    chipno = header['EXTNAME']
    obstech = header["HIERARCH ESO DPR TECH"]
    
    
    
    # Checks whether the flat is taken in the IPOL mode and with which retarder waveplate angle
    if obstech == 'POLARIMETRY':   
        
        retang = header["HIERARCH ESO INS RETA2 POSANG"]
        tplstart = header["HIERARCH ESO TPL START"]
        
        # Append template start time to tplstart_lst if not already contained
        if tplstart not in tplstart_lst:
            tplstart_lst.append(tplstart)
        n = tplstart_lst.index(tplstart)
        
        
        movedir = sortpath + "/NGC4696,IPOL/" + chipno + '/tpl{}'.format(n+1) 
        if os.path.exists(movedir):
            shutil.copy2(f   ,   movedir + '/' + f)
            
        else:
            os.makedirs(movedir)
            shutil.copy2(f    ,   movedir + '/' + f)
    
    
    
    elif obstech == 'IMAGE':

        movedir = sortpath + "/NGC4696,IMG/" + chipno
        if os.path.exists(movedir):
            shutil.copy2(f   ,   movedir + '/' + f)
            
        else:
            os.makedirs(movedir)
            shutil.copy2(f    ,   movedir + '/' + f)

    os.chdir("..")













