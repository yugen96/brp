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





# Specify main directory
os.chdir("..")
maindir = os.getcwd()
# Specify flat files directories
flat_directories = ["/FLAT,LAM", "/FLAT,SKY"]
# Extracts masterbias from the BIAS folder
masterbias_header, masterbias_data = extract_data(maindir + "/masterbias.fits")


# Sorts the flat files of both types(lamp or sky) according to chip number. Also checks whether the flat corresponds to the 'IMG     ' instrument mode and the 'IMAGE   ' observation technique.
for flat_dir in flat_directories:

    # Go to flat images directory
    os.chdir(maindir + flat_dir)
    # Create list with files contained within the directory
    dlst, flst = mk_lsts(maindir + flat_dir)
    
    
    # Sorts each flat in the directory according the chip number
    for f in flst:
        print(f)
        header, data = extract_data(f)
        chipno = header['EXTNAME']
        obstech = header["HIERARCH ESO DPR TECH"]
        
        # Checks whether the flat is taken in the IPOL mode
        if obstech == 'POLARIMETRY':   
            
            movedir = maindir + flat_dir + '_sorted,IPOL/' + chipno
            if os.path.exists(movedir):
                shutil.copy2(f   ,   movedir + '/' + f)
                
            else:
                os.makedirs(movedir)
                shutil.copy2(f    ,   movedir + '/' + f)
        
        elif obstech == 'IMAGE':

            movedir = maindir + flat_dir + '_sorted,IMG/' + chipno
            if os.path.exists(movedir):
                shutil.copy2(f   ,   movedir + '/' + f)
                
            else:
                os.makedirs(movedir)
                shutil.copy2(f    ,   movedir + '/' + f)



'''
# Creates a masterflat for each of the directories specified in flat_directories
for flat_dir in flat_directories:
    # Go to flat images directory
    os.chdir(maindir + flat_dir)
    # Create list with files contained within the directory
    dlst, flst = mk_lsts(maindir + flat_dir)
    
    
    # Create a list containing all the data arrays of the flatfiles contained in flat_dir, calibrated for the bias. Resulting list looks like: [data1, data2, data3, ...] where datai stands for the data array of the i-th flat
    calflat_datalst = []
    flat_headerlst = []
    flat_catgs, flat_sortlst = [], []
    for flat in flst:
        #print(flat)
        flat_header, flat_data = extract_data(flat)
        
        # Subtract masterbias from each flat
        flat_data = flat_data - masterbias_data
        
        # Append calibrated flat image to list of flats and header to headerlist
        calflat_datalst.append(flat_data)
        flat_headerlst.append(flat_header)
        
        

        ########## SORTING TEST ##########
        
        # Check whether flat is within a category of flat_catgs
        flat_CRPIXs = [flat_header["CRPIX1"], flat_header["CRPIX2"]]
        
        
        if len(flat_catgs) == 0:
            flat_catgs.append(flat_CRPIXs)
            flat_sortlst.append([flat])
            
        print(flat_CRPIXs)
        nrOfCatgs = len(flat_catgs)
        for catg_count in range(nrOfCatgs):
            print(len(flat_catgs))
            print(catg_count)
            print(flat_CRPIXs == flat_catgs[catg_count])
            if flat_CRPIXs == flat_catgs[catg_count]:
                print("MATCH")
                flat_sortlst[catg_count].append(flat)                
            else:
                print("MAKE")
                flat_catgs.append(flat_CRPIXs)
                flat_sortlst.append([flat])



        ########## SORTING TEST ##########        
'''   
   

   
     
'''        
    print("PRINTS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    i = 0
    for header in flat_headerlst:
        #print(header["HIERARCH ESO TPL NEXP"], header["HIERARCH ESO TPL EXPNO"])
        #print(header["HIERARCH ESO OBS PI-COI ID"])
        #print(header["CRPIX1"], header["CRPIX2"])
        if 
        i += 1
        
    print("PRINTS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    
    # Compute masterflat
    calflat_datalst = np.array(calflat_datalst)
    masterflat = np.median(calflat_datalst, axis=0)
    
    # Normalization
    norm = np.median(masterflat)
    masterflat_norm = masterflat / norm
    
    # Writes masterflat to separate fits files in the original directory
    os.chdir('..')
    fits.writeto("masterflat_norm{A}.fits".format(A=flat_dir.split('/')[1]), masterflat_norm, clobber=True)
''' 
 
    



    
            



    
    
    
    
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












