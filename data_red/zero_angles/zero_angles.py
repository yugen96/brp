import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt



def line(x, xc, yc, a):
    y = a*(x - xc) + yc
    return y
    


# Import retarder chromaticity data
labdaWP, eps_theta = np.loadtxt("ret_chrom.txt", comments="--", delimiter = "   ", skiprows=4, unpack=True) #nm #degrees
labdaWP = labdaWP/10.
# Import b_HIGH and b_bessel transmission data
labdaB, transmB = np.loadtxt("b_HIGH.txt", delimiter = "   ", skiprows=2, unpack=True) #nm #%
labdaBbes, transmBbes = np.loadtxt("M_BESS_B.txt", delimiter = "  ", skiprows=2, unpack=True) #nm #%
transmB = transmB/100. # [--]
# Import v_HIGH transmission data
labdaV, transmV = np.loadtxt("v_HIGH.txt", delimiter = "   ", skiprows=2, unpack=True) #nm #%
labdaVbes, transmVbes = np.loadtxt("M_BESS_V.txt", delimiter = "  ", skiprows=2, unpack=True) #nm #%
transmV = transmV/100. # [--]
# Import B1V star spectral energy density
labdaSED, SED = np.loadtxt("B1V_SED.txt", delimiter = " ", unpack=True) #nm #W m-2 microm-1


# Select data below maximum labdaSED value and above minimum SED value
mask1 = np.array(labdaWP<np.max(labdaSED))*np.array(labdaWP>np.min(labdaSED))
mask2 = np.array(labdaB<np.max(labdaSED))*np.array(labdaB>np.min(labdaSED))
eps_theta, transmB, transmV = eps_theta[mask1], transmB[mask2], transmV[mask2]
labdaWP, labdaB, labdaV = labdaWP[mask1], labdaB[mask2], labdaV[mask2]
# Flip transmission wavelengths and data # TODO FLIP OR DON'T FLIP
labdaB, labdaV = np.flipud(labdaB), np.flipud(labdaV)
transmB, transmV = np.flipud(transmB), np.flipud(transmV)


# Interpolate SED for waveplate wavelengths
SED_INTERP = np.interp(labdaB, labdaSED, SED)


# Interpolation to create same wavelength selection as labdaWP for labdaB and labdaV
convs1, convs2, convs3, convs4_0, convs4_1 = [], [], [], [], []
transm_INTERPs, eps_thetaINTERPs = [], []
transm_INTERPs2, eps_thetaINTERPs2 = [], []
for [labda, transm] in [[labdaB,transmB],[labdaV, transmV]]:
    for i,zipth in enumerate(zip(labda, transm)):
        print(zipth)
    print("\n\n")
    
    # Interpolate transmission filters for waveplate wavelengths  
    transm_INTERP = np.interp(labdaWP, labda, transm) 
    transm_INTERP2 = np.interp(labdaSED, labda, transm)
    transm_INTERPs.append(transm_INTERP)
    transm_INTERPs2.append(transm_INTERP2)
    # Interpolate chromaticity for filter transmission rates wavelengths
    eps_thetaINTERP = np.interp(labda, labdaWP, eps_theta) 
    eps_thetaINTERP2 = np.interp(labdaSED, labdaWP, eps_theta)
    eps_thetaINTERPs.append(eps_thetaINTERP)
    eps_thetaINTERPs2.append(eps_thetaINTERP2)
    
    
    conv1 = np.convolve(eps_thetaINTERP, transm, mode='same')
    conv2 = np.convolve(eps_theta, transm_INTERP, mode='same')
    conv3 = np.convolve(conv1, SED_INTERP, mode='same')
    conv4_0 = np.convolve(eps_thetaINTERP2, transm_INTERP2, mode='same')
    conv4_1 = np.convolve(conv4_0, SED, mode='same')
    
    #Normalize
    conv1, conv2, conv3 = conv1/max(conv1), conv2/max(conv2), conv3/max(conv3)
    conv4_0, conv4_1 = conv4_0/max(conv4_0), conv4_1/max(conv4_1)
    
    # Append to lists
    convs1.append( conv1 )
    convs2.append( conv2 )
    convs3.append( conv3 )
    convs4_0.append( conv4_0 )
    convs4_1.append( conv4_1 )
    

conv0_0 = np.convolve(eps_thetaINTERP, SED_INTERP, mode='same')
conv0_1 = np.convolve(eps_thetaINTERP2, SED, mode='same')
conv0_0, conv0_1 = conv0_0/max(conv0_0), conv0_1/max(conv0_1)

    

# Compute the waveplate chromatic zero angle corresponding to the wavelength of maximum convolution
MAXs, MAXsind = np.amax(convs4_0, axis=1), np.argmax(convs4_0, axis=1)
labdaMAXs = np.array(labdaSED)[MAXsind]
eps_thetaMAXs = np.array(eps_thetaINTERPs2)[[0,1],MAXsind]
print("labdas:\t\t{}".format(labdaMAXs)) 
print("eps_thetas:\t\t{}".format(eps_thetaMAXs))


# Mask for Bessel filters
#####mask3 = np.array(labdaBbes<np.max(labdaSED))*np.array(labdaBbes>np.min(labdaSED))
mask3 = np.array(labdaBbes<7e2)


# Plot of transmission filters and interpolations
fig01 = plt.figure(1)
ax01 = fig01.add_subplot(111)
'''
ax01.plot(labdaWP, transm_INTERPs[0]*100., color='0.3', marker='v')
ax01.plot(labdaWP, transm_INTERPs[1]*100., color='0.5', marker='v')
ax01.plot(labdaSED, transm_INTERPs2[0]*100., color='0.7', marker='v')
ax01.plot(labdaSED, transm_INTERPs2[1]*100., color='0.9', marker='v')
'''
ax01.plot(labdaB, transmB*100., color='b', label="b_HIGH")
ax01.plot(labdaV, transmV*100., color='g', label="v_HIGH")
ax01.plot(labdaBbes[mask3], transmBbes[mask3]*100., color='#66ccff', label="Bessel B")
ax01.plot(labdaVbes[mask3], transmVbes[mask3]*100., color='#99ff33', label="Bessel V")
ax01.axis(ymax=120)
ax01.grid()
ax01.legend(loc='best')
ax01.set_xlabel(r'$\lambda$ [nm]', fontsize=20)
ax01.set_ylabel(r'$T \ [\%]$', fontsize=20)
ax01.legend(loc = 'best')
plt.tight_layout()
plt.savefig('transmissions')


# Plot of chromaticity and interpolated values
fig02 = plt.figure(2)
ax02 = fig02.add_subplot(111)
'''
ax02.scatter(labdaB, eps_thetaINTERPs[0], color='0.5', label="Interpolated")
ax02.scatter(labdaSED, eps_thetaINTERPs2[0], color='0.8', label="Interpolated2")
'''
ax02.plot(labdaWP, eps_theta, color='b')
ax02.scatter(labdaMAXs[0], eps_thetaMAXs[0], color='r', marker="v", s=60., label=r"$\epsilon_{\theta,V}$")
ax02.scatter(labdaMAXs[1], eps_thetaMAXs[1], color='#ff6600', marker="^", s=60., label=r"$\epsilon_{\theta,B}$")
ax02.grid()
ax02.legend(loc='best')
ax02.set_xlabel(r'$\lambda$ [nm]', fontsize=20)
ax02.set_ylabel(r'$\epsilon_{\theta} \ [^{\circ}]$', fontsize=20)
plt.tight_layout()
plt.savefig('chrom')


# Plot of SED and interpolated values
fig02 = plt.figure(3)
ax02 = fig02.add_subplot(111)
'''
ax02.scatter(labdaWP, SED_INTERP, color='0.5', label="Interpolated")
'''
ax02.plot(labdaSED, SED, color='b')
ax02.grid()
ax02.legend(loc='best')
ax02.set_xlabel(r'$\lambda$ [nm]', fontsize=20)
ax02.set_ylabel(r'$SED \ [W m^{-2} \mu m^{-1}]$', fontsize=20)
plt.tight_layout()
plt.savefig('sed')

plt.show()





# Plot of the convolution of the transmission rates with the interpolated chromaticity values
fig1_1 = plt.figure(11)
ax1_1 = fig1_1.add_subplot(111)
ax1_1.plot(labdaB, convs1[0], color='b', label="b_HIGH transm_l")
ax1_1.plot(labdaV, convs1[1], color='g', label="v_HIGH transm_l")
# Plot of the convolution of the interpolated chromaticity with the interpolated filter transmission rates for the SED wavelengths
ax1_1.plot(labdaSED, convs4_0[0], color='#66ccff', label="b_HIGH SED_l")
ax1_1.plot(labdaSED, convs4_0[1], color='#99ff33', label="v_HIGH SED_l")
ax1_1.grid()
ax1_1.legend(loc='best')
ax1_1.set_xlabel(r'$\lambda$ [nm]', fontsize=20)
ax1_1.set_ylabel(r'$T * \epsilon_{{\theta}} \ [^{\circ}]$', fontsize=20)
plt.tight_layout()
plt.savefig('convolutions1') 


'''
# Plot of the convolution of the chromaticity with the interpolated filter transmission rates
fig1_2 = plt.figure(12)
ax1_2 = fig1_2.add_subplot(111)
ax1_2.plot(labdaWP, convs2[0], color='b', label="b_HIGH")
ax1_2.plot(labdaWP, convs2[1], color='g', label="v_HIGH")
ax1_2.grid()
ax1_2.legend(loc='best')
ax1_2.set_xlabel(r'$\lambda$ [nm]', fontsize=20)
ax1_2.set_ylabel(r'$T_{interp} * \epsilon_{\theta} \ [^{\circ}]$', fontsize=20)
plt.tight_layout()
plt.savefig('convolutions2') 
'''


# Plot of the convolution fo the chromaticity with the interpolated SED values
fig13 = plt.figure(13)
ax13 = fig13.add_subplot(111)
ax13.plot(labdaB, conv0_0, color='0.4')
ax13.plot(labdaSED, conv0_1, color='k')
ax13.grid()
ax13.legend(loc='best')
ax13.set_xlabel(r'$\lambda$ [nm]', fontsize=20)
ax13.set_ylabel(r'$SED * \epsilon_{\theta} \ [W m^{-2} \mu m^{-1}]$', fontsize=20)
plt.tight_layout()
plt.savefig('convolutions3') 


# Plot of convolution of chromaticity and filters convolved with SED values
fig14 = plt.figure(14)
ax14 = fig14.add_subplot(111)
ax14.plot(labdaB, convs3[0], color='b', label="b_HIGH transm_l")
ax14.plot(labdaB, convs3[1], color='g', label="v_HIGH transm_l")
ax14.plot(labdaSED, convs4_1[0], color='#66ccff', label="b_HIGH SED_l")
ax14.plot(labdaSED, convs4_1[1], color='#99ff33', label="v_HIGH SED_l")
ax14.grid()
ax14.legend(loc='best')
ax14.set_xlabel(r'$\lambda$ [nm]', fontsize=20)
ax14.set_ylabel(r'$T * SED * \epsilon_{\theta} \ [W m^{-2} \mu m^{-1}]$', fontsize=20)
plt.tight_layout()
plt.savefig('convolutions4')

plt.show()










'''
f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
ax1.plot(labdaB, transmB*100., color='b', label="b_HIGH")
ax2.plot(labdaV, transmV*100., color='g', label="v_HIGH")
plt.grid()
ax1.set_ylabel(r'$T [\%]$', fontsize=20)
ax2.set_xlabel(r'$\lambda$ [nm]', fontsize=20)
ax2.set_ylabel(r'$T [\%]$', fontsize=20)
plt.legend(loc = 'best')
plt.tight_layout()
plt.savefig('transmissions')



f, ax = plt.subplots(2, sharex=True, sharey=True)
ax[0].plot(convB, color='b', label="b_HIGH")
ax[0].grid()
ax[0].legend(loc='best')
ax[1].plot(convV, color='g', label="v_HIGH")
ax[1].grid()
ax[1].legend(loc='best')
ax[0].set_ylabel(r'$T * \epsilon_{\theta}$', fontsize=20)
ax[1].set_xlabel(r'$\lambda$ [nm]', fontsize=20)
ax[1].set_ylabel(r'$T * \epsilon_{\theta}$', fontsize=20)
plt.tight_layout()
plt.savefig('convolutions') 
plt.show()
'''


