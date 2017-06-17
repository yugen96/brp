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


# Mask for bessel filters
mask3 = np.array(labdaBbes<7e2)



# Compute a weighed average over transm*eps_theta
weightedavs = []
for [labda, transm] in [[labdaB,transmB],[labdaV, transmV]]:
    
    eps_thetaINTERP = np.interp(labda, labdaWP, eps_theta) 
    weightedav = np.average(transm*eps_thetaINTERP)#, weights=1./(eps_thetaINTERP**2))
    weightedavs.append( weightedav )


print(weightedavs)



# Plot of transmission filters and interpolations
fig01 = plt.figure(1)
ax01 = fig01.add_subplot(111)
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
ax02.plot(labdaWP, eps_theta, color='b')
ax02.grid()
ax02.legend(loc='best')
ax02.set_xlabel(r'$\lambda$ [nm]', fontsize=20)
ax02.set_ylabel(r'$\epsilon_{\theta} \ [^{\circ}]$', fontsize=20)
plt.tight_layout()
plt.savefig('chrom')


# Plot of SED and interpolated values
fig02 = plt.figure(3)
ax02 = fig02.add_subplot(111)
ax02.plot(labdaSED, SED, color='b')
ax02.grid()
ax02.legend(loc='best')
ax02.set_xlabel(r'$\lambda$ [nm]', fontsize=20)
ax02.set_ylabel(r'$SED \ [W m^{-2} \mu m^{-1}]$', fontsize=20)
plt.tight_layout()
plt.savefig('sed')

plt.show()






