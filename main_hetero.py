import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi, floor
from os import system

#Initializing all constants and variables
N = 400 #No. of particles
re = 1 #Equilibrium Radius
tTot = 2*10**6 #Total Time Steps
eta = 1 #Noise Intensity
r0 = 1.32*re #Maximum range
rc = 0.4*re #Core Radius
v0 = 0.1*re #Velocities i.e. distance per time unit dt
dt = 1 #Time step
l = ((N*pi*re**2)**0.5)/2 #Size of box
alpha = 0.01 #Relative weight of alignement interaction
beta22 = 2.5 #Relative weight of radial 2 body force between ectoderms
beta12 = 2.53 #Relative weight of radial 2 body force between ectoderm and endoderm
beta11 = 3.83 #Relative weight of radial 2 body force between endoderms
fnmInf = 10**5 #Replacement for infinity
tTrap = 4000 #Time step after which seeding of endoderms starts
saveTime = 200 #After what time steps to save data

#Initializing random positions and directions of each particle @t=0
xcords = np.random.random(N)*l #Initialize x coordinate anywhere between 0 to l
ycords = np.random.random(N)*l #Initialize y coordinate anywhere between 0 to l
thcords = 2*pi*np.random.random(N) #Initialize theta anywhere between 0 to 2pi

#Initialize type of all particles 0 is ectodermic and 1 is endodermic 3:1
Nend = floor(N/3)
Nect = N-Nend
typ = np.ones(Nend+Nect)
typ[:Nect] = 0
np.random.shuffle(typ)

#Maintain a dataframe for plotting data
cords = pd.DataFrame(columns = ['x','y','th','typ','gam']) #Maintaining a dataframe for each particle position (x,y) and direction (th)
cords.x = xcords
cords.y = ycords
cords.th = thcords
cords.typ = typ
ectCords = cords[cords.typ==0]
endCords = cords[cords.typ==1]

gam = np.zeros(N)

#Maintain record of gammas at save time instances
gamDic = pd.DataFrame(columns = ['gamma'])

for t in range(tTot):
	# if t%100==0:
	# 	system('cls')
	# 	print(t)
		
	xcords = xcords + v0*np.cos(thcords)
	ycords = ycords + v0*np.sin(thcords)

	for n in range(N):
		xn = xcords[n]
		yn = ycords[n]
		
		inm = 0 #average vector in x direction due to mth particle influence
		jnm = 0 #average vector in y direction due to mth particle influence

		#ngam is total non equal neighbours
		ngam = 0
		#neighTot is total number of neighbours
		neighTot = 0

		for m in range(N):
			if m == n:
				continue
			xm = xcords[m]
			ym = ycords[m]
			thm = thcords[m]
					
			rnm = ((xn-xm)**2+(yn-ym)**2)**0.5
			
			if rnm>r0:
				continue
			else:
				neighTot = neighTot+1
				if typ[m] != typ[n]:
					ngam = ngam+1

			ex = (xn-xm)/rnm
			ey = (yn-ym)/rnm

			if rnm<rc:
				fnm = fnmInf
				anm = 1
			elif (rnm<r0)&(rnm>=rc):
				fnm = 1 - rnm/re
				anm = 1
			else:
				fnm = 0
				anm = 0

			if (t>=tTrap)&(typ[n]!=typ[m]):
				beta = beta12
			elif (t>=tTrap)&(typ[n]==typ[m])&(typ[n]==0):
				beta = beta22
			elif (t>=tTrap)&(typ[n]==typ[m])&(typ[n]==1):
				beta = beta11
			else:
				beta = beta22
			
			inm = inm + alpha*anm*np.cos(thm) + beta*fnm*ex 
			jnm = jnm + alpha*anm*np.sin(thm) + beta*fnm*ey
		
		if neighTot != 0:
			gam[n] = ngam/neighTot
		else:
			gam[n] = 1

		randAngle =  np.random.random()*2*pi
		inoise = eta*np.cos(randAngle)
		jnoise = eta*np.sin(randAngle)
		
		angleNew = np.arctan2(jnm+jnoise,inm+inoise)
		if angleNew<0:
			angleNew = angleNew + 2*pi
		thcords[n] = angleNew
		
	if t%saveTime==0:
		cords.x = xcords
		cords.y = ycords
		cords.th = thcords
		cords.typ = typ
		cords.gam = gam
		ectCords = cords[cords.typ==0]
		endCords = cords[cords.typ==1]

		gam1 = endCords.gam.mean()
		gam2 = ectCords.gam.mean()
		gamDic.loc[t,'gamma'] = gam1
		system('cls')
		print(t,gam1)
		gamDic.to_csv(f'cellSorting5/gamma.csv')

		# plt.quiver(ectCords.y,ectCords.x,np.cos(ectCords.th),np.sin(ectCords.th),color='tab:orange')
		# plt.quiver(endCords.y,endCords.x,np.cos(endCords.th),np.sin(endCords.th),color='tab:purple')
		plt.scatter(ectCords.y,ectCords.x,color='tab:orange')
		plt.scatter(endCords.y,endCords.x,color='tab:purple')
		plt.xlim(-0.25*l,l*1.25)
		plt.ylim(-0.25*l,l*1.25)
		plt.savefig(f"cellSorting5/{t}.png",dpi=500)
		plt.clf()