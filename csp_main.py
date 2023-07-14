import os
import numpy as np
import numpy.linalg as LA
import itertools # for generating the binary configurations

np.set_printoptions(precision=3)

#M_type = input("\nDo we work with a preset [p] matrix M = ((2,2,0),(0,2,0),(0,0,2)) or a random [r] M? (p/r)")
#W_type = input("\nAre you working in the cluster [c] or in your local [l] workstation?")
M_type = 'p'
W_type = 'c'

''' Functions '''
# dist2 function for computing square of distance (faster than square root)
def dist2(p1,p2):
    '''
    Inputs: numpy array of len(3). Output: square of distance (faster than sq. root) between p1 and p2.
    '''
    sqdist = 0
    for i in range(3):
        sqdist += (p1[i] - p2[i]) * (p1[i] - p2[i])
    return sqdist

def pi1(ctau,Ea,Eb):
    '''
    Output: Provides the one-body contributions pi_0 in the figure expansion.
    Inputs: 
        (1) ctau - Composite array of dtype = d_ctau: spins and corresponding coordinates in crystal basis.
        (2),(3) Ea, Eb - One-body energy contributions of each of
    '''
    return (Ea-Eb)/2 * np.sum(ctau['spin'])

def shell2d(cntr,n=1,csys='crys'):
    '''
    Inputs:
        (1) cntr - center coordinates (in crystal coordinates coordinates) [m1, m2, m3]
        (2) n -  until which nth neighbor to consider. Default n=1 for nearest neighbors.
        (3) csys -  coordinate system of output. Default csys='crys' for crystal coordinates, csys='cart' for Cartesian.
    Output: list of coordinates of nth nearest neighbors in format [sq dist, m1, m2, m3]
    '''
    a = np.array(([0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5])) # primitive Bravais lattice vectors
    b = np.array(([1,1,-1],[1,-1,1],[-1,1,1])) # primitive reciprocal vectors 
    
    # list of nearest-neighbors for FCC in Cartesian coordinates 
    NN = np.array([[i,j]for j in [-0.5,0.5] for i in [-0.5,0.5]]*3)
    a_cart = np.zeros((12,3))
    for k,val in enumerate([0,4,8]):
        a_cart[val:val+4] = (np.insert(NN[val:val+4],k,0,axis=1)) # NN in Cartesian coordinates

    # list of next nearest neighbor (NNN) crystal coordinates 
    a_cart_nnn = np.array(([1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]))
    
    if csys=='cart': # if output is Cartesian coordinates
        cntr_cart = cntr@a # convert crys cntr coordinates to Cartesian
        shell = a_cart + cntr_cart # center the NN pts to the cntr 
        shell_nnn = a_cart_nnn + cntr_cart
    if csys=='crys': # if output is in crys coordinates
        # convert Cartesian to crys coordinates 
        a_crys = a_cart@b # convert NN coordinates to crys
        a_crys_nnn = a_cart_nnn@b 
        shell = a_crys + cntr
        shell_nnn = a_crys_nnn + cntr
    if n==1:
        return shell,'n=1 has no next nearest neighbors.'
    if n==2:
        return shell,shell_nnn

def coordmatch(clist,ref):
    '''
    Output: index of coordinate in clist that matches the ref point.
    Inputs:
        (1) clist - np.array of n 3-D coordinates of shape (n,3).
        (2) ref - 3-D coordinates of shape (1,3)
    '''
    bool_list = np.all(clist==ref,axis=1) # array of booleans. the correct match is the one with all True.
    if np.where(bool_list)[0].size==0: # if no match is found 
        return str('No match')
    else:
        return (np.where(bool_list))[0][0]

# function to determine the spin value given the supercell coordinates 
# input must be the supercell coordinates and their corresponding spin values (comptau)
def pi2_spinmap_frac(ctau,cntr_ctau,M_inv,n=1):
    '''
    Inputs:
        (1) ctau - Composite array of dtype = d_ctau.
        (2) cntr_ctau - Spin-coordinate variable (dtype = d_ctau) of center point.
        (3) M_inv - (3,3) Matrix to extract the coefficients of the shell points in the supercell basis.
        (4) n - Shells (distances) to include. n = 1 (nearest neighbors) by default. [n = 2 chooses next nearest neighbors].
    Outputs:
        (1) Mapped composite array of (spins, coordinates) according to the values from the supercell points.
        (2) Sum of the spin-products  of the mapped array.
    '''
    cntr_crys = cntr_ctau['crys']
    cntr_spin = cntr_ctau['spin']

    # generates the shell of cntr point
    shell2d_ind = int(n - 1) # which of the shell2d to include
    shell = shell2d(cntr_crys,n,csys='crys')[shell2d_ind]

    # generates the fractional coefficients of shell points
    X_shell = shell@M_inv
    
    # mapped coordinates outside of the supercell to its corresponding superlattice point by coord % mult
    mapped_coords = []
    for i,point in enumerate(X_shell):
      diff = ctau['frac']-point # difference of each point with the coords in supercell basis
      diff_rem = diff%1 # just getting the decimal part

      # sets floats close to 0, 1, or -1 to 0 as these should be multiple values 
      diff_rem = np.where(np.abs(diff_rem)<1e-14,0,diff_rem)
      diff_rem = np.where(np.abs(diff_rem-1)<1e-14,0,diff_rem)
      diff_rem = np.where(np.abs(diff_rem+1)<1e-14,0,diff_rem)

      bool_list = np.all(diff_rem==np.array([0.,0.,0.]),axis=1) # True if diff results in integer multiples of the supercell vectors
      mapped_coords.append(ctau['crys'][np.where(bool_list)[0][0]])
    mapped_coords = np.array(mapped_coords)

    # with shell_coord_mapped, use coordmatch to get the corresponding spin values from ctau
    mapped_spin = np.array([ctau['spin'][coordmatch(ctau['crys'],val)] for i,val in enumerate(mapped_coords)])
    spin_prod = mapped_spin*cntr_spin

    return mapped_spin,mapped_coords,np.sum(spin_prod)

# Generate QE file from comptau['spin'],comptau['crys']
def build_input_file(config_arr,A_vectors):
    '''
    Builds input file for QE. config_arr is the set of supercell (frac) coordinates and A_vectors is the matrix of (column) vectors of the supercell.
    '''
    input_file_start = """\n# self-consistent calculation
    &control
        calculation = 'scf'
        restart_mode='from_scratch',
        prefix='CuAu',
        pseudo_dir = '../',
        outdir='tmp.out'
    /
    &system
        ibrav= 0
        celldm(1) = 7.26202853
        nat = """ + str(len(config_arr)) + """
        ntyp = 2
        ecutwfc = 90
        ecutrho = 540
        occupations = 'smearing'
        smearing = 'mv'
        degauss = 0.02
    /
    &electrons
        mixing_mode = 'plain'
        mixing_beta = 0.7
        diagonalization = 'david'
        conv_thr = 1.0d-8
    /
ATOMIC_SPECIES
 Cu 63.546 Cu.pbe-dn-kjpaw_psl.1.0.0.UPF
 Au 196.967 Au.pbe-n-kjpaw_psl.1.0.0.UPF"""

    config_set = """\nATOMIC_POSITIONS crystal\n"""
    for i,v in enumerate(config_arr):
        config_set += """ """ + str(v[0]) + """ """ + str(v[1]) + """ """ + str(v[2]) + """ """ + str(v[3]) + """\n"""
    
    A_trans = np.transpose(A_vectors)
    cell_params = """\nCELL_PARAMETERS alat\n"""
    for j,av in enumerate(A_trans):
        cell_params += """ """ + str(av[0]) + """ """ + str(av[1]) + """ """ + str(av[2]) + """\n"""
    input_file_end = "\nK_POINTS automatic\n5 5 5 0 0 0\n"


    return input_file_start,config_set,cell_params,input_file_end

# Builds script file for running multiple times
def build_shell_script(wtype,i,input_file_start,config_set,cell_params,input_file_end):
	start = "#!/bin/bash\nmkdir -p config"+str(i)+"\ncd config"+str(i)+"\ncat << EOF > CuAu.scf.in"
	if wtype=='l':
		end = """EOF\n~/Desktop/qe-7.1/bin/pw.x < CuAu.scf.in > CuAu.scf.out\nrm -r tmp.out\ncd .."""
	elif wtype=='c':
		end = """EOF\n mpirun -np 4 /home/atan/q-e-qe-7.2/bin/pw.x < CuAu.scf.in > CuAu.scf.out\nrm -r tmp.out\ncd .."""
	script = start + input_file_start + config_set + cell_params + input_file_end + end
	return script



''' Supercell lattice generation '''

a = np.array(([0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]))
b = LA.inv(a) # Stored as columns

## Matrix supercell 

#M = np.random.randint(-3,3,size=(3,3)) # randomized
#if M_type=='r':
#    mtrial = 0
#    while(LA.det(M)==0): # just in case the matrix is singular 
#          mtrial += 1
#          print('Last ',mtrial, 'matrices were singular.')
#          M = np.random.randint(-3,3,size=(3,3))
#else:
M = np.array(([2,2,0],[0,2,0],[0,0,2])) # preset

M_inv = LA.inv(M)
A = M @ a # Supercell in Cartesian coords

A_area,a_area = LA.det(A),LA.det(a)
N_mult = np.abs(A_area/a_area)
print('Supercell volume is %.2f, primitive cell volume is %.2f, and the ratio is %.2f.'%(A_area,a_area,N_mult))

## Lattice generation
x = np.arange(-N_mult,N_mult+1,dtype=int)
y = np.arange(-N_mult,N_mult+1,dtype=int)
z = np.arange(-N_mult,N_mult+1,dtype=int)

coordlist_cart = [] 
coordlist_multiplier = [] 
for i in x:
    for j in y:
        for k in z:
            coordlist_cart.append(i*a[0] + j*a[1] + k*a[2])
            coordlist_multiplier.append([i,j,k])
coords = np.array(coordlist_cart)
coords_crys = np.array(coordlist_multiplier)

# Fractional coefficients (supercell vector basis)
X_crys = coords_crys @ M_inv # M_inv is B_crys
X_crys = np.where(np.abs(X_crys)<1e-14,0,X_crys) # set close values of 0 to 0
X_crys = np.where(np.abs(1-X_crys)<1e-14,1,X_crys) # sets close values of 1 to 1
X_crys = np.where(np.abs(1+X_crys)<1e-14,-1,X_crys) # sets close values of -1 to 1

# Gets indices of points within the supercell (0<=X<1)
tau_ind_crys = np.array([i for i,xval in enumerate(X_crys) if np.all(np.logical_and(X_crys[i]<1,0<=X_crys[i]))])

tau_cart = coords[tau_ind_crys] # Cartesian basis 
tau_crys = coords_crys[tau_ind_crys] # Crystal
tau_frac = tau_crys @ M_inv # Supercell


''' Figure Calculations '''

E_cu = -212.8141 # Copper
E_au = -774.9103 # Gold

#0-body
pi0 = (E_cu + E_au)/2


''' Configuration generation '''

# Composite data structure
d_ctau = np.dtype([('spin','i4'),('cart','f8',(3,)),('crys','f8',(3,)),('frac','f8',(3,))])

# Spin configuration
complist = np.array(list(itertools.product([0,1],repeat=len(tau_crys))))
spinpick = complist[np.random.choice(np.arange(2**len(tau_crys)),size=20)] # chooses 20 configs from 2^N possible ones
print(spinpick)

''' Iterations over configurations''' 

for iconfig,config in enumerate(spinpick):
    print('\n\nconfig #%i'%(iconfig+1))
    comptau = np.array([(spin,tau_cart[j],tau_crys[j],tau_frac[j]) for j,spin in enumerate(config)],dtype=d_ctau)
    print('\nConfiguration is:',comptau['spin'],'\n')
    # Summing all the pi_2 contributions for the supercell
    j2_nn = 0
    j2_nnn = 0
    for i,center in enumerate(comptau):
        j2_nn += pi2_spinmap_frac(comptau,center,M_inv,n=1)[2]
        j2_nnn += pi2_spinmap_frac(comptau,center,M_inv,n=2)[2]
    print('Zero-body contribution in the supercell is %i.'%pi0)
    print('Sum of one-body contributions in the supercell is %i.'%pi1(comptau,E_cu,E_au))
    print('Sum of two-body NN contributions in the supercell is %i.'%j2_nn)
    print('Sum of two-body NNN contributions in the supercell is %i.'%j2_nnn)
    printout = np.array((pi0,pi1(comptau,E_cu,E_au),j2_nn,j2_nnn))
    np.savetxt('printout_config_%i'%(iconfig+1),printout)
    

    
    ''' Regression with DFT data '''

    # Sort spins to be ready-made in the QE input file
    comptau_spinsorted = comptau[np.argsort(comptau['spin'])]
    config_species = []
    for i,spin in enumerate(comptau_spinsorted['spin']):
        if spin==0:
            config_species.append('Cu')
        else:
            config_species.append('Au')
    config_list = [(config_species[i],val['frac'][0],val['frac'][1],val['frac'][2]) for i,val in enumerate(comptau_spinsorted)] # for ATOMIC_POSITIONS crystal
    print('\n')


    ''' Building script file '''
    
    input_file_start,config_set,cell_params,input_file_end = build_input_file(config_list,A)
    # print(input_file_start,config_set,input_file_end,'\n')
    # print(config_list,'\n')
    script = build_shell_script(W_type,iconfig+1,input_file_start,config_set,cell_params,input_file_end)
    os.system(script)
