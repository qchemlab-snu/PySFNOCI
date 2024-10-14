from pyscf.sfnoci.SFNOCI import SFNOCI


from pyscf import scf, gto
import numpy
mol = gto.Mole()
mol.verbose = 5
mol.output = None
mol.atom = [['Li', (0, 0, 0)],['F',(0,0,1.4)]]
mol.basis = 'ccpvdz'

#High spin for Spin-Flip
mol.spin=2
mol.build(0,0)
rm=scf.ROHF(mol)

############ Reference MO setting################ 
molr = gto.Mole()
molr.verbose = 5
molr.output = None
molr.atom = [['Li', (0, 0, 0)],['F',(0,0,1.3)]]
molr.basis = 'ccpvdz'
mr=scf.RHF(molr)
mr.kernel()

mo0=mr.mo_coeff
occ=mr.mo_occ
setocc=numpy.zeros((2,occ.size))
setocc[:,occ==2]=1
setocc[1][3]=0
setocc[0][6]=1
ro_occ=setocc[0][:]+setocc[1][:]
dm_ro=rm.make_rdm1(mo0,ro_occ)
rm=scf.addons.mom_occ(rm,mo0,setocc)
rm.scf(dm_ro)
mo=rm.mo_coeff
#######################################################

#Active space setting
AS_list=[3,6,7,10]

#grouping by electron number in lowdin basis
# groupA = 'LI' means that the number of active electrons in Li atom becomes criterion of grouping.
mySFNOCI = SFNOCI(rm,4,4,groupA = 'Li')

#or grouping by occupation number of each orbital, It is recommended that the active orbitals are localied or semi-localized. 
mySFNOCI = SFNOCI(rm,4,4,groupA = [[0,1],[2,3]])

#target spin state
mySFNOCI.spin = 0

from pyscf.mcscf import addons
mo = addons.sort_mo(mySFNOCI,rm.mo_coeff, AS_list,1)
reei, ev = mySFNOCI.kernel(mo, nroots = 4)
print(reei,ev)