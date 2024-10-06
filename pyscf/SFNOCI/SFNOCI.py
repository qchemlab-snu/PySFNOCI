import sys

import numpy
import scipy.linalg
from scipy.linalg import eigh as gen_eig
import ctypes
from pyscf import scf
from pyscf import fci
from pyscf.fci import cistring
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import chkfile
from pyscf import __config__
from itertools import product
from pyscf import ao2mo
from pyscf.fci import fci_slow
from pyscf.mcscf.casci import CASBase, CASCI
from pyscf.fci import spin_op

WITH_META_LOWDIN = getattr(__config__, 'scf_analyze_with_meta_lowdin', True)
PRE_ORTH_METHOD = getattr(__config__, 'scf_analyze_pre_orth_method', 'ANO')
MO_BASE = getattr(__config__, 'MO_BASE', 1)
TIGHT_GRAD_CONV_TOL = getattr(__config__, 'scf_hf_kernel_tight_grad_conv_tol', True)
MUTE_CHKFILE = getattr(__config__, 'scf_hf_SCF_mute_chkfile', False)
PENALTY = getattr(__config__, 'mcscf_casci_CASCI_fix_spin_shift', 0.2)

libsf =  numpy.ctypeslib.load_library('SFNOCI_contract',__file__)
#libsf = lib.load_library('SFNOCI_contract')

##################################utils############################################
def possible_occ(ASN,nASE):
    def find_arrays(ASN,nASE):
        possible_values=[0,1,2]
        arrays=[]

        for combination in product(possible_values,repeat=ASN):
            if sum(combination)==nASE:
                arrays.append(numpy.array(combination))
        return arrays
    result_arrays=find_arrays(ASN,nASE)
    array_list=[]
    for arr in result_arrays:
        array_list.append(arr)
    concantenated_array=numpy.array(array_list, order = 'C', dtype = numpy.int32)
    return concantenated_array

def fill_array_with_sum_N(length, N):
    result = [0] * length
    assert N <= length *2
    if N <= length:
        for i in range(N):
            result[i] = 1
    else:
        num_ones = length
        num_twos = N - num_ones

        for i in range(length):
            result[i] = 1
        for i in range(num_twos):
            result[i] = 2


    return result

def group_occ(PO, group, groupA):
    g = numpy.max(group)
    P = PO[g]
    group_occ = numpy.zeros_like(P)
    for atom in groupA:
        N = sum(P[atom])
        length = len(atom)
        highspin_occ = fill_array_with_sum_N(length,N)
        for b,c in zip(highspin_occ,atom):
            group_occ[c] = b
    return group_occ

def num_to_group(groups,number):
    for i, group in enumerate(groups):
        if number in group:
            return i
    return None

def grouping_by_occ(PO, groupA):
    a = len(groupA)
    p = len(PO)
    n = len(PO[0])
    A_occ = numpy.zeros((p,a))
    for index, occ in enumerate(PO):
        for i in range(a):
            A_occ[index][i] = numpy.sum(occ[groupA[i]])
    grouped_rows = {}
    for i, row in enumerate(A_occ):
        row_tuple = tuple(row)
        if row_tuple not in grouped_rows:
            grouped_rows[row_tuple]=[]
        grouped_rows[row_tuple].append(i)
    return list(grouped_rows.values())

def find_matching_rows(matrix, target_row):
    matching_rows = numpy.where((matrix == target_row).all(axis=1))[0]
    return matching_rows

def python_list_to_c_array(python_list):
    if python_list is None: return ctypes.c_void_p(None), ctypes.c_void_p(None), 0 
    else:
        num_groups = len(python_list)
        flat_list = sum(python_list, [])
        flat_list = (ctypes.c_int *len(flat_list))(*flat_list)
        group_sizes = (ctypes.c_int * num_groups)()  
        for i, group in enumerate(python_list):
            group_size = len(group)
            group_sizes[i] = group_size  
        return flat_list, group_sizes, num_groups
    
def str2occ(str0,norb):
    occ=numpy.zeros(norb)
    for i in range(norb):
        if str0 & (1<<i):
            occ[i]=1

    return occ
####################################################################################

#######################Integral, SVD helper#########################################
def MO_overlap(MO1,MO2,s1e):
    MO_overlap=lib.einsum('ai,bj,ij->ab',numpy.conjugate(MO1.T),MO2.T,s1e)

    return MO_overlap

def biorthogonalize(MO1,MO2,s1e):
    U,S,Vt=numpy.linalg.svd(MO_overlap(MO1,MO2,s1e))
    MO1_bimo_coeff=MO1.dot(U)
    MO2_bimo_coeff=MO2.dot(Vt.T)
    return S, MO1_bimo_coeff, MO2_bimo_coeff, U, Vt
    
def J_matrix(eri,den):
    J=lib.einsum('abij,ij->ab',eri,den)
    return J

def K_matrix(eri,den):
    K=-lib.einsum('aijb,ji->ab',eri,den)
    return K
#######################################################################################


def FASSCF(mf,AS_list,core_list,highspin_mo_energy,highspin_mo_coeff,AS_occ,conv_tol=1e-10, conv_tol_grad=None, max_cycle = 100,
           dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):
    mf.max_cycle = max_cycle
    ASN=len(AS_list)
    N=highspin_mo_coeff.shape[1]
    cn=len(core_list)
    vir_list=numpy.array(range(numpy.max(AS_list)+1,N))
    highspin_mo_occ=numpy.zeros(N)
    for i in core_list:
        highspin_mo_occ[i]=2
    for idx, value in zip(AS_list,AS_occ):
        highspin_mo_occ[idx]=value
    for i in vir_list:
        highspin_mo_occ[i]=0

    if'init_dm' in kwargs:
        raise RuntimeError('''
You see this error message because of the API updates in pyscf v0.11.
Keyword argument "init_dm" is replaced by "dm0"''')
    cput0 = (logger.process_clock(), logger.perf_counter())
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        logger.info(mf, 'Set gradient conv threshold to %g', conv_tol_grad)

    mol = mf.mol
    if dm0 is None:
        dm = mf.make_rdm1(highspin_mo_coeff,highspin_mo_occ)
    else:
        dm = dm0

    h1e = mf.get_hcore(mol)
    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    logger.info(mf, 'init E= %.15g', e_tot)

    scf_conv = False
    mo_energy = mo_coeff = mo_occ = None

    s1e = mf.get_ovlp(mol)
    cond = lib.cond(s1e)
    logger.debug(mf, 'cond(S) = %s', cond)
    if numpy.max(cond)*1e-17 > conv_tol:
        logger.warn(mf, 'Singularity detected in overlap matrix (condition number = %4.3g). '
                    'SCF may be inaccurate and hard to converge.', numpy.max(cond))

    # Skip SCF iterations. Compute only the total energy of the initial density
    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, lib.diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback

        # We get the used orthonormalized AO basis from any old eigendecomposition.
        # Since the ingredients for the Fock matrix has already been built, we can
        # just go ahead and use it to determine the orthonormal basis vectors.
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        _, mf_diis.Corth = mf.eig(fock, s1e)
    else:
        mf_diis = None

    if dump_chk and mf.chkfile:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        chkfile.save_mol(mol, mf.chkfile)

    # A preprocessing hook before the SCF iteration
    mf.pre_kernel(locals())

    cput1 = logger.timer(mf, 'initialize scf', *cput0)
    
    mo_energy=highspin_mo_energy
    mo_coeff=highspin_mo_coeff
    mo_occ=highspin_mo_occ
    for cycle in range(mf.max_cycle):
        dm_last = dm
        last_hf_e = e_tot

        
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        mo_basis_fock=(mo_coeff.T.dot(fock)).dot(mo_coeff)
        I=numpy.identity(N-ASN)
        reduced_mo_basis_fock=mo_basis_fock[numpy.ix_(~numpy.isin(numpy.arange(mo_basis_fock.shape[0]),AS_list),~numpy.isin(numpy.arange(mo_basis_fock.shape[1]),AS_list))]
        mo_basis_s1e=(mo_coeff.T.dot(s1e)).dot(mo_coeff)
        
        reduced_mo_basis_s1e=numpy.zeros((N-ASN,N-ASN))
        reduced_mo_basis_s1e=mo_basis_s1e[numpy.ix_(~numpy.isin(numpy.arange(mo_basis_s1e.shape[0]),AS_list),~numpy.isin(numpy.arange(mo_basis_s1e.shape[1]),AS_list))] 
        new_mo_energy, mo_basis_new_mo_coeff=mf.eig(reduced_mo_basis_fock,I)
        reduced_mo_coeff=numpy.delete(mo_coeff,AS_list,axis=1)
        new_mo_coeff=reduced_mo_coeff.dot(mo_basis_new_mo_coeff)
        
 

        for i in AS_list:
            new_mo_coeff=numpy.insert(new_mo_coeff,i,highspin_mo_coeff[:,i],axis=1)
        mo_coeff=new_mo_coeff


        AS_fock_energy=lib.einsum('ai,aj,ij->a',numpy.conjugate(mo_coeff.T),mo_coeff.T,fock)
        for i in AS_list:
            new_mo_energy=numpy.insert(new_mo_energy,i,AS_fock_energy[i])
        mo_energy=new_mo_energy
        dm = mf.make_rdm1(mo_coeff,mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        e_tot = mf.energy_tot(dm, h1e, vhf)
        # Here Fock matrix is h1e + vhf, without DIIS.  Calling get_fock
        # instead of the statement "fock = h1e + vhf" because Fock matrix may
        # be modified in some methods.i
        fock_last=fock
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        norm_gorb = numpy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)
        norm_ddm = numpy.linalg.norm(dm-dm_last)
        logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                    cycle+1, e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol and norm_ddm < numpy.sqrt(conv_tol):
            scf_conv = True

        if dump_chk:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

        cput1 = logger.timer(mf, 'cycle= %d'%(cycle+1), *cput1)

        if scf_conv:
            break

    if scf_conv and conv_check:
        # An extra diagonalization, to remove level shift
        #fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf
        mo_basis_fock=(mo_coeff.T.dot(fock)).dot(mo_coeff)
        I=numpy.identity(N-ASN)
        reduced_mo_basis_fock=mo_basis_fock[numpy.ix_(~numpy.isin(numpy.arange(mo_basis_fock.shape[0]),AS_list),~numpy.isin(numpy.arange(mo_basis_fock.shape[1]),AS_list))]
        mo_basis_s1e=(mo_coeff.T.dot(s1e)).dot(mo_coeff)

        reduced_mo_basis_s1e=numpy.zeros((N-ASN,N-ASN))
        reduced_mo_basis_s1e=mo_basis_s1e[numpy.ix_(~numpy.isin(numpy.arange(mo_basis_s1e.shape[0]),AS_list),~numpy.isin(numpy.arange(mo_basis_s1e.shape[1]),AS_list))]

        new_mo_energy, mo_basis_new_mo_coeff=mf.eig(reduced_mo_basis_fock,reduced_mo_basis_s1e)
        reduced_mo_coeff=numpy.delete(mo_coeff,AS_list,axis=1)
        new_mo_coeff=reduced_mo_coeff.dot(mo_basis_new_mo_coeff)



        for i in AS_list:
            new_mo_coeff=numpy.insert(new_mo_coeff,i,mo_coeff[:,i],axis=1)

        mo_coeff=new_mo_coeff
            
        AS_fock_energy=lib.einsum('ai,aj,ij->a',numpy.conjugate(mo_coeff.T),mo_coeff.T,fock)
        for i in AS_list:
            new_mo_energy=numpy.insert(new_mo_energy,i,AS_fock_energy[i])
        mo_energy=new_mo_energy
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        vhf = mf.get_veff(mol, dm,dm_last,vhf)
        e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm)
        norm_gorb = numpy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)
        norm_ddm = numpy.linalg.norm(dm-dm_last)

        conv_tol = conv_tol * 10
        conv_tol_grad = conv_tol_grad * 3
        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol or norm_gorb < conv_tol_grad:
            scf_conv = True
        logger.info(mf, 'Extra cycle  E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                    e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)
        if dump_chk:
            mf.dump_chk(locals())

    if mf.disp is not None:
        e_disp = mf.get_dispersion()
        mf.scf_summary['dispersion'] = e_disp
        e_tot += e_disp

    logger.timer(mf, 'scf_cycle', *cput0)
    # A post-processing hook before return
    mf.post_kernel(locals())
    if scf_conv==False:
        mo_coeff=highspin_mo_coeff

    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

def optimized_mo(mf,mo_energy,mo_coeff,AS_list,core_list,nelecas,mode=0,conv_tol = 1e-10, max_cycle = 100, groupA = None):
    ASN=len(AS_list)
    nASE = nelecas[0] + nelecas[1]
    N=mo_coeff.shape[0]
    PO=possible_occ(ASN,nASE)
    p=len(PO)
    group = None
    if groupA is None:
       optimized_mo=numpy.zeros((p,N,N))
       #SF-NOCI
       if mode == 0:
          for i, occ in enumerate(PO):
              conv, et, moe, moce, moocc = FASSCF(mf,AS_list,core_list,mo_energy,mo_coeff,occ,conv_tol = conv_tol,max_cycle = max_cycle )
              print(conv, et)
              optimized_mo[i]=moce
              print("occuped pattern index:")
              print(i)
       #SF-CAS
       if mode==1:
          for i, occ in enumerate(PO):
              optimized_mo[i]=mo_coeff
    else :
        group = grouping_by_occ(PO,groupA)
        g = len(group)
        optimized_mo = numpy.zeros((g,N,N))
        for i in range(0,g):
            if mode == 0:
                occ = group_occ(PO,group[i],groupA)
                conv, et, moe, moce, moocc = FASSCF(mf,AS_list,core_list,mo_energy,mo_coeff,occ,conv_tol = conv_tol,max_cycle = max_cycle )
                print(conv, et)
                optimized_mo[i]=moce
            if mode == 1:
                optimized_mo[i] = mo_coeff
            print("occuped pattern index:")
            print(i)
    return optimized_mo, PO, group 
  


def absorb_h1eff(h1eff,eri,ncas,nelecas,fac=1):
    '''Modify 2e Hamiltonian to include effective 1e Hamiltonian contribution

    input : h1eff : (ngroup, ngroup, ncas, ncas)
            eri   : (ncas, ncas, ncas, ncas)

    return : erieff : (ngroup,ngroup,ncas,ncas,ncas,ncas)
    '''
    if not isinstance(nelecas, (int, numpy.number)):
        nelecas = sum(nelecas)
    h2e = ao2mo.restore(1, eri.copy(), ncas)
    p = h1eff.shape[0]
    f1e = h1eff
    f1e -= numpy.einsum('jiik->jk', h2e)[numpy.newaxis,numpy.newaxis,:,:]*0.5
    f1e = f1e * (1./(nelecas+1e-100))
    erieff = numpy.zeros((p,p,ncas,ncas,ncas,ncas))
    erieff += h2e[numpy.newaxis,numpy.newaxis,:,:,:,:]
    for k in range(ncas):
        erieff[:,:,k,k,:,:] += f1e
        erieff[:,:,:,:,k,k] += f1e
    return erieff * fac

def contract_H_slow(erieff, civec, ncas, nelecas, PO, group, TSc, energy_core ,link_index = None):
    '''Compute H|CI>
    '''
    if isinstance(nelecas, (int, numpy.integer)):
        nelecb = nelecas//2
        neleca = nelecas - nelecb
    else:
        neleca, nelecb = nelecas
    if link_index is None:
        link_indexa = cistring.gen_linkstr_index(range(ncas), neleca)
        link_indexb = cistring.gen_linkstr_index(range(ncas), nelecb)
    else:
        link_indexa, link_indexb = link_index
    na = cistring.num_strings(ncas,neleca)
    nb = cistring.num_strings(ncas,nelecb)
    civec = civec.reshape(na,nb)
    cinew = numpy.zeros((na,nb))
    stringsa = cistring.make_strings(range(ncas),neleca)
    stringsb = cistring.make_strings(range(ncas),nelecb)
    t2aa = numpy.zeros((ncas,ncas,ncas,ncas,na,na))
    t2bb = numpy.zeros((ncas,ncas,ncas,ncas,nb,nb))
    t1a = numpy.zeros((ncas,ncas,na,na))
    t1b = numpy.zeros((ncas,ncas,nb,nb)) 
    for str0a , taba in enumerate(link_indexa):
          for a1, i1, str1a, signa1 in link_indexa[str0a]:
              t1a[a1,i1,str1a,str0a] += signa1 
              for a2 , i2, str2a, signa2 in link_indexa[str1a]:
                  t2aa[a2, i2, a1, i1, str2a, str0a] += signa1 * signa2
    for str0b , tabb in enumerate(link_indexb):
          for a1, i1, str1b, signb1 in link_indexb[str0b]:
              t1b[a1,i1,str1b,str0b] += signb1
              for a2 , i2, str2b, signb2 in link_indexb[str1b]:
                  t2bb[a2, i2, a1, i1, str2b, str0b] += signb1 * signb2
    t1a_nonzero = numpy.array(numpy.nonzero(t1a)).T
    t1b_nonzero = numpy.array(numpy.nonzero(t1b)).T
    t2aa_nonzero = numpy.array(numpy.nonzero(t2aa)).T
    t2bb_nonzero = numpy.array(numpy.nonzero(t2bb)).T
    for aa, ia, str1a, str0a in t1a_nonzero:
        for ab, ib, str1b, str0b in t1b_nonzero:
            w_occa = str2occ(stringsa[str0a],ncas)
            w_occb = str2occ(stringsb[str0b],ncas)
            x_occa = str2occ(stringsa[str1a],ncas)
            x_occb = str2occ(stringsb[str1b],ncas)
            x_occ = x_occa + x_occb
            w_occ = w_occa + w_occb
            p1 = find_matching_rows(PO,x_occ)[0]
            p2 = find_matching_rows(PO,w_occ)[0]
            if group is not None:
                p1 = num_to_group(group,p1)
                p2 = num_to_group(group,p2)
           
            cinew[str1a,str1b] += civec[str0a,str0b] * erieff[p1,p2,aa,ia,ab,ib] *t1a[aa,ia,str1a,str0a]* t1b[ab,ib,str1b,str0b] * TSc[p1,p2] *2
    for a1, i1, a2,i2, str1a, str0a in t2aa_nonzero:
        for str0b, stringb in enumerate(stringsb):
            w_occa = str2occ(stringsa[str0a],ncas)
            w_occb = str2occ(stringsb[str0b],ncas)
            x_occa = str2occ(stringsa[str1a],ncas)
            x_occ = x_occa + w_occb
            w_occ = w_occa + w_occb
            p1 = find_matching_rows(PO,x_occ)[0]
            p2 = find_matching_rows(PO,w_occ)[0]
            if group is not None:
                p1 = num_to_group(group,p1)
                p2 = num_to_group(group,p2)
            cinew[str1a,str0b] += civec[str0a,str0b] * erieff[p1,p2,a1,i1,a2,i2] *t2aa[a1,i1,a2,i2,str1a,str0a] * TSc[p1,p2]
    for a1, i1, a2,i2, str1b, str0b in t2bb_nonzero:
        for str0a, stringa in enumerate(stringsa):
            w_occa = str2occ(stringsa[str0a],ncas)
            w_occb = str2occ(stringsb[str0b],ncas)
            x_occb = str2occ(stringsb[str1b],ncas)
            x_occ = w_occa + x_occb
            w_occ = w_occa + w_occb
            p1 = find_matching_rows(PO,x_occ)[0]
            p2 = find_matching_rows(PO,w_occ)[0]
            if group is not None:
                p1 = num_to_group(group,p1)
                p2 = num_to_group(group,p2)
            cinew[str0a,str1b] += civec[str0a,str0b] * erieff[p1,p2,a1,i1,a2,i2]* t2bb[a1,i1,a2,i2,str1b,str0b] * TSc[p1,p2]
    for str0a, stringa in enumerate(stringsa):
        for str0b, stringb in enumerate(stringsb):
            w_occa = str2occ(stringsa[str0a],ncas)
            w_occb = str2occ(stringsb[str0b],ncas)
            w_occ = w_occa + w_occb
            p = find_matching_rows(PO,w_occ)[0]
            if group is not None:
                p = num_to_group(group,p)
            cinew[str0a,str0b] += energy_core[p] * civec[str0a,str0b]
    cinew.reshape(-1)
    return cinew

def contract_H(erieff, civec, ncas, nelecas, PO, group, TSc, energy_core ,link_index = None):
    '''Compute H|CI>
    
    '''
    if isinstance(nelecas, (int, numpy.integer)):
        nelecb = nelecas//2
        neleca = nelecas - nelecb
    else:
        neleca, nelecb = nelecas
    if link_index is None:
        link_indexa = cistring.gen_linkstr_index(range(ncas), neleca)
        link_indexb = cistring.gen_linkstr_index(range(ncas), nelecb)
    else:
        link_indexa, link_indexb = link_index
    na = cistring.num_strings(ncas,neleca)
    nb = cistring.num_strings(ncas,nelecb)
    civec = numpy.asarray(civec, order = 'C')
    cinew = numpy.zeros_like(civec)
    erieff = numpy.asarray(erieff, order = 'C', dtype= numpy.float64)
    PO = numpy.asarray(PO, order = 'C', dtype=numpy.int32)
    PO_nrows = PO.shape[0]
    cgroup, group_sizes, num_groups = python_list_to_c_array(group)
    stringsa = cistring.make_strings(range(ncas),neleca)
    stringsb = cistring.make_strings(range(ncas),nelecb)
    t2aa = numpy.zeros((ncas,ncas,ncas,ncas,na,na), dtype=numpy.int32)
    t2bb = numpy.zeros((ncas,ncas,ncas,ncas,nb,nb),dtype=numpy.int32)
    t1a = numpy.zeros((ncas,ncas,na,na),dtype=numpy.int32)
    t1b = numpy.zeros((ncas,ncas,nb,nb),dtype=numpy.int32) 
    for str0a , taba in enumerate(link_indexa):
          for a1, i1, str1a, signa1 in link_indexa[str0a]:
              t1a[a1,i1,str1a,str0a] += signa1 
              for a2 , i2, str2a, signa2 in link_indexa[str1a]:
                  t2aa[a2, i2, a1, i1, str2a, str0a] += signa1 * signa2
    for str0b , tabb in enumerate(link_indexb):
          for a1, i1, str1b, signb1 in link_indexb[str0b]:
              t1b[a1,i1,str1b,str0b] += signb1
              for a2 , i2, str2b, signb2 in link_indexb[str1b]:
                  t2bb[a2, i2, a1, i1, str2b, str0b] += signb1 * signb2
    t1a_nonzero = numpy.array(numpy.array(numpy.nonzero(t1a)).T, order = 'C', dtype = numpy.int32)
    t1b_nonzero = numpy.array(numpy.array(numpy.nonzero(t1b)).T, order = 'C', dtype = numpy.int32)
    t2aa_nonzero = numpy.array(numpy.array(numpy.nonzero(t2aa)).T, order = 'C', dtype = numpy.int32)
    t2bb_nonzero = numpy.array(numpy.array(numpy.nonzero(t2bb)).T, order = 'C', dtype = numpy.int32)
    t1ann = t1a_nonzero.shape[0]
    t1bnn = t1b_nonzero.shape[0]
    t2aann = t2aa_nonzero.shape[0]
    t2bbnn = t2bb_nonzero.shape[0]
    TSc = numpy.asarray(TSc, order = 'C', dtype=numpy.float64)
    energy_core = numpy.asarray(energy_core, order = 'C', dtype=numpy.float64)
    libsf.SFNOCIcontract_H_spin1(erieff.ctypes.data_as(ctypes.c_void_p),
                                  civec.ctypes.data_as(ctypes.c_void_p),
                                 cinew.ctypes.data_as(ctypes.c_void_p),
                                  ctypes.c_int(ncas),
                                  ctypes.c_int(neleca), ctypes.c_int(nelecb),
                                  PO.ctypes.data_as(ctypes.c_void_p),ctypes.c_int(PO_nrows),
                                  ctypes.c_int(na), stringsa.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nb), stringsb.ctypes.data_as(ctypes.c_void_p),
                                  cgroup, group_sizes, ctypes.c_int(num_groups),
                                  t1a.ctypes.data_as(ctypes.c_void_p), t1a_nonzero.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(t1ann),
                                  t1b.ctypes.data_as(ctypes.c_void_p), t1b_nonzero.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(t1bnn),
                                  t2aa.ctypes.data_as(ctypes.c_void_p), t2aa_nonzero.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(t2aann),
                                  t2bb.ctypes.data_as(ctypes.c_void_p), t2bb_nonzero.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(t2bbnn),
                                  TSc.ctypes.data_as(ctypes.c_void_p), energy_core.ctypes.data_as(ctypes.c_void_p))

    return cinew
            
def make_hdiag(h1eff, eri, ncas, nelecas, PO, group, energy_core, opt = None):
    if isinstance(nelecas, (int, numpy.integer)):
        nelecb = nelecas//2
        neleca = nelecas - nelecb
    else:
        neleca, nelecb = nelecas
    occslista = cistring.gen_occslst(range(ncas), neleca)
    occslistb = cistring.gen_occslst(range(ncas), nelecb)
    eri = ao2mo.restore(1, eri, ncas)
    diagj = numpy.einsum('iijj->ij', eri)
    diagk = numpy.einsum('ijji->ij', eri)
    hdiag = []
    for aocc in occslista:
        for bocc in occslistb:
            occ = numpy.zeros(ncas)
            for i in aocc:
                occ[i] += 1
            for i in bocc:
                occ[i] +=1
            p = find_matching_rows(PO,occ)
            if group is not None:
                p = num_to_group(group, p)
            e1 = h1eff[p,p,aocc,aocc].sum() + h1eff[p,p,bocc,bocc].sum()
            e2 = diagj[aocc][:,aocc].sum() + diagj[aocc][:,bocc].sum() \
               + diagj[bocc][:,aocc].sum() + diagj[bocc][:,bocc].sum() \
               - diagk[aocc][:,aocc].sum() - diagk[bocc][:,bocc].sum()
            hdiag.append(e1 + e2*.5 + energy_core[p])
    return numpy.array(hdiag)

def _construct_block_hamiltonian(mol,nelec,ASN,PO,h1c,eri,TSc,Kc,group):
    stringsa = cistring.make_strings(range(ASN),nelec[0])
    stringsb = cistring.make_strings(range(ASN),nelec[1])
    link_indexa = cistring.gen_linkstr_index(range(ASN),nelec[0])
    link_indexb = cistring.gen_linkstr_index(range(ASN),nelec[1])
    na = cistring.num_strings(ASN,nelec[0])
    nb = cistring.num_strings(ASN,nelec[1])
    idx_a = numpy.arange(na)
    idx_b = numpy.arange(nb)
    mat1 = numpy.zeros((na,nb,na,nb))
    matTSc = numpy.zeros((na,nb,na,nb))
    for str0a, strs0a in enumerate(stringsa):
        for str1a, strsa in enumerate(stringsa):
            for str0b, strs0b in enumerate(stringsb):
                for str1b, strs1b in enumerate(stringsb):
                    w_occa = str2occ(stringsa[str0a],ASN)
                    w_occb = str2occ(stringsb[str0b],ASN)
                    x_occa = str2occ(stringsa[str1a],ASN)
                    x_occb = str2occ(stringsb[str1b],ASN)
                    x_occ = numpy.array(x_occa) + numpy.array(x_occb)
                    w_occ = numpy.array(w_occa) + numpy.array(w_occb)
                    p1=find_matching_rows(PO,x_occ)[0]
                    p2=find_matching_rows(PO,w_occ)[0]
                    if group is not None: 
                        p1 = num_to_group(group,p1)
                        p2 = num_to_group(group,p2)
                    matTSc[str1a,str1b,str0a,str0b] += TSc[p1,p2]
    for str0a, taba in enumerate(link_indexa):
        for pa, qa, str1a, signa in taba:
            for str0b, strsb in enumerate(stringsb):
                 w_occa = str2occ(stringsa[str0a],ASN)
                 w_occb = str2occ(stringsb[str0b],ASN)
                 x_occa = str2occ(stringsa[str1a],ASN)
                 x_occ = numpy.array(x_occa) + numpy.array(w_occb)
                 w_occ = numpy.array(w_occa) + numpy.array(w_occb)
                 p1=find_matching_rows(PO,x_occ)[0]
                 p2=find_matching_rows(PO,w_occ)[0]
                 if group is not None: 
                    p1 = num_to_group(group,p1)
                    p2 = num_to_group(group,p2)
                 if matTSc[str1a,str0b,str0a,str0b]==0:
                    matTSc[str1a,str0b,str0a,str0b] += TSc[p1,p2]
                 mat1[str1a,str0b,str0a,str0b] += signa * h1c[p1,p2,pa,qa]
    for str0b, tabb in enumerate(link_indexb):
        for pb, qb, str1b, signb in tabb:
            for str0a, strsa in enumerate(stringsa):
                w_occa = str2occ(stringsa[str0a],ASN)
                w_occb = str2occ(stringsb[str0b],ASN)
                x_occb = str2occ(stringsb[str1b],ASN)
                x_occ = numpy.array(w_occa) + numpy.array(x_occb)
                w_occ = numpy.array(w_occa) + numpy.array(w_occb)
                p1=find_matching_rows(PO,x_occ)[0]
                p2=find_matching_rows(PO,w_occ)[0]
                if group is not None: 
                   p1 = num_to_group(group,p1)
                   p2 = num_to_group(group,p2)
                if matTSc[str0a,str1b,str0a,str0b]==0:
                   matTSc[str0a,str1b,str0a,str0b] += TSc[p1,p2]
                mat1[str0a,str1b,str0a,str0b] += signb * h1c[p1,p2,pb,qb]
    #mat1 = mat1.reshape(na*nb,na*nb)
    h2 = fci_slow.absorb_h1e(h1c[0,0]*0, eri, ASN, nelec)
    t1 = numpy.zeros((ASN,ASN,na,nb,na,nb))
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            # alpha spin
            t1[a,i,str1,idx_b,str0,idx_b] += sign
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            # beta spin
            t1[a,i,idx_a,str1,idx_a,str0] += sign
    t1 = lib.einsum('psqr,qrABab->psABab', h2, t1)
    mat2 = numpy.zeros((na,nb,na,nb))
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            # alpha spin
            mat2[str1] += sign * t1[a,i,str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
           # beta spin
            mat2[:,str1] += sign * t1[a,i,:,str0]
    #mat2 = mat2.reshape(na*nb,na*nb)
    ham = (mat1+0.5*mat2)*matTSc
    ham = ham.reshape(na*nb,na*nb)
    K = numpy.zeros((na,nb))
    for i in range(0,na):
        for j in range(0,nb):
            x_occa = str2occ(stringsa[i],ASN)
            x_occb = str2occ(stringsb[j],ASN)        
            x_state_occ = numpy.array(x_occa) + numpy.array(x_occb)
            p1=find_matching_rows(PO,x_state_occ)[0]
            if group is not None: 
                p1 = num_to_group(group,p1) 
            K[i,j] = Kc[p1]
    K = K.reshape(-1)
    K = numpy.diag(K)
    print("mat1")
    print(mat1.reshape(na*nb,na*nb)) 
    print("mat2")
    print(0.5*mat2.reshape(na*nb,na*nb))
    print("K")
    print(K)
    
    hamiltonian = ham + K 
    print(hamiltonian)
    return hamiltonian

def construct_block_hamiltonian(mol,nelec,ASN,PO,h1c,eri,TSc,Kc,group):
    stringsa = cistring.make_strings(range(ASN),nelec[0])
    stringsb = cistring.make_strings(range(ASN),nelec[1])
    link_indexa = cistring.gen_linkstr_index(range(ASN),nelec[0])
    link_indexb = cistring.gen_linkstr_index(range(ASN),nelec[1])
    na = cistring.num_strings(ASN,nelec[0])
    nb = cistring.num_strings(ASN,nelec[1])
    idx_a = numpy.arange(na)
    idx_b = numpy.arange(nb)
    mat1 = numpy.zeros((na,nb,na,nb))
    matTSc = numpy.zeros((na,nb,na,nb))
    for str0a, taba in enumerate(link_indexa):
        for pa, qa, str1a, signa in taba:
            for str0b, tabb in enumerate(link_indexb):
                for pb, qb, str1b, signb in tabb:
                    w_occa = str2occ(stringsa[str0a],ASN)
                    w_occb = str2occ(stringsb[str0b],ASN)
                    x_occa = str2occ(stringsa[str1a],ASN)
                    x_occb = str2occ(stringsb[str1b],ASN)
                    x_state_occ = numpy.array(x_occa) + numpy.array(x_occb)
                    w_state_occ = numpy.array(w_occa) + numpy.array(w_occb)
                    p1=find_matching_rows(PO,x_state_occ)[0]
                    p2=find_matching_rows(PO,w_state_occ)[0]
                    if group is not None: 
                       p1 = num_to_group(group,p1)
                       p2 = num_to_group(group,p2) 
                    if matTSc[str1a,str1b,str0a,str0b]==0:
                        matTSc[str1a,str1b,str0a,str0b] += TSc[p1,p2]    
                    if pa==qa and pb ==qb:
                        mat1[str1a,str1b,str0a,str0b] += (signa * h1c[p1,p2,pa,qa]/nelec[1]  + signb * h1c[p1,p2,pb,qb]/nelec[0])
                    elif pa!=qa and pb == qb:
                        mat1[str1a,str1b,str0a,str0b] += signa * h1c[p1,p2,pa,qa]/nelec[1]
                    elif pa==qa and pb !=qb:
                        mat1[str1a,str1b,str0a,str0b] += signb * h1c[p1,p2,pb,qb]/nelec[0]
                    elif pa!=qa and pb !=qb:
                        mat1[str1a,str1b,str0a,str0b] += 0
                    #mat1[str1a,idx_b,str0a,idx_b] += signa * h1c[g1,g2,pa,qa]
                    #mat1[idx_a,str1b,idx_a,str0b] += signb * h1c[g1,g2,pb,qb]

    #mat1 = mat1.reshape(na*nb,na*nb)
    h2 = fci_slow.absorb_h1e(h1c[0,0]*0, eri, ASN, nelec)
    t1 = numpy.zeros((ASN,ASN,na,nb,na,nb))
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            # alpha spin
            t1[a,i,str1,idx_b,str0,idx_b] += sign
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            # beta spin
            t1[a,i,idx_a,str1,idx_a,str0] += sign
    t1 = lib.einsum('psqr,qrABab->psABab', h2, t1)
    mat2 = numpy.zeros((na,nb,na,nb))
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            # alpha spin
            mat2[str1] += sign * t1[a,i,str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
           # beta spin
            mat2[:,str1] += sign * t1[a,i,:,str0]
    #mat2 = mat2.reshape(na*nb,na*nb)
    ham = (mat1+0.5*mat2)*matTSc
    ham = ham.reshape(na*nb,na*nb)
    K = numpy.zeros((na,nb))
    for i in range(0,na):
        for j in range(0,nb):
            x_occa = str2occ(stringsa[i],ASN)
            x_occb = str2occ(stringsb[j],ASN)        
            x_state_occ = numpy.array(x_occa) + numpy.array(x_occb)
            p1=find_matching_rows(PO,x_state_occ)[0]
            if group is not None: 
                p1 = num_to_group(group,p1) 
            K[i,j] = Kc[p1]
    K = K.reshape(-1)
    K = numpy.diag(K)
    print("mat1")
    print(mat1.reshape(na*nb,na*nb)) 
    print("mat2")
    print(0.5*mat2.reshape(na*nb,na*nb))
    print("K")
    print(K)
    
    hamiltonian = ham + K 
    print(hamiltonian)
    
    return hamiltonian

def fix_spin(fciobj, shift=PENALTY, ss=None, **kwargs):
    r'''If FCI solver cannot stay on spin eigenfunction, this function can
    add a shift to the states which have wrong spin.

    .. math::

        (H + shift*S^2) |\Psi\rangle = E |\Psi\rangle

    Args:
        fciobj : An instance of :class:`FCISolver`

    Kwargs:
        shift : float
            Level shift for states which have different spin
        ss : number
            S^2 expection value == s*(s+1)

    Returns
            A modified FCI object based on fciobj.
    '''
    import types
    from pyscf.fci import direct_uhf
    if isinstance(fciobj, direct_uhf.FCISolver):
        raise NotImplementedError

    if isinstance (fciobj, types.ModuleType):
        raise DeprecationWarning('fix_spin should be applied on FCI object only')

    if 'ss_value' in kwargs:
        sys.stderr.write('fix_spin_: kwarg "ss_value" will be removed in future release. '
                         'It was replaced by "ss"\n')
        ss_value = kwargs['ss_value']
    else:
        ss_value = ss

    if isinstance (fciobj, SpinPenaltySFNOCISolver):
        # recursion avoidance
        fciobj.ss_penalty = shift
        fciobj.ss_value = ss_value
        return fciobj

    return lib.set_class(SpinPenaltySFNOCISolver(fciobj, shift, ss_value),
                         (SpinPenaltySFNOCISolver, fciobj.__class__))

def fix_spin_(fciobj, shift=.1, ss=None):
    sp_fci = fix_spin(fciobj, shift, ss)
    fciobj.__class__ = sp_fci.__class__
    fciobj.__dict__ = sp_fci.__dict__
    return fciobj

           
def h1e_for_SFNOCI(SFNOCI, Adm = None, MO = None, W = None, ncas = None, ncore = None):
    if W is None : SFNOCI.W
    if ncas is None : ncas = SFNOCI.ncas
    if ncore is None : ncore = SFNOCI.ncore
    if MO is None : MO = SFNOCI.MO           
    if Adm is None : Adm = SFNOCI.get_active_dm()           
    p = W.shape[0]           
    mo_cas = MO[0][:,ncore:ncore+ncas]
    hcore = SFNOCI.get_hcore()
    h1eff = numpy.zeros((p,p,ncas,ncas))       
    energy_core = numpy.zeros(p)
    energy_nuc = SFNOCI.energy_nuc()   
    eri = SFNOCI.mol.intor('int2e')
    ha1e = lib.einsum('ai,ab,bj->ij',mo_cas,hcore,mo_cas)      
    for i in range(0,p):
        for j in range(0,p):      
            h1eff[i,j] = ha1e + lib.einsum('ijab,ab->ij', Adm, 2*J_matrix(eri,W[i,j])+K_matrix(eri,W[i,j]))
            if i==j:
                energy_core[i] += (lib.einsum('ab,ab->',W[i,i],2*J_matrix(eri,W[i,i])+K_matrix(eri,W[i,i])))
                energy_core[i] += energy_nuc
                energy_core[i] += 2*lib.einsum('ab,ab->',hcore, W[i,i])   
    SFNOCI.h1eff = h1eff
    SFNOCI.core_energies = energy_core
    return h1eff, energy_core
                         
def spin_square(SFNOCI, rdm1, rdm2ab,rdm2ba):
    M_s = SFNOCI.spin/2
    mo = SFNOCI.mo_coeff
    s1e = SFNOCI.mol.intor('int1e_ovlp')
    rdm1mo = lib.einsum('qi,pl,kj,qp,lk->ij', mo, rdm1, mo,s1e,s1e)
    rdm2mo = lib.einsum('ai,bj,ck,dl,ap,bq,cr,ds,pqrs',mo,mo,mo,mo,s1e,s1e,s1e,s1e,rdm2ab+rdm2ba)
    
    return M_s**2 + 0.5*lib.einsum('ii ->',rdm1mo) - 0.5*lib.einsum('ijji ->', rdm2mo)       


def make_diag_precond(hdiag, level_shift=0):
    return lib.make_diag_precond(hdiag, level_shift)


###############################################################
# direct-CI driver
###############################################################
def kernel_SFNOCI(mySFNOCI, h1eff, eri, ncas, nelecas, PO, group, TSc, energy_core, ci0=None, link_index=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None, hop=None,
               max_memory=None, verbose=None, ecore=0, **kwargs):
    '''
    Args:
        h1e: ndarray
            1-electron Hamiltonian
        eri: ndarray
            2-electron integrals in chemist's notation
        norb: int
            Number of orbitals
        nelec: int or (int, int)
            Number of electrons of the system

    Kwargs:
        ci0: ndarray
            Initial guess
        link_index: ndarray
            A lookup table to cache the addresses of CI determinants in
            wave-function vector
        tol: float
            Convergence tolerance
        lindep: float
            Linear dependence threshold
        max_cycle: int
            Max. iterations for diagonalization
        max_space: int
            Max. trial vectors to store for sub-space diagonalization method
        nroots: int
            Number of states to solve
        davidson_only: bool
            Whether to call subspace diagonalization (davidson solver) or do a
            full diagonalization (lapack eigh) for small systems
        pspace_size: int
            Number of determinants as the threshold of "small systems",
        hop: function(c) => array_like_c
            Function to use for the Hamiltonian multiplication with trial vector

    Note: davidson solver requires more arguments. For the parameters not
    dispatched, they can be passed to davidson solver via the extra keyword
    arguments **kwargs
    '''
    if nroots is None: nroots = mySFNOCI.fcisolver.nroots
    if davidson_only is None: davidson_only = mySFNOCI.fcisolver.davidson_only
    if pspace_size is None: pspace_size = mySFNOCI.fcisolver.pspace_size
    if max_memory is None:
        max_memory = mySFNOCI.max_memory - lib.current_memory()[0]
    log = logger.new_logger(mySFNOCI, verbose)
    nelec = nelecas
    assert (0 <= nelec[0] <= ncas and 0 <= nelec[1] <= ncas)
    hdiag = mySFNOCI.make_hdiag(h1eff , eri , ncas , nelec , PO , group , energy_core , opt = None).ravel()
    num_dets = hdiag.size
    civec_size = num_dets
    precond = mySFNOCI.make_precond(hdiag)
    addr = [0]
    erieff = mySFNOCI.absorb_h1eff(h1eff,eri,ncas,nelec,.5)
    if hop is None:
        cpu0 = [logger.process_clock(), logger.perf_counter()]
        def hop(c):
            hc = mySFNOCI.contract_H(erieff, c, ncas, nelecas, PO, group, TSc, energy_core ,link_index)
            cpu0[:] = log.timer_debug1('contract_H', *cpu0)
            return hc.ravel()
    def init_guess():
        if callable(getattr(mySFNOCI, 'get_init_guess', None)):
            return mySFNOCI.get_init_guess(ncas,nelecas,nroots,hdiag)
        else:
            x0 = []
            for i in range(min(len(addr), nroots)):
                x = numpy.zeros(civec_size)
                x[addr[i]] = 1
                x0.append(x)
            return x0
    if ci0 is None:
        ci0 = init_guess
    if tol is None: tol = mySFNOCI.conv_tol
    if lindep is None: lindep = mySFNOCI.fcisolver.lindep
    if max_cycle is None: max_cycle = mySFNOCI.max_cycle
    if max_space is None: max_space = mySFNOCI.fcisolver.max_space
    with lib.with_omp_threads(None):
        e, c = mySFNOCI.eig(hop, ci0, precond, tol=tol, lindep=lindep,
                       max_cycle=max_cycle, max_space=max_space, nroots=nroots,
                       max_memory=max_memory, verbose=log, follow_state=True,
                       tol_residual=None, **kwargs)
    return e+ecore, c

class SFNOCI(CASBase):
  '''SF-NOCI
  W : density matrix of core orbitals between different bath in atomic basis : (ngroup, ngroup, N, N)
  PO : Possible occupation pattern. 
       for example, for (2e, 2o): PO = [[0,2], [1,1], [2,0]]. It is 2D numpy array.
  h1eff : effective one electron hamiltonian : (ngroup, ngroup, ncas, ncas)
  TSc : overlap between different bath : (ngroup, ngroup)
  Adm : density matrix between specific two active orbitals in atomic basis : (ncas, ncas, N, N)
  core_energies : 1D numpy array of core energies for each bath : (ngroup)
  '''
  conv_tol = 1e-10
  max_cycle = 100
  def __init__(self, mf, ncas=0, nelecas=0, ncore = None, spin = None, mo_coeff = None, mo_occ = None, groupA = None, mode = 0):  
      
      CASBase.__init__(self,mf, ncas, nelecas, ncore)
      if mo_coeff is None: mo_coeff = mf.mo_coeff
      if mo_occ is None: mo_occ   = mf.mo_occ     
      self._spin = spin
      self._groupA = groupA
      if isinstance(nelecas, (int, numpy.integer)):
         nelecb = (nelecas-self.spin)//2
         neleca = nelecas - nelecb
         self.nelecas = (neleca, nelecb)
      else:
         self.nelecas = (nelecas[0],nelecas[1])
################################################## don't modify the following attributes, they are not input options
      self.mo_coeff = mo_coeff
      self.mo_occ = mo_occ
      self.e_tot = 0       
      self.ha1e = None
      self.h1c = None
      self.PA = None
      self.W = None
      self.MO = None
      self.PO = None
      self.TSc = None
      self.h1eff = None
      self.core_energies = None
      self.energies = None
      self.mo_eri = None
      self.group = None

  @property
  def spin(self):
      if self._spin is None:
         return self.mol.spin
      else:
         return self._spin
      
  @spin.setter
  def spin(self,x):
      assert x is None or isinstance(x, (int, numpy.integer)) 
      self._spin = x
      nelecas = self.nelecas
      necas = nelecas[0] + nelecas[1]
      nelecb = (necas- x)//2
      neleca = necas - nelecb
      self.nelecas = (neleca,nelecb)
  @property
  def groupA(self):
      return self._groupA
  
  @groupA.setter
  def groupA(self,x):
      self._groupA = x
     
                            
  def possible_occ(self):
      self.PO = possible_occ(self.ncas,sum(self.nelecas))
      return self.PO
                 
  def FASSCF(self, occ, mo_coeff = None, ncas = None, ncore = None,conv_tol=1e-10, conv_tol_grad=None, max_cycle = 100):
      if mo_coeff is None : mo_coeff = self.mo_coeff
      if ncas is None : ncas = self.ncas
      if ncore is None : ncore = self.ncore
      mf = self._scf
      AS_list = numpy.array(range(ncore,ncore + ncas))  
      core_list = numpy.array(range(0,ncore))
      FAS_scf_conv, FAS_e_tot, FAS_mo_energy, FAS_mo_coeff, FAS_mo_occ = FASSCF(mf,AS_list,core_list,mf.mo_energy,mo_coeff,occ, conv_tol= conv_tol , max_cycle= max_cycle)
      return FAS_scf_conv, FAS_e_tot, FAS_mo_energy, FAS_mo_coeff, FAS_mo_occ 

  def optimize_mo(self,mo_coeff =None, debug = False, groupA = None):
      if mo_coeff is None : mo_coeff = self.mo_coeff
      if groupA is None : groupA = self.groupA
      mode = 0
      if debug : mode = 1 
      mf = self._scf
      mo_energy = mf.mo_energy
      ncore = self.ncore
      ncas = self.ncas       
      AS_list = numpy.array(range(ncore,ncore + ncas))       
      core_list = numpy.array(range(0,ncore))
      MO,PO,group= optimized_mo(self._scf,mo_energy,mo_coeff,AS_list,core_list,self.nelecas, mode, conv_tol = self.conv_tol, max_cycle = self.max_cycle , groupA = groupA)
      self.MO = MO
      self.PO = PO
      self.group = group
      self.AS_mo_coeff = MO[0][:,AS_list]
      return self.MO, self.PO, self.group
                 
  def get_SVD_matrices(self, MO = None, PO_or_group = None):
      if MO is None : MO = self.MO 
      if PO_or_group is None : 
         if self.group is None: PO_or_group = self.PO
         else : PO_or_group = self.group
      ncore = self.ncore
      ncas = self.ncas  
      s1e = self._scf.get_ovlp(self.mol)
      AS_list = numpy.array(range(ncore,ncore+ncas))
      core_list = numpy.array(range(0,ncore))       
      N = MO.shape[1]
      p = len(PO_or_group)
      W = numpy.zeros((p,p,N,N))
      TSc = numpy.zeros((p,p))       
      for i in range(0,p):
          xc_mo_coeff = MO[i][:,core_list]       
          for j in range(0,p):
              wc_mo_coeff = MO[j][:,core_list]
              S, xc_bimo_coeff, wc_bimo_coeff, U, Vt = biorthogonalize(xc_mo_coeff, wc_mo_coeff, s1e)       
              TSc[i,j] = numpy.prod(S[numpy.abs(S)>1e-10])*numpy.linalg.det(U)*numpy.linalg.det(Vt)
              for c in range(0,ncore):
                  W[i,j] +=numpy.outer(xc_bimo_coeff[:,c],wc_bimo_coeff[:,c])/S[c]
      self.W = W
      self.TSc = TSc
      return self.W, self.TSc       

  def get_active_dm(self,mo_coeff = None):
      ncas = self.ncas
      ncore = self.ncore
      nocc = ncore + ncas
      if mo_coeff is None:
          ncore = self.ncore
          mo_coeff = self.mo_coeff[:,ncore:nocc]
      elif mo_coeff.shape[1] != ncas:
           mo_coeff = mo_coeff[:,ncore:nocc] 
      N = mo_coeff.shape[0]
      Adm = numpy.zeros((ncas,ncas,N,N))
      for i in range(0,ncas):
          for j in range(0,ncas):
              Adm[i,j] = numpy.outer(mo_coeff[:,i],mo_coeff[:,j])
      self.Adm = Adm
      return Adm
           
  def get_h1cas(self, Adm = None, MO = None, W = None, ncas = None, ncore = None):
      return self.get_h1eff(Adm, MO, W, ncas, ncore)
  get_h1eff = h1e_for_SFNOCI = h1e_for_SFNOCI
             
  def get_h2eff(self, mo_coeff=None):
      '''Compute the active space two-particle Hamiltonian.
      '''
      return CASCI.get_h2eff(self,mo_coeff)

  def absorb_h1eff(self,h1eff,eri,ncas,nelecas,fac=1):
      return absorb_h1eff(h1eff,eri,ncas,nelecas,fac)

  def get_init_guess(self, ncas, nelecas, nroots, hdiag):
      return fci.direct_spin1.get_init_guess(ncas,nelecas,nroots,hdiag)

  def construct_reduced_hamiltonian(self, mo_coeff = None, h1eff= None, energy_core = None, PO = None, TSc = None):
      if mo_coeff is None : mo_coeff = self.mo_coeff
      if h1eff is None and energy_core is None: h1eff, energy_core = self.get_h1eff()
      if PO is None : PO = self.PO           
      if TSc is None : TSc = self.TSc          
      nelecas = self.nelecas   
      ncas = self.ncas  
      mol = self.mol
      group = self.group
      eri = self.get_h2eff(mo_coeff) 
      hamiltonian = _construct_block_hamiltonian(mol,nelecas,ncas,PO,h1eff,eri,TSc,energy_core,group)
      return hamiltonian
             
  def kernel_diag(self,mo = None, debug = False):
      '''
      Solve CI problem by just diagonalizing Hamiltonian matrix
      '''
      if mo is not None: self.mo_coeff = mo
      MO, PO, group = self.optimize_mo(mo, debug)
      Adm = self.get_active_dm(mo)
      if group is None :
         W, TSc = self.get_SVD_matrices(MO , PO)
      else: W, TSc = self.get_SVD_matrices(MO , group)       
      h1eff, energy_core = self.get_h1cas(Adm , MO , W)
      self.mo_eri = self.get_h2eff(mo)
      hamiltonian = self.construct_reduced_hamiltonian(mo, h1eff, energy_core, PO, TSc)
      eigenvalues, eigenvectors = gen_eig(hamiltonian)
      self.ci = eigenvectors
      return eigenvalues, eigenvectors
  
  def matrix_kernel(self, mo = None):
      '''Calculate necessary matrices before constructing hamiltonian.
      '''
      if mo is not None : self.mo_coeff = mo
      MO, PO, group = self.optimize_mo(mo)
      Adm = self.get_active_dm(mo)
      if group is None :
         W, TSc = self.get_SVD_matrices(MO , PO)
      else: W, TSc = self.get_SVD_matrices(MO , group)       
      h1eff, energy_core = self.get_h1cas(Adm , MO , W)
      self.mo_eri = self.get_h2eff(mo)
      return MO, PO, group, W, h1eff, energy_core, TSc

  def kernel(self, mo = None, ci0=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None,
               orbsym=None, wfnsym=None, ecore=0, **kwargs):
      if mo is not None: self.mo_coeff = mo
      cput0 = (logger.process_clock(), logger.perf_counter())
      MO, PO, group = self.optimize_mo(mo)
      cput1 = logger.timer(self, 'core-vir rotation', *cput0)
      Adm = self.get_active_dm(mo)
      if group is None :
         W, TSc = self.get_SVD_matrices(MO , PO)
      else: W, TSc = self.get_SVD_matrices(MO , group)       
      h1eff, energy_core = self.get_h1cas(Adm , MO , W)
      eri = self.get_h2eff(mo)
      self.mo_eri = eri
      cput1 = logger.timer(self,'Matrices calculation', *cput1)
      e, c = kernel_SFNOCI(self, h1eff, eri, self.ncas, self.nelecas, PO, group, TSc, energy_core, ci0, link_index=None,
                           tol = tol, lindep= lindep, max_cycle=max_cycle, max_space=max_space, nroots=nroots, davidson_only= davidson_only,
                           pspace_size= pspace_size, ecore= ecore, verbose=self.verbose, **kwargs)
      logger.timer(self, 'CI solving', *cput1)
      logger.timer(self, 'All process', *cput0)
      return e, c
  
  def skip_scf(self, mo = None, ci0=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None,
               orbsym=None, wfnsym=None, ecore=0, **kwargs ):
      if mo is not None: self.mo_coeff
      e, c = kernel_SFNOCI(self, self.h1eff, self.mo_eri, self.ncas, self.nelecas, self.PO, self.group, self.TSc, self.core_energies, ci0, link_index=None,
                           tol = tol, lindep= lindep, max_cycle=max_cycle, max_space=max_space, nroots=nroots, davidson_only= davidson_only,
                           pspace_size= pspace_size, ecore= ecore, verbose=self.verbose, **kwargs)
      return e, c

  def make_hdiag(self, h1eff = None, eri = None, ncas = None, nelecas = None, PO = None, group = None, energy_core = None, opt = None):
      if h1eff is None and energy_core is None: h1eff, energy_core = self.get_h1eff()
      if eri is None : eri = self.get_h2eff(self.mo_coeff)
      if ncas is None : ncas = self.ncas
      if nelecas is None : nelecas = self.nelecas
      if PO is None : PO = self.PO
      if group is None : group = self.group
      return make_hdiag(h1eff, eri, ncas, nelecas, PO, group, energy_core, opt)   
  
  def make_precond(self, hdiag = None, level_shift=0):
      if hdiag is None : hdiag = self.make_hdiag()
      return make_diag_precond(hdiag,level_shift)
 
  def make_rdm1s(self, mo_coeff = None, ci = None, W = None, PO = None,TSc = None, ncas = None, nelecas = None, ncore = None):
      if mo_coeff is None : mo_coeff = self.mo_coeff  
      if W is None : W = self.W
      if PO is None : PO = self.PO
      if TSc is None : TSc = self.TSc
      if ci is None : ci = self.ci
      if isinstance(ci,numpy.ndarray) and ci.ndim != 1: ci = ci[:,0]
      if ncas is None : ncas = self.ncas
      if nelecas is None : nelecas = self.nelecas
      if ncore is None : ncore = self.ncore
      N = mo_coeff.shape[0]
      group = self.group
      mo_cas = mo_coeff[:,ncore:ncore+ncas]
      stringsa = cistring.make_strings(range(ncas),nelecas[0])
      stringsb = cistring.make_strings(range(ncas),nelecas[1])
      link_indexa = cistring.gen_linkstr_index(range(ncas),nelecas[0])
      link_indexb = cistring.gen_linkstr_index(range(ncas),nelecas[1])
      na = cistring.num_strings(ncas,nelecas[0])
      nb = cistring.num_strings(ncas,nelecas[1])
      rdm1c = numpy.zeros((N,N))
      ci = ci.reshape(na,nb)
      for str0a, strsa in enumerate(stringsa):
          for str0b, strsb in enumerate(stringsb):
              w_occa = str2occ(strsa,ncas)
              w_occb = str2occ(strsb,ncas)
              w_occ = w_occa + w_occb
              p = find_matching_rows(PO,w_occ)[0]
              if group is not None: 
                       p = num_to_group(group,p)
              rdm1c += numpy.conjugate(ci[str0a,str0b])*ci[str0a,str0b]*W[p,p]
    
      rdm1asmoa = numpy.zeros((ncas,ncas))
      rdm1asmob = numpy.zeros((ncas,ncas))    
      for str0a , taba in enumerate(link_indexa):
          for aa, ia, str1a, signa in link_indexa[str0a]:
              for str0b, strsb in enumerate(stringsb):
                  w_occa = str2occ(stringsa[str0a],ncas)
                  w_occb = str2occ(strsb,ncas)
                  x_occa = str2occ(stringsa[str1a],ncas)
                  x_occ = x_occa + w_occb
                  w_occ = w_occa + w_occb   
                  p1 = find_matching_rows(PO,x_occ)[0]
                  p2 = find_matching_rows(PO,w_occ)[0]
                  if group is not None: 
                       p1 = num_to_group(group,p1)
                       p2 = num_to_group(group,p2) 
                  rdm1asmoa[aa,ia] += signa * numpy.conjugate(ci[str1a,str0b]) * ci[str0a,str0b] * TSc[p1,p2]
      for str0b, tabb in enumerate(link_indexb):
          for ab, ib, str1b, signb in link_indexb[str0b]:
              for str0a, strsa in enumerate(stringsa):
                  w_occa = str2occ(strsa,ncas)
                  w_occb = str2occ(stringsb[str0b],ncas)
                  x_occb = str2occ(stringsb[str1b],ncas)
                  x_occ = w_occa + x_occb
                  w_occ = w_occa + w_occb  
                  p1 = find_matching_rows(PO,x_occ)[0]
                  p2 = find_matching_rows(PO,w_occ)[0] 
                  if group is not None: 
                       p1 = num_to_group(group,p1)
                       p2 = num_to_group(group,p2)       
                  rdm1asmob[ab,ib] += signb * numpy.conjugate(ci[str0a,str1b]) * ci[str0a,str0b] * TSc[p1,p2]
      rdm1a = lib.einsum('ia,ab,jb -> ij', numpy.conjugate(mo_cas),rdm1asmoa,mo_cas)
      rdm1b = lib.einsum('ia,ab,jb-> ij', numpy.conjugate(mo_cas),rdm1asmob,mo_cas)
      rdm1a += rdm1c
      rdm1b += rdm1c
      
      return rdm1a, rdm1b
  def make_rdm1(self, mo_coeff = None, ci = None, W = None, PO = None,TSc = None, ncas = None, nelecas = None, ncore = None):
      rdm1a, rdm1b = self.make_rdm1s(mo_coeff ,ci,  W , PO ,TSc , ncas , nelecas , ncore)
      return rdm1a + rdm1b

  def make_rdm2s(self, mo_coeff = None, ci = None, Adm = None, W = None, PO = None, TSc = None, ncas =None, nelecas = None, ncore = None):
      if mo_coeff is None : mo_coeff = self.mo_coeff  
      if W is None : W = self.W
      if Adm is None : Adm = self.Adm
      if PO is None : PO = self.PO
      if TSc is None : TSc = self.TSc
      if ci is None : ci = self.ci
      if isinstance(ci,numpy.ndarray) and ci.ndim != 1: ci = ci[:,0]
      if ncas is None : ncas = self.ncas
      if nelecas is None : nelecas = self.nelecas
      if ncore is None : ncore = self.ncore
      s1e = self.mol.intor('int1e_ovlp')
      mo_cas = mo_coeff[:,ncore:ncore+ncas]
      N = mo_coeff.shape[0]
      group = self.group
      rdm2aa = numpy.zeros((N,N,N,N))
      rdm2ab = numpy.zeros((N,N,N,N))
      rdm2ba = numpy.zeros((N,N,N,N))
      rdm2bb = numpy.zeros((N,N,N,N))
      stringsa = cistring.make_strings(range(ncas),nelecas[0])
      stringsb = cistring.make_strings(range(ncas),nelecas[1])
      link_indexa = cistring.gen_linkstr_index(range(ncas),nelecas[0])
      link_indexb = cistring.gen_linkstr_index(range(ncas),nelecas[1])
      na = cistring.num_strings(ncas,nelecas[0])
      nb = cistring.num_strings(ncas,nelecas[1])
      ci = ci.reshape(na,nb)
      t2aa = numpy.zeros((ncas,ncas,ncas,ncas,na,na))
      t2bb = numpy.zeros((ncas,ncas,ncas,ncas,nb,nb))
      t1a = numpy.zeros((ncas,ncas,na,na))
      t1b = numpy.zeros((ncas,ncas,nb,nb)) 
      
      rdm2aaac = numpy.zeros((ncas,ncas,ncas,ncas))
      rdm2abac = numpy.zeros((ncas,ncas,ncas,ncas))
      rdm2baac = numpy.zeros((ncas,ncas,ncas,ncas))
      rdm2bbac = numpy.zeros((ncas,ncas,ncas,ncas))
      for str0a , taba in enumerate(link_indexa):
          for a1, i1, str1a, signa1 in link_indexa[str0a]:
              t1a[a1,i1,str1a,str0a] += signa1 
              for a2 , i2, str2a, signa2 in link_indexa[str1a]:
                  t2aa[a2, i2, a1, i1, str2a, str0a] += signa1 * signa2
      for str0b , tabb in enumerate(link_indexb):
          for a1, i1, str1b, signb1 in link_indexb[str0b]:
              t1b[a1,i1,str1b,str0b] += signb1
              for a2 , i2, str2b, signb2 in link_indexb[str1b]:
                  t2bb[a2, i2, a1, i1, str2b, str0b] += signb1 * signb2
      for str0a, strs0a in enumerate(stringsa):
          for str0b, strs0b in enumerate(stringsb):
              w_occa = str2occ(strs0a,ncas)
              w_occb = str2occ(strs0b,ncas)
              w_occ = w_occa + w_occb
              p2 = find_matching_rows(PO,w_occ)[0]
              if group is not None: 
                 p2 = num_to_group(group,p2) 
              rdm2aa += numpy.conjugate(ci[str0a,str0b])*ci[str0a,str0b] * (lib.einsum('pq,rs -> pqrs', W[p2,p2,:,:],W[p2,p2,:,:]) - lib.einsum('ps,rq -> pqrs',W[p2,p2,:,:],W[p2,p2,:,:])) 
              rdm2ab += numpy.conjugate(ci[str0a,str0b])*ci[str0a,str0b] * lib.einsum('pq,rs -> pqrs',W[p2,p2,:,:],W[p2,p2,:,:])
              rdm2ba += numpy.conjugate(ci[str0a,str0b])*ci[str0a,str0b] * lib.einsum('pq,rs -> pqrs',W[p2,p2,:,:],W[p2,p2,:,:])
              rdm2bb += numpy.conjugate(ci[str0a,str0b])*ci[str0a,str0b] * (lib.einsum('pq,rs -> pqrs', W[p2,p2,:,:],W[p2,p2,:,:]) - lib.einsum('ps,rq -> pqrs',W[p2,p2,:,:],W[p2,p2,:,:]))  
              for str1a, strs1a in enumerate(stringsa):
                  x_occa = str2occ(strs1a,ncas)
                  x_occ = x_occa + w_occb
                  p1 = find_matching_rows(PO,x_occ)[0]
                  if group is not None: 
                       p1 = num_to_group(group,p1)
                  rdm2aaac[:,:,:,:] += numpy.conjugate(ci[str1a,str0b])*ci[str0a,str0b]*t2aa[:,:,:,:,str1a,str0a]*TSc[p1,p2]
                  for k in range(ncas):
                      rdm2aaac[:,k,k,:] -= numpy.conjugate(ci[str1a,str0b])*ci[str0a,str0b]*t1a[:,:,str1a,str0a]*TSc[p1,p2]
              for str1b, strs1b in enumerate(stringsb):
                  x_occb = str2occ(strs1b,ncas)
                  x_occ = w_occa + x_occb
                  p1 = find_matching_rows(PO,x_occ)[0]
                  if group is not None: 
                       p1 = num_to_group(group,p1)
                  rdm2bbac[:,:,:,:] += numpy.conjugate(ci[str0a,str1b])*ci[str0a,str0b]*t2bb[:,:,:,:,str1b,str0b]*TSc[p1,p2]
                  for k in range(ncas):
                      rdm2bbac[:,k,k,:] -= numpy.conjugate(ci[str0a,str1b])*ci[str0a,str0b] * t1b[:,:,str1b,str0b]*TSc[p1,p2]
              for str1a, strs1a in enumerate(stringsa):
                  for str1b, strs1b in enumerate(stringsb):
                      w_occa = str2occ(strs0a,ncas)
                      w_occb = str2occ(strs0b,ncas)
                      x_occa = str2occ(strs1a,ncas)
                      x_occb = str2occ(strs1b,ncas)
                      w_occ = w_occa + w_occb
                      x_occ = x_occa + x_occb
                      p1 = find_matching_rows(PO,x_occ)[0] 
                      p2 = find_matching_rows(PO,w_occ)[0]
                      if group is not None: 
                         p1 = num_to_group(group,p1)
                         p2 = num_to_group(group,p2)  
                      rdm2abac += numpy.conjugate(ci[str1a,str1b])*ci[str0a,str0b]*lib.einsum('pq,rs-> pqrs',t1a[:,:,str1a,str0a],t1b[:,:,str1b,str0b])*TSc[p1,p2]
                      rdm2baac += numpy.conjugate(ci[str1a,str1b])*ci[str0a,str0b]*lib.einsum('pq,rs-> pqrs',t1b[:,:,str1b,str0b],t1a[:,:,str1a,str0a])*TSc[p1,p2]
      
      rdm2aa += lib.einsum('pa,qb,rc,sd,abcd -> pqrs',mo_cas,mo_cas,mo_cas,mo_cas,rdm2aaac)
      rdm2ab += lib.einsum('pa,qb,rc,sd,abcd -> pqrs',mo_cas,mo_cas,mo_cas,mo_cas,rdm2abac)
      rdm2ba += lib.einsum('pa,qb,rc,sd,abcd -> pqrs',mo_cas,mo_cas,mo_cas,mo_cas,rdm2baac)
      rdm2bb += lib.einsum('pa,qb,rc,sd,abcd -> pqrs',mo_cas,mo_cas,mo_cas,mo_cas,rdm2bbac)
      t1aao = lib.einsum('ia,jb,abcd -> ijcd', mo_cas, mo_cas, t1a)
      t1bao = lib.einsum('ia,jb,abcd -> ijcd', mo_cas, mo_cas, t1b)
      
      
      for str0a, taba in enumerate(link_indexa):
          for str1a in numpy.unique(link_indexa[str0a][:,2]):
              for str0b, strsb in enumerate(stringsb):
                  w_occa = str2occ(stringsa[str0a],ncas)
                  w_occb = str2occ(strsb,ncas)
                  x_occa = str2occ(stringsa[str1a],ncas)
                  x_occ = x_occa + w_occb
                  w_occ = w_occa + w_occb   
                  p1 = find_matching_rows(PO,x_occ)[0]
                  p2 = find_matching_rows(PO,w_occ)[0]
                  if group is not None: 
                       p1 = num_to_group(group,p1)
                       p2 = num_to_group(group,p2) 
                  rdm2aa += numpy.conjugate(ci[str1a,str0b])*ci[str0a,str0b]*(lib.einsum('pq,rs->pqrs',t1aao[:,:,str1a,str0a],W[p1,p2,:,:])+lib.einsum('rs,pq->pqrs',t1aao[:,:,str1a,str0a],W[p1,p2,:,:])-lib.einsum('ps,rq->pqrs',t1aao[:,:,str1a,str0a],W[p1,p2,:,:])-lib.einsum('rq,ps->pqrs',t1aao[:,:,str1a,str0a],W[p1,p2,:,:]))*TSc[p1,p2]
                  rdm2ab += numpy.conjugate(ci[str1a,str0b])*ci[str0a,str0b]*(lib.einsum('pq,rs->pqrs',t1aao[:,:,str1a,str0a],W[p1,p2,:,:]))*TSc[p1,p2]
                  rdm2ba += numpy.conjugate(ci[str1a,str0b])*ci[str0a,str0b]*(lib.einsum('rs,pq->pqrs',t1aao[:,:,str1a,str0a],W[p1,p2,:,:]))*TSc[p1,p2]
    
      for str0b, tabb in enumerate(link_indexb):
          for str1b in numpy.unique(link_indexb[str0b][:,2]):
              for str0a, strsa, in enumerate(stringsa):
                  w_occa = str2occ(strsa,ncas)
                  w_occb = str2occ(stringsb[str0b],ncas)
                  x_occb = str2occ(stringsb[str1b],ncas)
                  x_occ = w_occa + x_occb
                  w_occ = w_occa + w_occb  
                  p1 = find_matching_rows(PO,x_occ)[0]
                  p2 = find_matching_rows(PO,w_occ)[0]
                  if group is not None: 
                       p1 = num_to_group(group,p1)
                       p2 = num_to_group(group,p2)  
                  rdm2bb += numpy.conjugate(ci[str0a,str1b])*ci[str0a,str0b]* (lib.einsum('pq,rs->pqrs',t1bao[:,:,str1b,str0b],W[p1,p2,:,:])+lib.einsum('rs,pq->pqrs',t1bao[:,:,str1b,str0b],W[p1,p2,:,:])-lib.einsum('ps,rq->pqrs',t1bao[:,:,str1b,str0b],W[p1,p2,:,:])-lib.einsum('rq,ps->pqrs',t1bao[:,:,str1b,str0b],W[p1,p2,:,:]))*TSc[p1,p2]
                  rdm2ab += numpy.conjugate(ci[str0a,str1b])*ci[str0a,str0b]* (lib.einsum('rs,pq->pqrs',t1bao[:,:,str1b,str0b],W[p1,p2,:,:]))*TSc[p1,p2]
                  rdm2ba += numpy.conjugate(ci[str0a,str1b])*ci[str0a,str0b]* (lib.einsum('pq,rs->pqrs',t1bao[:,:,str1b,str0b],W[p1,p2,:,:]))*TSc[p1,p2]           
      
      return rdm2aa, rdm2ab, rdm2ba, rdm2bb
  
  def make_rdm2(self, mo_coeff = None, ci = None, Adm = None, W = None, PO = None, TSc = None, ncas =None, nelecas = None, ncore = None):
      rdm2aa, rdm2ab, rdm2ba, rdm2bb = self.make_rdm2s(mo_coeff, ci, Adm, W, PO, TSc, ncas, nelecas, ncore)
      return rdm2aa + rdm2ab + rdm2ba + rdm2bb
  
  def contract_H(self, erieff, civec, ncas, nelecas, PO, group, TSc, energy_core ,link_index = None):
      return  contract_H(erieff, civec, ncas, nelecas, PO, group, TSc, energy_core ,link_index)
  
  def contract_ss(self, civec, ncas = None, nelecas = None):
      if ncas is None : ncas = self.ncas
      if nelecas is None : nelecas = self.nelecas
      return spin_op.contract_ss(civec,ncas,nelecas)
  
  def spin_square(self, civec, ncas = None, nelecas = None):
      if ncas is None : ncas = self.ncas
      if nelecas is None : nelecas = self.nelecas
      return spin_op.spin_square0(civec, ncas, nelecas)

  def eig(self, op, x0=None, precond=None, **kwargs):
        if isinstance(op, numpy.ndarray):
            self.converged = True
            return scipy.linalg.eigh(op)
        
        self.converged, e, ci = \
                lib.davidson1(lambda xs: [op(x) for x in xs],
                              x0, precond, lessio=False, **kwargs)
        if kwargs['nroots'] == 1:
            self.converged = self.converged[0]
            e = e[0]
            ci = ci[0]
        return e, ci
  
  def fix_spin_(self, shift=PENALTY, ss = None):
       '''Use level shift to control FCI solver spin.

        .. math::

            (H + shift*S^2) |\Psi\rangle = E |\Psi\rangle

        Kwargs:
            shift : float
                Energy penalty for states which have wrong spin
            ss : number
                S^2 expection value == s*(s+1)
        '''
       fix_spin_(self, shift, ss)
       return self
  fix_spin = fix_spin_

  def S2_list(self, mo_coeff = None, ci = None):
      if ci is None: ci = self.ci
      cin = ci.shape[1]
      S2_list = numpy.zeros(cin) 
      for i in range(cin):
          civec = ci[:,i]
          #rdm1 = self.make_rdm1(mo_coeff, ci = civec)
          #rdm2aa, rdm2ab, rdm2ba, rdm2bb = self.make_rdm2s(mo_coeff, ci = civec)
          S2_list[i] = self.spin_square(civec)[0]
      return S2_list    
  
class SpinPenaltySFNOCISolver:
    __name_mixin__ = 'SpinPenalty'
    _keys = {'ss_value', 'ss_penalty', 'base'}
    def __init__(self, mySFNOCI, shift, ss_value):
        self.base = mySFNOCI.copy()
        self.__dict__.update (mySFNOCI.__dict__)
        self.ss_value = ss_value
        self.ss_penalty = shift
        self.davidson_only = self.base.fcisolver.davidson_only = True

    def undo_fix_spin(self):
        obj = lib.view(self, lib.drop_class(self.__class__, SpinPenaltySFNOCISolver))
        del obj.base
        del obj.ss_value
        del obj.ss_penalty
        return obj
    
    def base_contract_H (self, *args, **kwargs):
        return super().contract_H (*args, **kwargs)
    
    def contract_H(self, erieff, civec, ncas, nelecas, PO, group, TSc, energy_core ,link_index = None, **kwargs):
        if isinstance(nelecas, (int, numpy.number)):
            sz = (nelecas % 2) * .5
        else:
            sz = abs(nelecas[0]-nelecas[1]) * .5
        if self.ss_value is None:
            ss = sz*(sz+1)
        else:
            ss = self.ss_value
        if ss < sz*(sz+1)+.1:
            # (S^2-ss)|Psi> to shift state other than the lowest state
            ci1 = self.contract_ss(civec, ncas, nelecas).reshape(civec.shape)
            ci1 -= ss * civec
        else:
            # (S^2-ss)^2|Psi> to shift states except the given spin.
            # It still relies on the quality of initial guess
            tmp = self.contract_ss(civec, ncas, nelecas).reshape(civec.shape)
            tmp -= ss * civec
            ci1 = -ss * tmp
            ci1 += self.contract_ss(tmp, ncas, nelecas).reshape(civec.shape)
            tmp = None
        ci1 *= self.ss_penalty
        ci0 = super().contract_H(erieff, civec, ncas, nelecas, PO, group, TSc, energy_core, link_index, **kwargs)
        ci1 += ci0.reshape(civec.shape)
        return ci1
    
if  __name__ == '__main__':
    from pyscf import scf
    import matplotlib.pyplot as plt
    import pandas as pd
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = None

    mol.atom = [['Li', (0, 0, 0)],['F',(0,0,1.4)]]
    mol.basis = 'ccpvdz'
    
    x_list=[]
    e1_list=[]
    e2_list=[]
    e3_list=[]
    e4_list=[]
    e5_list=[]
    e6_list=[]
    e7_list=[]
    e8_list=[]
    ma=[]
    mode=0
    
    mol.spin=2
    mol.build(0,0)
    rm=scf.ROHF(mol)
    rm.kernel()
  
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
    #setocc[1][2]=0
    setocc[1][3]=0
    setocc[0][6]=1
    #setocc[0][9]=1
    #setocc[0][8]=1
    ro_occ=setocc[0][:]+setocc[1][:]
    dm_ro=rm.make_rdm1(mo0,ro_occ)
    rm=scf.addons.mom_occ(rm,mo0,setocc)
    rm.scf(dm_ro)
    mo=rm.mo_coeff
    AS_list=[3,6,7,10]
    s1e = mol.intor('int1e_ovlp')
    mySFNOCI = SFNOCI(rm,4,4)
    mySFNOCI.spin = 0
    from pyscf.mcscf import addons
    mo = addons.sort_mo(mySFNOCI,rm.mo_coeff, AS_list,1)
    
    reei, ev = mySFNOCI.kernel(mo, nroots = 4)
    print(reei,ev)
    
    i=1
    while i<=4:
        mol.atom=[['Li',(0,0,0)],['F',(0,0,i)]]
        mol.build(0,0)
        mol.spin=2
        m=scf.RHF(mol)
        m.kernel()
        m=scf.addons.mom_occ(m,mo0,setocc)
        m.scf(dm_ro)

        mySFNOCI = SFNOCI(m,4,4,groupA = [[0,1],[2,3]])
        mySFNOCI.spin = 0
        mo = addons.sort_mo(mySFNOCI,m.mo_coeff,AS_list,1)
        eigenvalues, eigenvectors = mySFNOCI.kernel(mo, nroots = 4)


        print(eigenvalues)
        x_list.append(i)
        e1_list.append((eigenvalues[0]-reei[0])*627.503)
        e2_list.append((eigenvalues[1]-reei[0])*627.503)
        e3_list.append((eigenvalues[2]-reei[0])*627.503)
        e4_list.append((eigenvalues[3]-reei[0])*627.503)
        
        #e5_list.append((eigenvalues[4]-reei[0])*627.503)
        #e6_list.append((eigenvalues[5]-reei[0])*627.503)
        #e7_list.append((eigenvalues[6]-reei[0])*627.503)
        #e8_list.append((eigenvalues[7]-reei[0])*627.503)

        #ma.append([i,(eigenvalues[0]-reei[0])*627.503,(eigenvalues[1]-reei[0])*627.503,(eigenvalues[2]-reei[0])*627.503, (eigenvalues[3]-reei[0])*627.503])
        #ma.append([i,S2[0],S2[1],S2[2],S2[3]])
        #print(S2)
        i+=0.1
        
    plt.plot(x_list,e1_list,label='ground state(Singlet)')
    plt.plot(x_list,e2_list,label='first excited state(Triplet)')
    plt.plot(x_list,e3_list,label='second excited state(Singlet)')
    plt.plot(x_list,e4_list,label='3excited state')
    #plt.plot(x_list,e5_list,label='4excited state')
    #plt.plot(x_list,e6_list,label='5excited state')
    #plt.plot(x_list,e7_list,label='6excited state')
    #plt.plot(x_list,e8_list,label='7excited state')

    plt.legend()
    plt.show()
    #df=pd.DataFrame(ma)
    #df.to_excel('LiF_Spinsquare_SFNOCI.xlsx',index=False)
        


    
    
  
    
    
