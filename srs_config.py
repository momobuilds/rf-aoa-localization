from dataclasses import dataclass
import numpy as np
from sympy import primerange

#currently the scheduling is not implemented: #todo
    #pag 324 36.213 version 18.3.0 Release 18 to implement scheduling (10*nframe + sub frame index within frame - T_offset) mod T_periodicity = 0
    #5.5.3.3 Sounding reference signal subframe configuration  (keep 0 for all the subframes)
#should be both satisfied to transmit a single SRS


class SystemParams:
    def __init__(self, C_RNTI, N_RB_UL,  PCID, sequence_hopping_enabled = 0, FDD = 1, group_hopping_enabled = 0):
        self.FDD = FDD
        self.C_RNTI = C_RNTI
        self.N_RB_UL = N_RB_UL       
        self.PCID = PCID
        self.group_hopping_enabled = group_hopping_enabled
        self.sequence_hopping_enabled = sequence_hopping_enabled

class SRSConfigCommon:
    def __init__(self, srs_bandwidthConfig = 5, srs_subframeConfig = 0):
        self.srs_bandwidthConfig = srs_bandwidthConfig
        self.srs_subframeConfig = srs_subframeConfig


class SRSConfigDedicated:
    def __init__(self, srs_bandwidth = 0, srs_hoppingBandwidth = 3, freqDomainPosition = 0,
        duration = 1, config_Index = 77, transmissionComb = 0, cyclicShift = 0, srs_AntennaPort = 10, transmissionCombNum = 2):
        self.srs_bandwidth = srs_bandwidth
        self.srs_hoppingBandwidth = srs_hoppingBandwidth
        self.freqDomainPosition = freqDomainPosition
        self.duration = duration
        self.config_Index = config_Index
        self.transmissionComb = transmissionComb
        self.cyclicShift = cyclicShift
        self.srs_AntennaPort = srs_AntennaPort
        self.transmissionCombNum = transmissionCombNum


#Input parameters
def generateSRSsequence(cfg_ded, cfg_com, sys_params):

    group_hopping_enabled = sys_params.group_hopping_enabled
    sequence_hopping_enabled = sys_params.sequence_hopping_enabled
    N_RB_UL = sys_params.N_RB_UL
    PCID = sys_params.PCID


    C_SRS = cfg_com.srs_bandwidthConfig
            
    
    n_cs = cfg_ded.cyclicShift
    K_TC = cfg_ded.transmissionCombNum
    B_SRS = cfg_ded.srs_bandwidth
    srs_AntennaPort = cfg_ded.srs_AntennaPort 
    periodic = cfg_ded.duration
    k_tc = cfg_ded.transmissionComb
    b_hop = cfg_ded.srs_hoppingBandwidth
    n_RRC = cfg_ded.freqDomainPosition
    config_idx = cfg_ded.config_Index

    if group_hopping_enabled == 1:
        raise ValueError("Group hopping should be disabled!")
    if sequence_hopping_enabled == 1:
        raise ValueError("Sequence hopping should be disabled!")
    if K_TC > 2:
        raise ValueError("transmissionCombNum should be 2")
    if srs_AntennaPort != 10:
        raise ValueError("The UE should only use a single antenna port")
    if periodic == 0:
        raise ValueError("The SRS transmission should be periodic")
    if B_SRS > b_hop:
        raise ValueError("The SRS hopping should be disabled")
    if sys_params.FDD == 0:
        raise ValueError("Only FDD is supported")
    

    r = generate_base_sequence(PCID, C_SRS, B_SRS, N_RB_UL, K_TC, n_cs)
    k0, sc_idx = get_srs_starting_subcarrier(B_SRS, C_SRS, N_RB_UL, K_TC, k_tc, n_RRC)
    T_per, T_off = get_srs_periodicity_and_subframe_offset(config_idx)
    

    return r, k0, sc_idx, T_per, T_off


def get_srs_starting_subcarrier(B_SRS, C_SRS, N_RB_UL, K_TC, k_tc, n_RRC):

    m_srs = get_m_srs(C_SRS, B_SRS, N_RB_UL)
    M_SRS = int(m_srs*12/K_TC)

    M_SRS_vec = np.zeros(B_SRS + 1, dtype=int)
    n_b_vec = np.zeros(B_SRS + 1, dtype=int)
    for b in range(B_SRS + 1):
        m_srs_b = get_m_srs(C_SRS, b, N_RB_UL)
        M_SRS_vec[b] = m_srs_b * 12 / K_TC
        N_b = get_N_b(C_SRS, b, N_RB_UL)
        n_b_vec[b] = np.floor(4*n_RRC / m_srs_b) % N_b
        

    m_srs_0 = get_m_srs(C_SRS, 0, N_RB_UL)
    k0_bar = (np.floor(N_RB_UL/2) - m_srs_0/2)*12 + k_tc 
    k0 = k0_bar + K_TC * np.sum(M_SRS_vec * n_b_vec)

    if k0 < 0:
        raise ValueError("C_SRS is too low to N_RB_UL currently used")

    sc_idx = k0 + np.arange(0, K_TC*M_SRS, K_TC)

    return k0, sc_idx


def get_srs_periodicity_and_subframe_offset(srs_configuration_index):
    #subframeoffset [subframes]
    #periodicity [ms]

    idx = srs_configuration_index

    if 0 <= idx <= 1:
        periodicity = 2
        offset = idx          # ISRS

    elif 2 <= idx <= 6:
        periodicity = 5
        offset = idx - 2      # ISRS - 2

    elif 7 <= idx <= 16:
        periodicity = 10
        offset = idx - 7      # ISRS - 7

    elif 17 <= idx <= 36:
        periodicity = 20
        offset = idx - 17     # ISRS - 17

    elif 37 <= idx <= 76:
        periodicity = 40
        offset = idx - 37     # ISRS - 37

    elif 77 <= idx <= 156:
        periodicity = 80
        offset = idx - 77     # ISRS - 77

    elif 157 <= idx <= 316:
        periodicity = 160
        offset = idx - 157    # ISRS - 157

    elif 317 <= idx <= 636:
        periodicity = 320
        offset = idx - 317    # ISRS - 317

    else:
        # 637–1023 -> reserved
        return None, None

    return periodicity, offset

def get_m_srs(C_SRS, B_srs, N_RB_UL):
        #Table 5.5.3.2-1 36.211
    m_table = {
        0: [36, 12, 4, 4],
        1: [32, 16, 8, 4],
        2: [24, 4, 4, 4],
        3: [20, 4, 4, 4],
        4: [16, 4, 4, 4],
        5: [12, 4, 4, 4],
        6: [8,  4, 4, 4],
        7: [4,  4, 4, 4]
    }

    # Controlli
    if C_SRS not in m_table:
        raise ValueError("C_SRS must be in range 0–7")
    if B_srs not in [0, 1, 2, 3]:
        raise ValueError("B_srs must be 0, 1, 2, or 3")
    if not (6 <= N_RB_UL <= 40):
        raise ValueError("N_RB_UL must be between 6 and 40")

    return m_table[C_SRS][B_srs]


def get_N_b(C_SRS, B_srs, N_RB_UL):
    # #Table 5.5.3.2-1 36.211
    N_table = {
        0: [1, 3, 3, 1],
        1: [1, 2, 2, 2],
        2: [1, 6, 1, 1],
        3: [1, 5, 1, 1],
        4: [1, 4, 1, 1],
        5: [1, 3, 1, 1],
        6: [1, 2, 1, 1],
        7: [1, 1, 1, 1]
    }

    # Controlli
    if C_SRS not in N_table:
        raise ValueError("C_SRS must be in range 0–7")
    if B_srs not in [0, 1, 2, 3]:
        raise ValueError("B_srs must be 0, 1, 2, or 3")
    if not (6 <= N_RB_UL <= 40):
        raise ValueError("N_RB_UL must be between 6 and 40")

    return N_table[C_SRS][B_srs]

def generate_base_sequence(PCID, C_SRS, B_SRS, N_RB_UL, K_TC, n_cs):
             
        u = PCID % 30  #here is assumed that srs-VirtualCellID is FALSE
        v = 0
        
        m_srs = get_m_srs(C_SRS, B_SRS, N_RB_UL)
        M_SRS = int(m_srs * 12 / K_TC)
        
        if M_SRS >= 36:
            r_ = zadoff_chu(u, v, M_SRS)
        else:
            r_ = short_base_sequence(u, M_SRS)

        alpha = 2*np.pi*n_cs/8 

        n = np.arange(M_SRS)
        r = r_*np.exp(1j*alpha*n)

        return r

def zadoff_chu(u, v, M):

    p = list(primerange(1, M))
    if len(p) == 0:
        raise ValueError(f"No primes < {M}")

    N_ZC = p[-1]

    q_ = N_ZC * (u + 1) / 31
    q = int(round(q_)) + v * ((-1) ** int(np.floor(2 * q_)))

    m = np.arange(N_ZC)
    x_q = np.exp(-1j * np.pi * q * m * (m + 1) / N_ZC)

    n = np.arange(M)
    r_ = x_q[np.mod(n, N_ZC)]

    return r_

def short_base_sequence(u, M):

    if M == 12:
        phi = extractPhiRow_T55121(u)
    elif M == 24:
        phi = extractPhiRow_T55122(u)
    elif M == 18:
        phi = extractPhiRow_T55124(u)

    phi = np.array(phi) 
    r_ = np.exp(1j * phi * np.pi / 4)
    return r_

def extractPhiRow_T55124(u):

    phi_table = [
    [-3, -3, -3, -3, -3, -1,  1, -1, -3,  3, -1,  3, -1,  3, -3, -1, -1,  3],  # u = 0
    [-3, -3, -3, -3, -3, -1,  1, -1,  1, -3, -3, -3,  1, -1,  3, -3, -3,  1],  # u = 1
    [-3, -3, -3, -3, -3, -1,  1,  1,  3, -3,  1,  1, -3,  1, -3,  3,  1, -1],  # u = 2
    [-3, -3, -3, -3, -3, -1,  1,  3, -3, -1,  3, -1,  3,  1, -1, -3,  3, -3],  # u = 3
    [-3, -3, -3, -3, -3, -1,  3, -3, -1,  1, -1, -3,  3,  3,  1, -3,  1, -1],  # u = 4
    [-3, -3, -3, -3, -3,  1, -3, -3, -3, -3,  1,  1,  1, -3,  1,  1, -3, -3],  # u = 5
    [-3, -3, -3, -3, -3,  1, -3, -3,  1,  1, -3, -3, -3,  1, -1,  3, -1,  3],  # u = 6
    [-3, -3, -3, -3, -3,  1, -3, -1,  3, -1,  3,  3, -1, -1,  1,  3,  3, -1],  # u = 7
    [-3, -3, -3, -3, -3,  1, -1, -1, -1, -3,  3, -1,  3, -3,  3, -1,  1,  3],  # u = 8
    [-3, -3, -3, -3, -3,  3, -3,  1, -1,  3, -3,  3,  3, -1, -3,  1,  1, -3],  # u = 9
    [-3, -3, -3, -3, -3,  3, -1, -3, -3,  1,  1,  3, -3, -1,  3, -1,  3,  1],  # u = 10
    [-3, -3, -3, -3, -3,  3,  3, -1, -1, -1,  3,  1, -3,  3, -1,  1, -3,  1],  # u = 11
    [-3, -3, -3, -3, -1, -3, -3, -3,  1,  3,  1, -1,  3, -3, -1, -3,  1,  1],  # u = 12
    [-3, -3, -3, -3, -1, -3, -3,  1, -1, -1,  3, -3, -3,  1,  3,  1, -3,  1],  # u = 13
    [-3, -3, -3, -3, -1, -3, -3,  1,  3, -3, -1,  3,  1,  3, -1,  3, -1, -3],  # u = 14
    [-3, -3, -3, -3, -1, -3, -1,  3, -3,  1, -3,  1, -1, -3, -3,  1,  1,  3],  # u = 15
    [-3, -3, -3, -3, -1, -1,  3, -3,  3, -1, -3,  1,  1, -1, -3, -1,  3, -3],  # u = 16
    [-3, -3, -3, -3, -1, -1,  3, -1, -3,  1,  3, -1, -3, -3,  1,  3, -1,  1],  # u = 17
    [-3, -3, -3, -3, -1,  3, -1, -1,  3,  3, -1, -3,  1,  1,  1, -1, -3, -1],  # u = 18
    [-3, -3, -3, -3, -1,  3,  1, -3, -1, -3,  3,  1, -1,  3, -1,  1,  3, -1],  # u = 19
    [-3, -3, -3, -3,  1, -3, -3,  3,  1,  1, -3, -1,  1,  3,  3, -1,  3, -1],  # u = 20
    [-3, -3, -3, -3,  1, -3,  1,  3,  1, -1, -1,  3,  3, -1,  1,  1, -3,  3],  # u = 21
    [-3, -3, -3, -3,  1, -3,  3, -3, -1,  3,  1,  1, -1, -1,  3,  3, -1,  3],  # u = 22
    [-3, -3, -3, -3,  1, -3,  3, -1,  3, -3, -1, -1, -1,  1, -3, -3,  3,  1],  # u = 23
    [-3, -3, -3, -3,  1,  1,  3,  1,  1, -1,  3,  1,  1,  3, -1, -3,  1,  3],  # u = 24
    [-3, -3, -3, -3,  1,  3,  3,  3,  1, -3,  1, -3, -3,  3, -3,  1, -1, -3],  # u = 25
    [-3, -3, -3, -3,  3,  1,  3,  3, -1,  3, -3, -3, -1,  3, -1, -1, -3,  1],  # u = 26
    [-3, -3, -3, -1, -3, -3, -1, -1, -3,  3,  3,  1, -3, -1, -1,  3,  1, -3],  # u = 27
    [-3, -3, -3, -1, -3,  1, -1,  1, -3,  3,  1, -3, -1,  1,  3,  1, -1, -1],  # u = 28
    [-3, -3, -3, -1, -3,  3,  1,  1, -1, -1,  1,  3,  1, -3,  1, -3, -1,  1]   # u = 29
    ]

    row = phi_table[u]
    return row

def extractPhiRow_T55122(u):

    phi_table = [
    [-1,  3,  1, -3,  3, -1,  1,  3, -3,  3,  1,  3, -3,  3,  1,  1, -1,  1,  3, -3,  3, -3, -1, -3],  # u = 0
    [-3,  3, -3, -3, -3,  1, -3, -3,  3, -1,  1,  1,  1,  3,  1, -1,  3, -3, -3,  1,  3,  1,  1, -3],  # u = 1
    [3, -1,  3,  3,  1,  1, -3,  3,  3,  3,  3,  1, -1,  3, -1,  1,  1, -1, -3, -1, -1,  1,  3,  3],  # u = 2
    [-1, -3,  1,  1,  3, -3,  1,  1, -3, -1, -1,  1,  3,  1,  3,  1, -1,  3,  1,  1, -3, -1, -3, -1],  # u = 3
    [-1, -1, -1, -3, -3, -1,  1,  1,  3,  3, -1,  3, -1,  1, -1, -3,  1, -1, -3, -3,  1, -3, -1, -1],  # u = 4
    [-3,  1,  1,  3, -1,  1,  3,  1, -3,  1, -3,  1,  1, -1, -1,  3, -1, -3,  3, -3, -3, -3,  1,  1],   # u = 5
    [1,  1, -1, -1,  3, -3, -3,  3, -3,  1, -1, -1,  1, -1,  1,  1, -1, -3, -1,  1, -1,  3, -1, -3],   # u = 6
    [-3,  3,  3, -1, -1, -3, -1,  3,  1,  3,  1,  3,  1,  1, -1,  3,  1, -1,  1,  3, -3, -1, -1,  1],   # u = 7
    [-3,  1,  3, -3,  1, -1, -3,  3, -3,  3, -1, -1, -1, -1,  1, -3, -3, -3,  1, -3, -3, -3,  1, -3],   # u = 8
    [1,  1, -3,  3,  3, -1, -3, -1,  3, -3,  3,  3,  3, -1,  1,  1, -3,  1, -1,  1,  1, -3,  1,  1],   # u = 9
    [-1,  1, -3, -3,  3, -1,  3, -1, -1, -3, -3, -3, -1, -3, -3,  1, -1,  1,  3,  3, -1,  1, -1,  3],   # u = 10
    [1,  3,  3, -3, -3,  1,  3,  1, -1, -3, -3, -3,  3,  3, -3,  3,  3, -1, -3,  3, -1,  1, -3,  1],   # u = 11
    [1,  3,  3,  1,  1,  1, -1, -1,  1, -3,  3, -1,  1,  1, -3,  3,  3, -1, -3,  3, -3, -1, -3, -1],   # u = 12
    [3, -1, -1, -1, -1, -3, -1,  3,  3,  1, -1,  1,  3,  3,  3, -1,  1,  1, -3,  1,  3, -1, -3,  3],   # u = 13
    [-3, -3,  3,  1,  3,  1, -3,  3,  1,  3,  1,  1,  3,  3, -1, -1, -3,  1, -3, -1,  3,  1,  1,  3],   # u = 14
    [-1, -1,  1, -3,  1,  3, -3,  1, -1, -3, -1,  3,  1,  3,  1, -1, -3, -3, -1, -1, -3, -3, -3, -1],   # u = 15
    [-1, -3,  3, -1, -1, -1, -1,  1,  1, -3,  3,  1,  3,  3,  1, -1,  1, -3,  1, -3,  1,  1, -3, -1],   # u = 16
    [1,  3, -1,  3,  3, -1, -3,  1, -1, -3,  3,  3,  3, -1,  1,  1,  3, -1, -3, -1,  3, -1, -1, -1],   # u = 17
    [1,  1,  1,  1,  1, -1,  3, -1, -3,  1,  1,  3, -3,  1, -3, -1,  1,  1, -3, -3,  3,  1,  1, -3],   # u = 18
    [1,  3,  3,  1, -1, -3,  3, -1,  3,  3,  3, -3,  1, -1,  1, -1, -3, -1,  1,  3, -1,  3, -3, -3],   # u = 19
    [-1, -3,  3, -3, -3, -3, -1, -1, -3, -1, -3,  3,  1,  3, -3, -1,  3, -1,  1, -1,  3, -3,  1, -1],   # u = 20
    [-3, -3,  1,  1, -1,  1, -1,  1, -1,  3,  1, -3, -1,  1, -1,  1, -1, -1,  3,  3, -3, -1,  1, -3],   # u = 21
    [-3, -1, -3,  3,  1, -1, -3, -1, -3, -3,  3, -3,  3, -3, -1,  1,  3,  1, -3,  1,  3,  3, -1, -3],   # u = 22
    [-1, -1, -1, -1,  3,  3,  3,  1,  3,  3, -3,  1,  3, -1,  3, -1,  3,  3, -3,  3,  1, -1,  3,  3],   # u = 23
    [1, -1,  3,  3, -1, -3,  3, -3, -1, -1,  3, -1,  3, -1, -1,  1,  1,  1,  1, -1, -1, -3, -1,  3],   # u = 24
    [1, -1,  1, -1,  3, -1,  3,  1,  1, -1, -1, -3,  1,  1, -3,  1,  3, -3,  1,  1, -3, -3, -1, -1],   # u = 25
    [-3, -1,  1,  3,  1,  1, -3, -1, -1, -3,  3, -3,  3,  1, -3,  3, -3,  1, -1,  1, -3,  1,  1,  1],   # u = 26
    [-1, -3,  3,  3,  1,  1,  3, -1, -3, -1, -1, -1,  3,  1, -3, -3, -1,  3, -3, -1, -3, -1, -3, -1],   # u = 27
    [-1, -3, -1, -1,  1, -3, -1, -1,  1, -1, -3,  1,  1, -3,  1, -3, -3,  3,  1,  1, -1,  3, -1, -1],   # u = 28
    [1,  1, -1, -1, -3, -1,  3, -1,  3, -1,  1,  3,  1, -1,  3,  1,  3, -3, -3,  1, -1, -1,  1,  3]    # u = 29
    ]

    row = phi_table[u]
    return row


def extractPhiRow_T55121(u):


    phi_table = [
        [ 0, -1,  1,  3, -3,  3,  3,  1,  1,  3,  1, -3],  # u = 0
        [ 3,  1,  1,  1,  3,  3, -1,  1, -3, -3,  1, -3],  # u = 1
        [ 3,  2,  1,  1, -3, -3, -3, -1, -3, -3,  1, -3],  # u = 2
        [-1,  3,  1,  1,  1,  1, -1, -3, -3,  1, -3,  3],  # u = 3
        [-1,  3,  1, -1,  1, -1, -3, -1,  1, -1,  1,  3],  # u = 4
        [ 1, -3,  3, -1, -1,  1,  1, -1, -1,  3, -3,  1],  # u = 5
        [-1,  3, -3, -3, -3,  3,  1, -1,  3,  3, -3,  1],  # u = 6
        [-3, -1, -1, -1,  1, -3,  3, -1,  1, -3,  3,  1],  # u = 7
        [ 1, -3,  3,  1, -1, -1, -1,  1,  1,  3, -1,  1],  # u = 8
        [ 1, -3, -1,  3,  3, -1, -3,  1,  1,  1,  1,  1],  # u = 9
        [-1,  3, -1,  1,  1, -3, -3, -1, -3, -3,  3, -1],  # u = 10
        [ 3,  1, -1, -1,  3,  3, -3,  1,  3,  1,  3,  3],  # u = 11
        [ 1, -3,  1,  1, -3,  1,  1,  1, -3, -3, -3,  1],  # u = 12
        [ 3,  3, -3,  3, -3,  1,  1,  3, -1, -3,  3,  3],  # u = 13
        [-3,  1, -1, -3, -1,  3,  1,  3,  3,  3, -1,  1],  # u = 14
        [ 3, -1,  1, -3, -1, -1,  1,  1,  3,  1, -1, -3],  # u = 15
        [ 1,  3,  1, -1,  1,  3,  3,  3, -1, -1,  3, -1],  # u = 16
        [-3,  1,  1,  3, -3,  3, -3, -3,  3,  1,  3, -1],  # u = 17
        [-3,  3,  1,  1, -3,  1, -3, -3, -1, -1,  1, -3],  # u = 18
        [-1,  3,  1,  3,  1, -1, -1,  3, -3, -1, -3, -1],  # u = 19
        [-1, -3,  1,  1,  1,  1,  3,  1, -1,  1, -3, -1],  # u = 20
        [-1,  3, -1,  1, -3, -3, -3, -3, -3,  1, -1, -3],  # u = 21
        [ 1,  1, -3, -3, -3, -3, -1,  3, -3,  1, -3,  3],  # u = 22
        [ 1,  1, -1, -3, -1, -3,  1, -1,  1,  3, -1,  1],  # u = 23
        [ 1,  1,  3,  1,  3,  3, -1,  1, -1, -3, -3,  1],  # u = 24
        [ 1, -3,  3,  3,  1,  3,  3,  1, -3, -1, -1,  3],  # u = 25
        [ 1,  3, -3, -3,  3, -3,  1, -1, -1,  3, -1, -3],  # u = 26
        [-3, -1, -3, -1, -3,  3,  1, -1,  1,  3, -3, -3],  # u = 27
        [-1,  3, -3,  3, -1,  3,  3, -3,  3,  3, -1, -1],  # u = 28
        [ 3, -3, -3, -1, -1, -3, -1,  3, -3,  3,  1, -1]   # u = 29
        ]

    row = phi_table[u]
    return row

