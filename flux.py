import numpy as np
from numpy import linalg as la
from scipy import constants, sparse, linalg
import time
# import Sub180221 as Sub


# notations in comment
# -: virtual index;
# |: physical index;
# \: excitation index


class Flux:
    def __init__(self, E_Jb=7.5e9, C_Jb=40e-15, C_J=32.9e-15,
                 N_J=10, N_truncation=10, N_excited=5, N_Lanczos=6,
                 Ds=8, basis='flux'):
        """
        E_Jb, C_Jb, C_J: parameters of circuit elements, default values are set as ref.1
        N_J: number of junctions (sites of MPS)
        N_truncation: number of truncation when setting basis, the scale_ = N_truncation * 2 + 1
        N_excited: dimension of excitation index (calculate N_excited-1 excitations)
        N_Lanczos: maximal step of block Lanczos
            suggest N_Lanczos > 2
        Ds: maximal bond dimension (maximal dimension of virtual index)
            suggest Ds >= N_excited
        basis: 'charge' or 'flux', basis of MPO
            'flux' is more stable
        """
        self.usingLanczos = True
        self.phi_ext = 0
        self.E_J = self.E_Jb = E_Jb * 1e-9
        self.C_Jb = C_Jb
        self.C_J = C_J
        self.E_C = constants.e ** 2 / 2 / self.C_J / constants.h * 1e-9

        self.N_J = N_J
        self.N_truncation = N_truncation
        self.scale_ = N_truncation * 2 + 1
        self.Ds = Ds
        self.n_g = 0
        self.N_excited = N_excited
        self.N_Lanczos = N_Lanczos

        self.H_loc = None
        self.MPO = None
        phi_ext = 0
        self.L_bond = np.array([1, 0, 0.5 * self.E_Jb, 0.5 * self.E_Jb]).reshape((1, 4, 1))
        self.R_bond = np.array([0, 1, np.exp(1j * phi_ext), np.exp(-1j * phi_ext)]).reshape((1, 4, 1))
        self.HL = [self.L_bond for _ in range(N_J)]
        # self.HL = [None for _ in range(N_J)]
        self.HL[0] = self.L_bond
        self.HR = [self.R_bond for _ in range(N_J)]
        # self.HR = [None for _ in range(N_J)]
        # self.HR[-1] = self.R_bond
        self.T = []
        self.Eng = []
        self.init_MPS()
        self.set_MPO(basis)
        self.t = time.time()

    def init_MPS(self):
        self.T = []
        for i in range(self.N_J):
            Dl = min(self.scale_ ** i, self.scale_ ** (self.N_J - i), self.Ds)
            Dr = min(self.scale_ ** (i + 1), self.scale_ ** (self.N_J - 1 - i), self.Ds)
            if i:
                self.T.append(np.random.rand(Dl, self.scale_, Dr))
            else:
                self.T.append(np.random.rand(Dl, self.scale_, Dr, self.N_excited))

        '''def Mps_LQP(TL, UR):
            A = np.tensordot(TL, UR, (-1, 0))
            A = A.reshape((A.shape[0], -1))
            UL, TR = np.linalg.qr(A)
            TR = TR.reshape(TL.shape)
            return UL, TR'''
            
        def Mps_LQP(T,UR):  # TODO
            """ (0-T-L)(0-UR-1) -> (0-UL-1)(0-Tnew-L) """
            shapeT = np.asarray(np.shape(T))
            rankT = len(shapeT)
            
            A = np.tensordot(T,UR,(rankT-1,0))
            A = np.reshape(A,[shapeT[0],np.prod(shapeT[1:])])
            UL,Tnew = linalg.rq(A,mode = 'economic')
            Sign = np.diag(np.sign(np.diag(UL)))
            UL = np.dot(UL,Sign)
            Tnew = np.dot(Sign,Tnew)
            Tnew = np.reshape(Tnew,shapeT)
            return UL, Tnew

        U = np.eye(np.shape(self.T[-1])[-1])
        for i in range(self.N_J - 1, 0, -1):
            U, self.T[i] = Mps_LQP(self.T[i], U)

    def set_MPO(self, basis: str):
        if basis == 'charge':
            cos_theta = 0.5 * np.eye(self.scale_, k=1) + 0.5 * np.eye(self.scale_, k=-1)
            sin_theta = -0.5j * np.eye(self.scale_, k=1) + 0.5j * np.eye(self.scale_, k=-1)
            exp_i_theta = cos_theta + 1j * sin_theta
            n_operator = np.diag(np.arange(-self.N_truncation, self.N_truncation + 1))
            n_g = np.diag(np.array([0] * self.scale_))
            self.H_loc = 4 * self.E_C * (n_operator - n_g) ** 2 - self.E_J * cos_theta
        elif basis == 'flux':
            phi = np.linspace(-1 * np.pi, 1 * np.pi, self.scale_)
            d_phi = phi[-1] - phi[-2]
            cos_theta = np.diag(np.cos(phi))
            sin_theta = np.diag(np.sin(phi))
            exp_i_theta = cos_theta + 1j * sin_theta
            n_operator = 1j / 2 / d_phi * np.eye(self.scale_, k=1) - 1j / 2 / d_phi * np.eye(self.scale_, k=-1)
            n_g = self.n_g * np.eye(self.scale_, k=1) - self.n_g * np.eye(self.scale_, k=-1)
            # self.H_loc = (4 * self.E_C * (n_operator - n_g)**2
            self.H_loc = (4 * self.E_C * np.multiply(n_operator - n_g, n_operator - n_g)
                          - self.E_J * cos_theta)
        else:
            print('Not available basis')
            return None
        assert self.H_loc.shape == exp_i_theta.shape == (self.scale_, self.scale_)
        self.MPO = np.zeros((4, self.scale_, 4, self.scale_), dtype=complex)
        self.MPO[0, :, 0, :] = np.identity(self.scale_)
        self.MPO[0, :, 1, :] = self.H_loc
        self.MPO[1, :, 1, :] = np.identity(self.scale_)
        self.MPO[2, :, 2, :] = exp_i_theta
        self.MPO[3, :, 3, :] = exp_i_theta.conj()
        return self.MPO

    def set_phiExt(self, phi_ext: float):
        self.phi_ext = phi_ext
        self.R_bond = np.array([0, 1, np.exp(1j * phi_ext), np.exp(-1j * phi_ext)]).reshape((1, 4, 1))
        self.HR[-1] = self.R_bond
        for i in range(self.N_J - 1, 0, -1):
            # # print(i)
            '''self.HR[i - 1] = Sub.NCon([
                self.HR[i], self.T[i], self.MPO, np.conj(self.T[i])
            ], [
                [1, 3, 5], [-1, 2, 1], [-2, 2, 3, 4], [-3, 4, 5]
            ])'''
            self.HR[i - 1] = np.einsum(
                self.HR[i], [1, 3, 5],
                self.T[i], [11, 2, 1],
                self.MPO, [12, 2, 3, 4],
                np.conj(self.T[i]), [13, 4, 5],
                [11, 12, 13]
            )

    def optimizeT_site(self, site_: int):
        g = self.N_excited  # as noted in reference
        if self.usingLanczos:   # using block Lanczos
            HL = self.HL[site_]   # type: np.ndarray
            HR = self.HR[site_]   # type: np.ndarray
            T = self.T[site_]   # type: np.ndarray
            DT = np.shape(T)
            # H:
            # -1    -2   -3
            # HL 1 MPO 2 HR
            # -4    -5   -6
            '''H = Sub.NCon([HL, self.MPO, HR], [[-1, 1, -4], [1, -5, 2, -2], [-6, 2, -3]])'''
            H = np.einsum(HL, [11, 1, 14], self.MPO, [1, 15, 2, 12], HR, [16, 2, 13], [11, 12, 13, 14, 15, 16])
            
            Psi = [T for _ in range(self.N_Lanczos)]
            Alpha = [np.matrix(np.zeros((g, g)), dtype=complex)
                     for _ in range(self.N_Lanczos)]
            Beta = [np.matrix(np.zeros((g, g)), dtype=complex)
                    for _ in range(self.N_Lanczos)]
            M = np.zeros((self.N_Lanczos * g, self.N_Lanczos * g), dtype=complex)
            for n in range(self.N_Lanczos):
                # step A
                #     |         |     \     \
                # - Psi_n - = - U - ,  D  ,  V
                #      \         \      \     \
                U, D, Vh = la.svd(Psi[n].reshape((-1, g)), full_matrices=False)
                # step B
                #   |     \          |
                # - U - ,  V   =  - Psi_n -
                #    \      \         \
                Psi[n] = (U @ Vh).reshape(DT)

                # step C
                # HPsi = H|Psi_n>
                # [-           - ]
                # [      |       ]
                # [      H       ]
                # [      |       ]
                # [-   Psi[n]  - ]
                #         \
                HPsi = np.einsum(H, [11, 12, 13, 1, 2, 3], Psi[n], [1, 2, 3, 14], [11, 12, 13, 14])
                # Alpha_n = <Psi_n|H|Psi_n>
                Alpha[n] = np.matrix(np.einsum(HPsi, [1, 2, 3, 12],
                                               np.conj(Psi[n]), [1, 2, 3, 11],
                                               [11, 12]))
                M[n * g:g + n * g, n * g:g + n * g] = Alpha[n]

                # |Psi_{n+1}> = H|Psi_n> - Alpha_n|Psi_n> - Alpha_n^H|Psi_{n-1}>
                if n < self.N_Lanczos - 1:
                    # step D
                    Psi[n + 1] = HPsi - np.tensordot(Psi[n], Alpha[n], (-1, 0))
                    if n:
                        # Beta_n
                        Beta[n] = np.matrix(np.asmatrix(Vh).H @ np.diag(D) @ Vh)
                        M[n * g:g + n * g, n * g - g:n * g] = Beta[n]
                        M[n * g - g:n * g, n * g:g + n * g] = Beta[n].H

                        # step E
                        Psi[n + 1] -= np.tensordot(Psi[n - 1], Beta[n].H, (-1, 0))

            # M =
            # A0    B1^H    0      ...
            # B1    A1      B2^H   ...
            # 0 ...                      Bn^H
            #                      Bn     An

            Eig, V = sparse.linalg.eigsh(M, k=g, which='SA')
            V = np.reshape(V, (self.N_Lanczos, g, g))
            '''psi = Sub.NCon([V, np.array(Psi)], [[1, 2, -4], [1, -1, -2, -3, 2]])'''
            psi = np.einsum(V, [1, 2, 14], np.array(Psi), [1, 11, 12, 13, 2], [11, 12, 13, 14])
            return psi, Eig
        else:
            HL, HR, T = self.HL[site_], self.HR[site_], self.T[site_]
            A = np.einsum(HL, [11, 1, 14], self.MPO, [1, 15, 2, 12], HR, [16, 2, 13], [11, 12, 13, 14, 15, 16])
            '''A = Sub.NCon([HL, self.MPO, HR], [[-1, 1, -4], [1, -5, 2, -2], [-6, 2, -3]])'''
            # A:
            # -1    -2   -3
            # HL 1 MPO 2 HR
            # -4    -5   -6
            '''A = Sub.Group(A, [[0, 1, 2], [3, 4, 5]])'''
            As = A.shape
            A = A.reshape((As[0]*As[1]*As[2], As[3]*As[4]*As[5]))
            Eig, V = sparse.linalg.eigsh(A, k=self.N_excited, which='SA')
            return np.reshape(V, np.shape(T)), Eig

    def pt(self, n):
        print(n, time.time() - self.t)
        self.t = time.time()

    def SVD_T(self, Ti):
        # SVD decomposition of Ti:
        #    |1            |1
        # 0- Ti -2  ->  0- U - , - S - , - V -2
        #     \3                            \3
        T_ = Ti.shape
        U, S, V = la.svd(Ti.reshape((T_[0] * T_[1], T_[2] * T_[3])), full_matrices=False)
        U = U[:, :self.Ds].reshape((T_[0], T_[1], -1))
        S = np.diag(S[:self.Ds])
        V = V[:self.Ds, :].reshape((-1, T_[2], T_[3]))
        # if no truncation:
        # U, S, V = U.reshape((T_[0], T_[1], -1)), np.diag(S), V.reshape((-1, T_[2], T_[3]))
        return U, S, V

    def optimizeT(self, usingLanczos: bool, Eng_criterion: float, N_criterion: int, Quiet: bool):
        self.usingLanczos = usingLanczos
        # self.init_MPS()
        Eng0 = np.zeros((self.N_J, self.N_excited))
        Eng1 = np.zeros((self.N_J, self.N_excited))

        for r in range(1000):
            for i in range(self.N_J - 1):
                Ti, Eng1[i] = self.optimizeT_site(i)  # time cost most: optimizeT_site
                U, S, V = self.SVD_T(Ti)
                self.T[i] = U
                '''self.HL[i + 1] = Sub.NCon([
                    self.HL[i], np.conj(self.T[i]), self.MPO, self.T[i]
                ], [
                    [1, 3, 5], [1, 2, -1], [3, 4, -2, 2], [5, 4, -3]
                ])'''
                self.HL[i + 1] = np.einsum(
                    self.HL[i], [1, 3, 5],
                    self.T[i], [1, 2, 11],
                    self.MPO, [3, 4, 12, 2],
                    np.conj(self.T[i]), [5, 4, 13],
                    [11, 12, 13]
                )
                #                   |3              |3
                # 0- S - V -1 1- T[i+1] -4 -> 0- T[i+1] - 4
                #         \2                           \2
                self.T[i + 1] = np.einsum(np.tensordot(S, V, 1), [0, 1, 2],
                                          self.T[i + 1], [1, 3, 4],
                                          [0, 3, 4, 2])

            for i in range(self.N_J - 1, 0, -1):
                Ti, Eng1[i] = self.optimizeT_site(i)  # time cost most: optimizeT_site

                # SVD decomposition of Ti:
                #    |1            |1
                # 0- Ti -2  ->  2- U - , - S - , - V -0
                #     \3                            \3
                Ti = np.einsum('ijkl->kjil', Ti)
                U, S, V = self.SVD_T(Ti)
                self.T[i] = np.einsum('ijk->kji', U)
                '''self.HR[i - 1] = Sub.NCon([
                    self.HR[i], self.T[i], self.MPO, np.conj(self.T[i])
                ], [
                    [1, 3, 5], [-1, 2, 1], [-2, 2, 3, 4], [-3, 4, 5]
                ])'''
                self.HR[i - 1] = np.einsum(
                    self.HR[i], [1, 3, 5],
                    self.T[i], [11, 2, 1],
                    self.MPO, [12, 2, 3, 4],
                    np.conj(self.T[i]), [13, 4, 5],
                    [11, 12, 13]
                )
                #       |1                          |1
                # 0- T[i-1] -2 2- V - S -3 -> 0- T[i-1] - 3
                #                  \4              \4
                self.T[i - 1] = np.einsum(self.T[i - 1], [0, 1, 2],
                                          np.tensordot(S, V, 1), [3, 2, 4],
                                          [0, 1, 3, 4])

            if not Quiet:
                print('\r' + str(np.round(Eng1[1][:N_criterion] - Eng0[1][:N_criterion], 3)), end='\r')
            if (abs(Eng1[1][:N_criterion] - Eng0[1][:N_criterion]) < Eng_criterion).all():
                break
            Eng0 = Eng1.copy()

        self.Eng.append(Eng1[1] / self.N_J)

        return Eng1[1] / self.N_J
