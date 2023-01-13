import numpy as np


class Fluxonium:
    def __init__(self, EJ, EC, EL, phi_ext, N_truncation, basis, theta_begin, theta_end):
        self.EJ, self.EC, self.EL, self.phi_ext, self.N_truncation = EJ, EC, EL, phi_ext, N_truncation
        scale_ = N_truncation * 2 + 1

        if basis == 'charge':
            Nrange = 3
            qn = np.linspace(-Nrange, Nrange, scale_)
            dn = 1 / 2 / (qn[-1] - qn[-2])
            theta = -1j * dn * np.eye(scale_, k=1) + 1j * dn * np.eye(scale_, k=-1)
            n_operator = np.diag(qn)
            n_g = np.diag(np.array([0] * scale_))
            self.H = (4 * EC * (n_operator - n_g) ** 2
                      + 1 / 2 * EL * theta ** 2
                      - EJ * np.cos(theta - phi_ext))
        elif basis == 'flux':
            theta = np.linspace(theta_begin * np.pi, theta_end * np.pi, scale_)
            d_theta = theta[-1] - theta[-2]
            n_operator = -1j / 2 / d_theta * np.eye(scale_, k=1) + 1j / 2 / d_theta * np.eye(scale_, k=-1)
            n_g = 0 * np.eye(scale_, k=1) - 0 * np.eye(scale_, k=-1)
            self.H = (4 * EC * (n_operator - n_g) ** 2
                      + 1 / 2 * EL * np.diag(theta) ** 2
                      - EJ * np.diag(np.cos(theta + (phi_ext + np.pi/2))))

        else:
            print('Not available basis')
            return

    def eigenValues(self, N_excited):
        return np.linalg.eigvalsh(self.H)[:N_excited]
