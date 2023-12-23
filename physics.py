# Generate Grid Points with i = 59, j = 60 (total 3540 points)
import itertools
from globals import *

class GridGenerator:
    def __init__(self):
        self.grid_pts = None
        self.k = []
        self.h = []
    def grid_generation(self): 
        i_range = range(0, 59)
        j_range = range(0, 60)
        x = []
        y = []
        for i in i_range:
            if i <= 17:
                x.append(0.5 * (1 - (1.1 ** (17 - i))))
            elif i <= 37:
                x.append(0.05 * (i - 17))
            else:
                x.append(1.0 - (0.5 * (1 - (1.1 ** (i - 37)))))

        for j in j_range:
            if j <= 26:
                y.append(-0.15 + 0.5 * (1 - (1.1 ** (26 - j))))
            elif j <= 30:
                y.append(0.05 * (j - 29))
            elif j <= 33:
                y.append(0.05 * (j - 30))
            else:
                y.append(0.15 - 0.5 * (1.0 - (1.1 ** (j - 33))))

        for i in i_range:
            if 0 < i < 58:
                self.h.append((x[i + 1] - x[i - 1]) / 2)
            else:
                self.h.append(0)
        for j in j_range:
            if j == 29:
                self.k.append((y[j + 2] - y[j - 1]) / 2)
            elif 0 < j < 59:
                b = (y[j + 1] - y[j - 1]) / 2
                self.k.append(b)
            else:
                self.k.append(0)

        self.grid_pts = np.array([p for p in itertools.product(*[x, y])])
        return 

# Compute phis based on boundary conditions
def phi_calc(phi, pts, mach, alp, k):
    vsound = 330
    ww = 0.9
    vinf = vsound * mach

    for j in range(1, 59):
        for i in range(1, 58):
            if 17 <= i <= 37:
                if j == 28:
                    phi[j][i] = - (k[j] * vinf * np.sin(alp * np.pi / 180))
                    phi[j + 1][i] = phi[j][i] - (k[j] * vinf * np.sin(alp * np.pi / 180))
                    # print("loc1:", phi[j+1][i], phi[j][i], k[j], vinf, alp)
                elif j == 29:
                    phi[j][i] = - (2 * (k[j] * vinf * np.sin(alp * np.pi / 180)))
                    phi[j][i] = phi[j - 1][i] - (k[j] * vinf * np.sin(alp * np.pi / 180))
                    # print("loc2:", phi[j - 1][i], phi[j][i], k[j], vinf, alp)
                # elif j == 30:
                #     phi[j][i] = (2 * (k[j] * vinf * np.sin(alp * np.pi / 180)))
                #     phi[j][i] = phi[j + 1][i] + (k[j] * vinf * np.sin(alp * np.pi / 180))
                    # print("loc2:", phi[j - 1][i], phi[j][i], k[j], vinf, alp)
                elif j == 31:
                    phi[j][i] = (k[j] * vinf * np.sin(alp * np.pi / 180))
                    phi[j - 1][i] = phi[j][i] + (k[j] * vinf * np.sin(alp * np.pi / 180))
                else:
                    # phi[j][i] = phi[j][i]
                    phi_lower = - (k[j] * vinf * np.sin(alp * np.pi / 180))
                    phi_upper = (k[j] * vinf * np.sin(alp * np.pi / 180))
                    phi[j][i] = np.random.uniform(low=(phi_lower / 10000), high=(phi_upper / 10000))

            elif 38 <= i < 59:
                l = ((pts[j][58][0] - pts[j][i][0]) / (
                        pts[j][58][0] - pts[j][37][0]))  # flow leakage factor for wake points
                p = 1 - ww + ww * l  # Arya 26.03.2020
                if j == 28:
                    phi[j][i] = - (p * k[j] * vinf * np.sin(alp * np.pi / 180))
                    phi[j + 1][i] = phi[j][i] - (p * k[j] * vinf * np.sin(alp * np.pi / 180))
                    # print("loc3:", phi[j + 1][i], phi[j][i], k[j], vinf, alp)
                elif j == 28 and i == 58:
                    phi[j][i] = - (p * k[j] * vinf * np.sin(alp * np.pi / 180))
                    phi[j + 1][i] = phi[j][i] - (p * k[j] * vinf * np.sin(alp * np.pi / 180))
                elif j == 29:
                    phi[j][i] = - (2 * p * (k[j] * vinf * np.sin(alp * np.pi / 180)))
                    phi[j][i] = phi[j - 1][i] - (p * k[j] * vinf * np.sin(alp * np.pi / 180))
                elif j == 29 and i == 58:
                    phi[j][i] = - (2 * p * (k[j] * vinf * np.sin(alp * np.pi / 180)))
                    phi[j][i] = phi[j - 1][i] - (p * k[j] * vinf * np.sin(alp * np.pi / 180))
                    # print("loc4:", phi[j + 1][i], phi[j][i], k[j], vinf, alp)
                # elif j == 30:
                #     phi[j][i] = (2 * p * (k[j] * vinf * np.sin(alp * np.pi / 180)))
                #     phi[j][i] = phi[j + 1][i] + (p * k[j] * vinf * np.sin(alp * np.pi / 180))
                #     # print("loc5:", phi[j - 1][i], phi[j][i], k[j], vinf, alp)
                # elif j == 30 and i == 58:
                #     phi[j][i] = (2 * p * (k[j] * vinf * np.sin(alp * np.pi / 180)))
                #     phi[j][i] = phi[j + 1][i] + (p * k[j] * vinf * np.sin(alp * np.pi / 180))
                    # print("loc6:", phi[j - 1][i], phi[j][i], k[j], vinf, alp)
                elif j == 31:
                    phi[j][i] = (p * k[j] * vinf * np.sin(alp * np.pi / 180))
                    phi[j - 1][i] = phi[j][i] + (p * k[j] * vinf * np.sin(alp * np.pi / 180))
                    # print("loc5:", phi[j - 1][i], phi[j][i], k[j], vinf, alp)
                elif j == 31 and i == 58:
                    phi[j][i] = (p * k[j] * vinf * np.sin(alp * np.pi / 180))
                    phi[j - 1][i] = phi[j][i] + (p * k[j] * vinf * np.sin(alp * np.pi / 180))
                    # print("loc6:", phi[j - 1][i], phi[j][i], k[j], vinf, alp)
                else:
                    # phi[j][i] = phi[j][i]
                    phi_lower = - (k[j] * vinf * np.sin(alp * np.pi / 180))
                    phi_upper = (k[j] * vinf * np.sin(alp * np.pi / 180))
                    phi[j][i] = np.random.uniform(low=(phi_lower / 10000), high=(phi_upper / 10000))
            else:
                # phi[j][i] = phi[j][i]
                phi_lower = - (k[j] * vinf * np.sin(alp * np.pi / 180))
                phi_upper = (k[j] * vinf * np.sin(alp * np.pi / 180))
                phi[j][i] = np.random.uniform(low=(phi_lower / 10000), high=(phi_upper / 10000))

    return phi

# Compute residuals from phis
def calculation(phi, pts, mach, alp, h, k):  
    vsound = 330  # speed of sound
    rho = np.ones(
        shape=(60, 59))  # rho should be 60,59 because the first index is for J which ranges till 60.
    res = np.zeros(shape=(60, 59))
    vinf = vsound * mach  # freestream velocity
    ww = 0.9  # Arya 26.03.2020

    I = range(1, 58)
    J = range(1, 59)

    for j in J:
        for i in I:

            if i == 1 and (j < 28 or j > 31):
                if j == 1:  # For point A
                    dphix = (phi[j][i + 1] - phi[j][i]) / (2 * h[i])  # Ann/a
                    dphiy = (phi[j + 1][i] - phi[j][i]) / (2 * k[j])  # Ann/b
                    vxx = vinf * np.cos(alp * np.pi / 180) + dphix  # Ann/c
                    vyy = vinf * np.sin(alp * np.pi / 180) + dphiy  # Ann/d
                    vtot = vxx * vxx + vyy * vyy  # Ann/e
                    rho[j][i] = (1 + 0.2 * mach * mach * (1 - vtot / vinf ** 2)) ** 2.5  # Ann/f
                    rhox = (rho[j][i + 1] - rho[j][i]) / (2 * h[i])  # Ann/g
                    rhoy = (rho[j + 1][i] - rho[j][i]) / (2 * k[j])  # Ann/h
                    rr1 = (rhox * vxx + rhoy * vyy) / (rho[j][i])
                    ra = (phi[j][i + 1] - phi[j][i]) / (h[i] * h[i]) + (phi[j + 1][i] - phi[j][i]) / (k[j] * k[j]) + rr1
                    res[j][i] = ra
                    # if ra > 0:
                    #     print('ra;', res[j][i], j, i, mach, alp)

                elif j == 58:  # for point B
                    dphix = (phi[j][i + 1] - phi[j][i]) / (2 * h[i])  # Bnn/a
                    dphiy = (phi[j][i] - phi[j - 1][i]) / (2 * k[j])  # Bnn/b
                    vxx = vinf * np.cos(alp * np.pi / 180) + dphix  # Bnn/c
                    vyy = vinf * np.sin(alp * np.pi / 180) + dphiy  # Bnn/d
                    vtot = vxx * vxx + vyy * vyy  # Bnn/e
                    rho[j][i] = (1 + 0.2 * mach * mach * (1 - vtot / vinf ** 2)) ** 2.5  # Ann/f
                    rhox = (rho[j][i + 1] - rho[j][i]) / (2 * h[i])  # Bnn/g
                    rhoy = (rho[j][i] - rho[j - 1][i]) / (2 * k[j])  # Bnn/h
                    rr1 = (rhox * vxx + rhoy * vyy) / (rho[j][i])
                    rb = (phi[j][i + 1] - phi[j][i]) / (h[i] * h[i]) + (phi[j - 1][i] - phi[j][i]) / (k[j] * k[j]) + rr1
                    res[j][i] = rb
                    # if rb > 0.01:
                    #     print('rb;', res[j][i], j, i, mach, alp)

                elif (1 < j < 28) or (31 < j < 58):  # for E

                    dphix = (phi[j][i + 1] - phi[j][i]) / (2 * h[i])  # Enna
                    dphiy = (phi[j + 1][i] - phi[j - 1][i]) / (2 * k[j])  # Nnb
                    vxx = vinf * np.cos(alp * np.pi / 180) + dphix
                    vyy = vinf * np.sin(alp * np.pi / 180) + dphiy
                    vtot = vxx * vxx + vyy * vyy
                    rho[j][i] = (1 + 0.2 * mach * mach * (1 - vtot / vinf ** 2)) ** 2.5  # Ann/f
                    rhox = (rho[j][i + 1] - rho[j][i]) / (2 * h[i])  # Enng
                    rhoy = (rho[j + 1][i] - rho[j - 1][i]) / (2 * k[j])  # nnh
                    rr1 = (rhox * vxx + rhoy * vyy) / (rho[j][i])
                    re = (phi[j][i + 1] - phi[j][i]) / (h[i] * h[i]) + (
                            phi[j + 1][i] - 2 * phi[j][i] + phi[j - 1][i]) / (k[j] * k[j]) + rr1
                    res[j][i] = re
                    # if re > 0.01:
                    #     print('re;', res[j][i], j, i, mach, alp)

            elif i == 57 and (j < 28 or j > 31):
                if j == 1:  # for point D all exactly like Dnn/equations
                    dphix = (phi[j][i] - phi[j][i - 1]) / (2 * h[i])
                    dphiy = (phi[j + 1][i] - phi[j][i]) / (2 * k[j])
                    vxx = vinf * np.cos(alp * np.pi / 180) + dphix
                    vyy = vinf * np.sin(alp * np.pi / 180) + dphiy
                    vtot = vxx * vxx + vyy * vyy
                    rho[j][i] = (1 + 0.2 * mach * mach * (1 - vtot / vinf ** 2)) ** 2.5  # Ann/f
                    rhox = (rho[j][i] - rho[j][i - 1]) / (2 * h[i])
                    rhoy = (rho[j + 1][i] - rho[j][i]) / (2 * k[j])
                    rr1 = (rhox * vxx + rhoy * vyy) / (rho[j][i])
                    rd = (phi[j][i - 1] - phi[j][i]) / (h[i] * h[i]) + (phi[j + 1][i] - phi[j][i]) / (k[j] * k[j]) + rr1
                    res[j][i] = rd
                    # if rd > 0.01:
                    #     print('rd;', res[j][i], j, i, mach, alp)

                elif j == 58:  # for point C all exact like Cnn/equations
                    dphix = (phi[j][i] - phi[j][i - 1]) / (2 * h[i])
                    dphiy = (phi[j][i] - phi[j - 1][i]) / (2 * k[j])
                    vxx = vinf * np.cos(alp * np.pi / 180) + dphix
                    vyy = vinf * np.sin(alp * np.pi / 180) + dphiy
                    vtot = vxx * vxx + vyy * vyy
                    rho[j][i] = (1 + 0.2 * mach * mach * (1 - vtot / vinf ** 2)) ** 2.5
                    rhox = (rho[j][i] - rho[j][i - 1]) / (2 * h[i])
                    rhoy = (rho[j][i] - rho[j - 1][i]) / (2 * k[j])
                    rr1 = (rhox * vxx + rhoy * vyy) / (rho[j][i])
                    rc = (phi[j][i - 1] - phi[j][i]) / (h[i] * h[i]) + (phi[j - 1][i] - phi[j][i]) / (k[j] * k[j]) + rr1
                    res[j][i] = rc
                    # if rc > 0.01:
                    #     print('rc;', res[j][i], j, i, mach, alp)

                elif (1 < j < 28) or (31 < j < 58):  # for G

                    dphix = (phi[j][i] - phi[j][i - 1]) / (2 * h[i])  # Gnn/a
                    dphiy = (phi[j + 1][i] - phi[j - 1][i]) / (2 * k[j])  # Nnb
                    vxx = vinf * np.cos(alp * np.pi / 180) + dphix
                    vyy = vinf * np.sin(alp * np.pi / 180) + dphiy
                    vtot = vxx * vxx + vyy * vyy
                    rho[j][i] = (1 + 0.2 * mach * mach * (1 - vtot / vinf ** 2)) ** 2.5
                    rhox = (rho[j][i] - rho[j][i - 1]) / (2 * h[i])  # Gnng
                    rhoy = (rho[j + 1][i] - rho[j - 1][i]) / (2 * k[j])  # nnh
                    rr1 = (rhox * vxx + rhoy * vyy) / (rho[j][i])
                    rg = (phi[j][i - 1] - phi[j][i]) / (h[i] * h[i]) + (
                            phi[j + 1][i] - 2 * phi[j][i] + phi[j - 1][i]) / (k[j] * k[j]) + rr1
                    res[j][i] = rg
                    # if rg > 0.01:
                    #     print('rg;', res[j][i], j, i, mach, alp)

            elif j == 1 and 1 < i < 57:  # for H all same as equations listed in Hnn series

                dphix = (phi[j][i + 1] - phi[j][i - 1]) / (2 * h[i])
                dphiy = (phi[j + 1][i] - phi[j][i]) / (2 * k[j])
                vxx = vinf * np.cos(alp * np.pi / 180) + dphix
                vyy = vinf * np.sin(alp * np.pi / 180) + dphiy
                vtot = vxx * vxx + vyy * vyy
                rho[j][i] = (1 + 0.2 * mach * mach * (1 - vtot / vinf ** 2)) ** 2.5
                rhox = (rho[j][i + 1] - rho[j][i - 1]) / (2 * h[i])
                rhoy = (rho[j + 1][i] - rho[j][i]) / (2 * k[j])
                rr1 = (rhox * vxx + rhoy * vyy) / (rho[j][i])
                rh = (phi[j][i + 1] - 2 * phi[j][i] + phi[j][i - 1]) / (h[i] * h[i]) + (phi[j + 1][i] + phi[j][i]) / (
                        k[j] * k[j]) + rr1
                res[j][i] = rh
                # if rh > 0.01:
                #     print('rh;', res[j][i], j, i, mach, alp)

            elif j == 58 and 1 < i < 57:  # for F
                dphix = (phi[j][i + 1] - phi[j][i - 1]) / (2 * h[i])  # NN1a
                dphiy = (phi[j][i] - phi[j - 1][i]) / (2 * k[j])  # FNNb
                vxx = vinf * np.cos(alp * np.pi / 180) + dphix
                vyy = vinf * np.sin(alp * np.pi / 180) + dphiy
                vtot = vxx * vxx + vyy * vyy
                rho[j][i] = (1 + 0.2 * mach * mach * (1 - vtot / vinf ** 2)) ** 2.5
                rhox = (rho[j][i + 1] - rho[j][i - 1]) / (2 * h[i])  # NNg
                rhoy = (rho[j][i] - rho[j - 1][i]) / (2 * k[j])  # FNNh
                rr1 = (rhox * vxx + rhoy * vyy) / (rho[j][i])
                rf = (phi[j][i + 1] - 2 * phi[j][i] + phi[j][i - 1]) / (h[i] * h[i]) + (phi[j - 1][i] - phi[j][i]) / (
                        k[j] * k[j]) + rr1
                res[j][i] = rf
                # if rf > 0.01:
                #     print('rf;', res[j][i], j, i, mach, alp)

            # points near the plate
            elif j == 31 and 17 <= i <= 37:  # I' (point just above the plate)
                dphix = (phi[j][i + 1] - phi[j][i - 1]) / (2 * h[i])  # NN1a
                dphiy = (phi[j + 1][i] - phi[j][i] - (k[j] * vinf * np.sin(alp * np.pi / 180))) / (
                        2 * k[j])  # Innb - original
                # dphiy = (phi[j + 1][i] - phi[j - 1][i]) / (2 * k[j])  # Innb
                vxx = (vinf * np.cos(alp * np.pi / 180)) + dphix
                vyy = (vinf * np.sin(alp * np.pi / 180)) + dphiy
                vtot = vxx * vxx + vyy * vyy
                rho[j][i] = (1 + (0.2 * mach * mach * (1 - vtot / (vinf ** 2)))) ** 2.5
                rhox = (rho[j][i + 1] - rho[j][i - 1]) / (2 * h[i])  # NNg
                rhoy = (rho[j + 1][i] - rho[j - 1][i]) / (2 * k[j])  # INNh
                rr1 = ((rhox * vxx) + (rhoy * vyy)) / (rho[j][i])
                # if abs(rr1) > 0.7:
                #     print('rr1;', rr1, j, i, mach, alp)
                ri_p2 = ((phi[j][i + 1] - (2 * phi[j][i]) + phi[j][i - 1]) / (h[i] * h[i])) + (
                        (phi[j + 1][i] - phi[j][i] + (k[j] * vinf * np.sin(alp * np.pi / 180))) / (
                        k[j] * k[j])) + rr1  # original
                # ri_p2 = ((phi[j][i + 1] - (2 * phi[j][i]) + phi[j][i - 1]) / (h[i] * h[i])) + (
                #         (phi[j + 1][i] - (2 * phi[j][i]) + phi[j - 1][i]) / (k[j] * k[j])) + rr1
                res[j][i] = ri_p2

            elif j == 30 and 17 <= i <= 37:  # residual for point I (point on the upper surface of plate)
                ri_p1 = (phi[j][i] - phi[j + 1][i] - k[j] * vinf * np.sin(alp * np.pi / 180))  # original
                # ri_p1 = 0     # forced to zero
                res[j][i] = ri_p1

            elif j == 28 and 17 <= i <= 37:  # for J' (point just below the plate)
                dphix = (phi[j][i + 1] - phi[j][i - 1]) / (2 * h[i])  # NN1a
                dphiy = (phi[j][i] - phi[j - 1][i] - k[j] * vinf * np.sin(alp * np.pi / 180)) / (
                        2 * k[j])  # JNNb - original
                # dphiy = (phi[j + 1][i] - phi[j - 1][i]) / (2 * k[j])  # JNNb
                vxx = vinf * np.cos(alp * np.pi / 180) + dphix
                vyy = vinf * np.sin(alp * np.pi / 180) + dphiy
                vtot = vxx * vxx + vyy * vyy
                rho[j][i] = (1 + 0.2 * mach * mach * (1 - vtot / vinf ** 2)) ** 2.5
                rhox = (rho[j][i + 1] - rho[j][i - 1]) / (2 * h[i])  # NNg
                rhoy = (rho[j + 1][i] - rho[j - 1][i]) / (2 * k[j])  # JNNh
                rr1 = (rhox * vxx + rhoy * vyy) / (rho[j][i])
                rj_p2 = ((phi[j][i + 1] - 2 * phi[j][i] + phi[j][i - 1]) / (h[i] * h[i])) + (
                        (phi[j - 1][i] - phi[j][i] - k[j] * vinf * np.sin(alp * np.pi / 180)) / (k[j] * k[
                    j])) + rr1  # original
                # rj_p2 = ((phi[j][i + 1] - (2 * phi[j][i]) + phi[j][i - 1]) / (h[i] * h[i])) + (
                #         (phi[j - 1][i] - (2 * phi[j][i]) + phi[j + 1][i]) / (k[j] * k[j])) + rr1
                res[j][i] = rj_p2

            elif j == 29 and 17 <= i <= 37:  # residual for point J (point on the upper surface of plate)
                rj_p1 = (phi[j][i] - phi[j - 1][i] + k[j] * vinf * np.sin(alp * np.pi / 180))  # original
                # rj_p1 = 0     # forced to zero
                res[j][i] = rj_p1

            # For points in the wake
            elif j == 31 and 37 < i < 58:  # For I1' (point just above the upper wake)

                if i < 57:  # For I1' other than the point in G
                    # p=1-w+w*l[i-38]
                    # p = 1 # Arya ; forced to 1
                    l = ((pts[j][58][0] - pts[j][i][0]) / (
                            pts[j][58][0] - pts[j][37][0]))  # flow leakage factor for wake points
                    p = 1 - ww + ww * l  # Arya 26.03.2020
                    dphix = (phi[j][i + 1] - phi[j][i - 1]) / (2 * h[i])  # NN1a
                    dphiy = (phi[j + 1][i] - phi[j][i] - vinf * p * k[j] * np.sin(alp * np.pi / 180)) / (
                            2 * k[j])  # I1NNb - original
                    # dphiy = (phi[j + 1][i] - phi[j - 1][i]) / (2 * k[j])  # I1NNb
                    vxx = vinf * np.cos(alp * np.pi / 180) + dphix
                    vyy = vinf * np.sin(alp * np.pi / 180) + dphiy
                    vtot = vxx * vxx + vyy * vyy
                    rho[j][i] = (1 + 0.2 * mach * mach * (1 - vtot / vinf ** 2)) ** 2.5
                    rhox = (rho[j][i + 1] - rho[j][i - 1]) / (2 * h[i])  # NNg
                    rhoy = (rho[j + 1][i] - rho[j - 1][i]) / (2 * k[j])  # I1NNh
                    rr1 = (rhox * vxx + rhoy * vyy) / (rho[j][i])
                    ri_w2 = ((phi[j][i + 1] - 2 * phi[j][i] + phi[j][i - 1]) / (h[i] * h[i])) + (
                            (phi[j + 1][i] - phi[j][i] + p * k[j] * vinf * np.sin(alp * np.pi / 180)) / (
                            k[j] * k[j])) + rr1  # original
                    # ri_w2 = ((phi[j][i + 1] - (2 * phi[j][i]) + phi[j][i - 1]) / (h[i] * h[i])) + (
                    #         (phi[j + 1][i] - (2 * phi[j][i]) + phi[j - 1][i]) / (k[j] * k[j])) + rr1
                    res[j][i] = ri_w2

                elif i == 57:  # when upper wake points are at farfield boundary (like point G)

                    l = ((pts[j][58][0] - pts[j][i][0]) / (
                            pts[j][58][0] - pts[j][37][0]))  # flow leakage factor for wake points
                    # p=1-w+w*l[i-38]
                    # p = 1 # Arya; forced
                    p = 1 - ww + ww * l  # Arya 26.03.2020
                    dphix = (phi[j][i] - phi[j][i - 1]) / (2 * h[i])  # Gnna
                    dphiy = (phi[j + 1][i] - phi[j][i] - vinf * p * k[j] * np.sin(alp * np.pi / 180)) / (
                            2 * k[j])  # same as I1NNb - original
                    # dphiy = (phi[j + 1][i] - phi[j - 1][i]) / (2 * k[j])  # same as I1NNb
                    vxx = vinf * np.cos(alp * np.pi / 180) + dphix
                    vyy = vinf * np.sin(alp * np.pi / 180) + dphiy
                    vtot = vxx * vxx + vyy * vyy
                    rho[j][i] = (1 + 0.2 * mach * mach * (1 - vtot / vinf ** 2)) ** 2.5
                    rhox = (rho[j][i] - rho[j][i - 1]) / (2 * h[i])  # Gnng
                    rhoy = (rho[j + 1][i] - rho[j - 1][i]) / (2 * k[j])  # I1NNh
                    rr1 = (rhox * vxx + rhoy * vyy) / (rho[j][i])
                    ri_w2_g = ((phi[j][i - 1] - phi[j][i]) / (h[i] * h[i])) + (
                            (phi[j + 1][i] - phi[j][i] + p * k[j] * vinf * np.sin(alp * np.pi / 180)) / (
                            k[j] * k[j])) + rr1  # original
                    # ri_w2_g = ((phi[j][i - 1] - phi[j][i]) / (h[i] * h[i])) + (
                    #         (phi[j + 1][i] - (2 * phi[j][i]) + phi[j - 1][i]) / (k[j] * k[j])) + rr1
                    res[j][i] = ri_w2_g

            elif j == 30 and 37 < i < 58:
                if i < 57:  # residual for point I1 (point on the upper wake) - other than point like G
                    l = ((pts[j][58][0] - pts[j][i][0]) / (
                            pts[j][58][0] - pts[j][37][0]))  # flow leakage factor for wake points
                    p = 1 - ww + ww * l  # Arya 26.03.2020
                    ri_w1 = (phi[j][i] - phi[j + 1][i] - p * k[j] * vinf * np.sin(alp * np.pi / 180))  # original
                    # ri_w1 = 0     # forced to zero
                    res[j][i] = ri_w1

                elif i == 57:  # SIMILAR FOR I1 POINT
                    l = ((pts[j][58][0] - pts[j][i][0]) / (
                            pts[j][58][0] - pts[j][37][0]))  # flow leakage factor for wake points
                    p = 1 - ww + ww * l  # Arya 26.03.2020
                    ri_w1_g = (phi[j][i] - phi[j + 1][i] - p * k[j] * vinf * np.sin(alp * np.pi / 180))  # original
                    # ri_w1_g = 0       # forced to zero
                    res[j][i] = ri_w1_g

            elif j == 28 and 37 < i < 58:  # For J1' (point just below the lower wake)

                if i < 57:  # For J1' other than the point in G
                    # p=1-w+w*l[i-38]
                    # p = 1 # Arya ; forced
                    l = ((pts[j][58][0] - pts[j][i][0]) / (
                            pts[j][58][0] - pts[j][37][0]))  # flow leakage factor for wake points
                    p = 1 - ww + ww * l  # Arya 26.03.2020
                    dphix = (phi[j][i + 1] - phi[j][i - 1]) / (2 * h[i])  # NN1a
                    dphiy = (phi[j][i] - phi[j - 1][i] - vinf * p * k[j] * np.sin(alp * np.pi / 180)) / (
                            2 * k[j])  # J1NNb - original
                    # dphiy = (phi[j + 1][i] - phi[j - 1][i]) / (2 * k[j])  # I1NNb
                    vxx = vinf * np.cos(alp * np.pi / 180) + dphix
                    vyy = vinf * np.sin(alp * np.pi / 180) + dphiy
                    vtot = vxx * vxx + vyy * vyy
                    rho[j][i] = (1 + 0.2 * mach * mach * (1 - vtot / vinf ** 2)) ** 2.5
                    rhox = (rho[j][i + 1] - rho[j][i - 1]) / (2 * h[i])  # NNg
                    rhoy = (rho[j + 1][i] - rho[j - 1][i]) / (2 * k[j])  # J1NNh
                    rr1 = (rhox * vxx + rhoy * vyy) / (rho[j][i])
                    rj_w2 = ((phi[j][i + 1] - 2 * phi[j][i] + phi[j][i - 1]) / (h[i] * h[i])) + (
                            (phi[j - 1][i] - phi[j][i] - p * k[j] * vinf * np.sin(alp * np.pi / 180)) / (
                            k[j] * k[j])) + rr1  # original
                    # rj_w2 = ((phi[j][i + 1] - (2 * phi[j][i]) + phi[j][i - 1]) / (h[i] * h[i])) + (
                    #         (phi[j - 1][i] - (2 * phi[j][i]) + phi[j + 1][i]) / (k[j] * k[j])) + rr1
                    res[j][i] = rj_w2

                elif i == 57:  # when lower wake points are at farfield boundary (like point G)

                    l = ((pts[j][58][0] - pts[j][i][0]) / (
                            pts[j][58][0] - pts[j][37][0]))  # flow leakage factor for wake points
                    # p=1-w+w*l[i-38]
                    # p = 1 # Arya ; forced
                    p = 1 - ww + ww * l  # Arya 26.03.2020
                    dphix = (phi[j][i] - phi[j][i - 1]) / (2 * h[i])  # Gnna
                    dphiy = (phi[j][i] - phi[j - 1][i] - vinf * p * k[j] * np.sin(alp * np.pi / 180)) / (
                            2 * k[j])  # J1NNb - original
                    # dphiy = (phi[j + 1][i] - phi[j - 1][i]) / (2 * k[j])  # I1NNb
                    vxx = vinf * np.cos(alp * np.pi / 180) + dphix
                    vyy = vinf * np.sin(alp * np.pi / 180) + dphiy
                    vtot = vxx * vxx + vyy * vyy
                    rho[j][i] = (1 + 0.2 * mach * mach * (1 - vtot / vinf ** 2)) ** 2.5
                    rhox = (rho[j][i] - rho[j][i - 1]) / (2 * h[i])  # Gnng
                    rhoy = (rho[j + 1][i] - rho[j - 1][i]) / (2 * k[j])  # J1NNh
                    rr1 = (rhox * vxx + rhoy * vyy) / (rho[j][i])
                    rj_w2_g = ((phi[j][i - 1] - phi[j][i]) / (h[i] * h[i])) + (
                            (phi[j - 1][i] - phi[j][i] - p * k[j] * vinf * np.sin(alp * np.pi / 180)) / (
                            k[j] * k[j])) + rr1  # original
                    # rj_w2_g = ((phi[j][i - 1] - phi[j][i]) / (h[i] * h[i])) + (
                    #         (phi[j - 1][i] - (2 * phi[j][i]) + phi[j + 1][i]) / (k[j] * k[j])) + rr1
                    res[j][i] = rj_w2_g

            elif j == 29 and 37 < i < 58:
                if i < 57:  # residual at J1 (point on the lower wake)
                    l = ((pts[j][58][0] - pts[j][i][0]) / (
                            pts[j][58][0] - pts[j][37][0]))  # flow leakage factor for wake points
                    p = 1 - ww + ww * l  # Arya 26.03.2020
                    rj_w1 = (phi[j][i] - phi[j - 1][i] + p * k[j] * vinf * np.sin(alp * np.pi / 180))  # original
                    # rj_w1 = 0     # forced to zero
                    res[j][i] = rj_w1

                elif i == 57:  # residual at J1 for point like G
                    l = ((pts[j][58][0] - pts[j][i][0]) / (
                            pts[j][58][0] - pts[j][37][0]))  # flow leakage factor for wake points
                    p = 1 - ww + ww * l  # Arya 26.03.2020
                    rj_w1_g = (phi[j][i] - phi[j - 1][i] + p * k[j] * vinf * np.sin(alp * np.pi / 180))
                    # rj_w1_g = 0
                    res[j][i] = rj_w1_g

            # For upstream points
            elif 28 < j < 31 and 0 < i < 17:  # For J2, I2
                if i > 1 and j == 29:  # For J2 not next to the border
                    dphix = (phi[j][i + 1] - phi[j][i - 1]) / (2 * h[i])  # NN1a
                    dphiy = (phi[j + 2][i] - phi[j - 1][i]) / (2 * k[j])  # J2NNb
                    vxx = vinf * np.cos(alp * np.pi / 180) + dphix
                    vyy = vinf * np.sin(alp * np.pi / 180) + dphiy
                    vtot = vxx * vxx + vyy * vyy
                    rho[j][i] = (1 + 0.2 * mach * mach * (1 - vtot / vinf ** 2)) ** 2.5
                    rhox = (rho[j][i + 1] - rho[j][i - 1]) / (2 * h[i])  # NNg
                    rhoy = (rho[j + 2][i] - rho[j - 1][i]) / (2 * k[j])  # J2NNh
                    rr1 = (rhox * vxx + rhoy * vyy) / (rho[j][i])
                    rj_up = ((phi[j][i + 1] - 2 * phi[j][i] + phi[j][i - 1]) / (h[i] * h[i])) + (
                            (phi[j + 2][i] - 2 * phi[j][i] + phi[j - 1][i]) / (k[j] * k[j])) + rr1
                    res[j][i] = rj_up

                elif i == 1 and j == 29:  # when upstream points on lower side of plate are at farfield boundary (like point E)
                    dphix = (phi[j][i + 1] - phi[j][i]) / (2 * h[i])  # Enna
                    dphiy = (phi[j + 2][i] - phi[j - 1][i]) / (2 * k[j])  # J2NNb
                    vxx = vinf * np.cos(alp * np.pi / 180) + dphix
                    vyy = vinf * np.sin(alp * np.pi / 180) + dphiy
                    vtot = vxx * vxx + vyy * vyy
                    rho[j][i] = (1 + 0.2 * mach * mach * (1 - vtot / vinf ** 2)) ** 2.5
                    rhox = (rho[j][i + 1] - rho[j][i]) / (2 * h[i])  # Enng
                    rhoy = (rho[j + 2][i] - rho[j - 1][i]) / (2 * k[j])  # J2NNh
                    rr1 = (rhox * vxx + rhoy * vyy) / (rho[j][i])
                    rj_up_e = (phi[j][i + 1] - phi[j][i]) / (h[i] * h[i]) + (
                            phi[j + 2][i] - 2 * phi[j][i] + phi[j - 1][i]) / (k[j] * k[j]) + rr1
                    res[j][i] = rj_up_e

                elif i > 1 and j == 30:  # For I2 not next to the border
                    dphix = (phi[j][i + 1] - phi[j][i - 1]) / (2 * h[i])  # NN1a
                    dphiy = (phi[j + 1][i] - phi[j - 2][i]) / (2 * k[j])  # I2NNb
                    vxx = vinf * np.cos(alp * np.pi / 180) + dphix
                    vyy = vinf * np.sin(alp * np.pi / 180) + dphiy
                    vtot = vxx * vxx + vyy * vyy
                    rho[j][i] = (1 + 0.2 * mach * mach * (1 - vtot / vinf ** 2)) ** 2.5
                    rhox = (rho[j][i + 1] - rho[j][i - 1]) / (2 * h[i])  # NNg
                    rhoy = (rho[j + 1][i] - rho[j - 2][i]) / (2 * k[j])  # I2NNh
                    rr1 = (rhox * vxx + rhoy * vyy) / (rho[j][i])
                    ri_up = ((phi[j][i + 1] - 2 * phi[j][i] + phi[j][i - 1]) / (h[i] * h[i])) + (
                            (phi[j + 1][i] - 2 * phi[j][i] + phi[j - 2][i]) / (k[j] * k[j])) + rr1
                    res[j][i] = ri_up

                elif i == 1 and j == 30:    # when upstream points on upper side of plate are at farfield boundary (like point E)
                    dphix = (phi[j][i + 1] - phi[j][i]) / (2 * h[i])  # Enna
                    dphiy = (phi[j + 1][i] - phi[j - 2][i]) / (2 * k[j])  # I2NNb
                    vxx = vinf * np.cos(alp * np.pi / 180) + dphix
                    vyy = vinf * np.sin(alp * np.pi / 180) + dphiy
                    vtot = vxx * vxx + vyy * vyy
                    rho[j][i] = (1 + 0.2 * mach * mach * (1 - vtot / vinf ** 2)) ** 2.5
                    rhox = (rho[j][i + 1] - rho[j][i]) / (2 * h[i])  # Enng
                    rhoy = (rho[j + 1][i] - rho[j - 2][i]) / (2 * k[j])  # I2NNh
                    rr1 = (rhox * vxx + rhoy * vyy) / (rho[j][i])
                    ri_up_e = ((phi[j][i + 1] - phi[j][i]) / (h[i] * h[i])) + (
                            (phi[j + 1][i] - 2 * phi[j][i] + phi[j - 2][i]) / (k[j] * k[j])) + rr1
                    res[j][i] = ri_up_e

            # For interior points

            elif i == 0 or i == 58 or j == 0 or j == 59 or j == 29 or j == 30:
                pass  # TAKEN CARE IN BOUNDARY CONDS

            else:

                # INTERIOR POINTS
                # STANDARD FORMULAE of NN equations

                dphix = (phi[j][i + 1] - phi[j][i - 1]) / (2 * h[i])
                # print('dhix;', dphix, j, i, mach, alp)
                dphiy = (phi[j + 1][i] - phi[j - 1][i]) / (2 * k[j])
                # print('dphiy;', dphiy, j, i, mach, alp)
                vxx = vinf * np.cos(alp * np.pi / 180) + dphix
                # print('vxx;', vxx, j, i, mach, alp)
                vyy = vinf * np.sin(alp * np.pi / 180) + dphiy
                # print('vyy;', vyy, j, i, mach, alp)
                vtot = vxx * vxx + vyy * vyy
                # print('vtot;', vtot, j, i, mach, alp)
                rho[j][i] = (1 + 0.2 * mach * mach * (1 - vtot / vinf ** 2)) ** 2.5
                # print('rho;', rho[j][i], j, i, mach, alp)
                rhox = (rho[j][i + 1] - rho[j][i - 1]) / (2 * h[i])
                # if rhox != 0:
                # print('rhox;', rhox, j, i, mach, alp)
                rhoy = (rho[j + 1][i] - rho[j - 1][i]) / (2 * k[j])
                # if rhoy != 0:
                # print('rhoy;', rhoy, j, i, mach, alp)
                rr1 = (rhox * vxx + rhoy * vyy) / (rho[j][i])
                r_int = (phi[j][i + 1] - 2 * phi[j][i] + phi[j][i - 1]) / (h[i] * h[i]) + (
                        phi[j + 1][i] - 2 * phi[j][i] + phi[j - 1][i]) / (k[j] * k[j]) + rr1
                res[j][i] = r_int
    return res
