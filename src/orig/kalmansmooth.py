import numpy as np



def kalmansmooth(A, C, Q, R, x0, P0, Y):
    (N, p, T) = Y.shape
    K = len(x0)
    tiny = np.exp(-700)
    I = np.eye(K)
    const = (2 * np.pi) ** (-p / 2)
    problem = 0
    lik = 0

    Xpre = np.zeros((N, K))
    Xcur = np.zeros((N, K, T))
    Xfin = np.zeros((N, K, T))

    Ppre = np.zeros((K, K, T))
    Pcur = np.zeros((K, K, T))
    Pfin = np.zeros((K, K, T))

    Pt = np.zeros((K, K))
    Pcov = np.zeros((K, K))
    Kcur = np.zeros((K, p))
    invP = np.zeros((p, p))
    J = np.zeros((K, K, T))

    #
    # Forward pass.

    R = R + (R == 0) * tiny

    Xpre = np.ones((N, 1)) @ x0.T
    Ppre[:, :, 0] = P0
    invR = np.diag(1 / R)

    for t in range(0, T):
        if K < p:
            # TODO
            # temp1=rdiv(C,R);
            # temp2=temp1*Ppre(:,:,t); % inv(R)*C*Ppre
            # temp3=C'*temp2;
            # temp4=inv(I+temp3)*temp1';
            # invP=invR-temp2*temp4;
            # CP= temp1' - temp3*temp4;    % C'*invP
            raise Exception('Not yet implemented!')
        else:
            temp1 = np.diag(R) + C @ Ppre[:, :, t] @ C.T
            invP = np.linalg.inv(temp1)
            CP = C.T @ invP

        Kcur = Ppre[:, :, t] @ CP
        KC = Kcur @ C
        Ydiff = Y[:, :, t] - Xpre @ C.T
        Xcur[:, :, t] = Xpre + Ydiff @ Kcur.T
        Pcur[:, :, t] = Ppre[:, :, t] - KC @ Ppre[:, :, t]
        if t < T - 1:
            Xpre = Xcur[:, :, t] @ A.T
            Ppre[:, :, t + 1] = A @ Pcur[:, :, t] @ A.T + Q

        detiP = np.sqrt(np.linalg.det(invP))
        if np.isreal(detiP) and detiP > 0:
            lik = lik + N * np.log(detiP) - 0.5 * np.sum(np.sum(np.multiply(Ydiff, Ydiff @ invP), axis = 0), axis = 0)
        else:
            problem = 1

    lik = lik + N * T * np.log(const)

    #
    # Backward pass.

    A1 = np.zeros((K, K))
    A2 = np.zeros((K, K))
    A3 = np.zeros((K, K))
    Ptsum = np.zeros((K, K))
    YX = np.zeros((p, K))

    t = T - 1
    Xfin[:, :, t] = Xcur[:, :, t]
    Pfin[:, :, t] = Pcur[:, :, t]
    Pt = Pfin[:, :, t] + Xfin[:, :, t].T @ Xfin[:, :, t] / N
    A2 = -Pt
    Ptsum = Pt

    YX = Y[:, :, t].T @ Xfin[:, :, t]

    for t in reversed(range(0, T - 1)):
        J[:, :, t] = Pcur[:, :, t] @ A.T @ np.linalg.inv(Ppre[:, :, t + 1])
        Xfin[:, :, t] = Xcur[:, :, t] + (Xfin[:, :, t + 1] - Xcur[:, :, t] @ A.T) @ J[:, :, t].T

        Pfin[:, :, t] = Pcur[:, :, t] + J[:, :, t] @ (Pfin[:, :, t + 1] - Ppre[:, :, t + 1]) @ J[:, :, t].T
        Pt = Pfin[:, :, t] + Xfin[:, :, t].T @ Xfin[:, :, t] / N
        Ptsum = Ptsum + Pt
        YX = YX + Y[:, :, t].T @ Xfin[:, :, t]

    A3 = Ptsum - Pt
    A2 = Ptsum + A2

    t = T - 1
    Pcov = (I - KC) @ A @ Pcur[:, :, t - 1]
    A1 = A1 + Pcov + Xfin[:, :, t].T @ Xfin[:, :, t - 1] / N

    for t in reversed(range(1, T - 1)):
        Pcov = (Pcur[:, :, t] + J[:, :, t] @ (Pcov - A @ Pcur[:, :, t])) @ J[:, :, t - 1].T
        A1 = A1 + Pcov + Xfin[:, :, t].T @ Xfin[:, :, t - 1] / N

    if problem:
        print('problem')
        problem = 0

    return lik, Xfin, Pfin, Ptsum, YX, A1, A2, A3
