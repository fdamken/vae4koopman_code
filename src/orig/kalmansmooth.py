import numpy as np



def kalmansmooth(A, C, Q, R, x0, P0, Y):
    (no_observation_sequences, observation_dim, T) = Y.shape
    state_dim = len(x0)
    I = np.eye(state_dim)
    likelihood = 0

    m = np.zeros((no_observation_sequences, state_dim, T))
    P = np.zeros((state_dim, state_dim, T))
    V = np.zeros((state_dim, state_dim, T))
    J = np.zeros((state_dim, state_dim, T))

    #
    # Forward pass.

    R = R + (R == 0) * np.exp(-700)

    # Initialize the forward pass.
    m_pre = np.ones((no_observation_sequences, 1)) @ x0.T
    P[:, :, 0] = P0
    for t in range(0, T):
        # TODO: state_dim < observation_dim!
        # invR = np.diag(1 / R)
        # temp1 = C / R[:, np.newaxis]  # temp1=rdiv(C,R);
        # temp2 = temp1 @ P[:, :, t]  # temp2=temp1*Ppre(:,:,t);
        # temp3 = C.T @ temp2  # temp3=C'*temp2;
        # temp4 = np.linalg.inv(I + temp3) @ temp1.T  # temp4=inv(I+temp3)*temp1';
        # invP = invR - temp2 @ temp4  # invP=invR-temp2*temp4;
        # CP = temp1.T - temp3 @ temp4  # CP= temp1' - temp3*temp4;

        if state_dim < observation_dim:
            raise Exception('state_dim < observation_dim is not (yet) supported!')

        invP = np.linalg.inv(C @ P[:, :, t] @ C.T + np.diag(R))
        K = P[:, :, t] @ C.T @ invP
        Ydiff = Y[:, :, t] - m_pre @ C.T
        m[:, :, t] = m_pre + Ydiff @ K.T
        V[:, :, t] = P[:, :, t] - K @ C @ P[:, :, t]

        # Initialize the next pass (iff there is a next pass).
        if t < T - 1:
            m_pre = m[:, :, t - 1] @ A.T
            P[:, :, t + 1] = A @ V[:, :, t] @ A.T + Q

        # Likelihood computation.
        detiP = np.sqrt(np.linalg.det(invP))
        if np.isreal(detiP) and detiP > 0:
            likelihood = likelihood + no_observation_sequences * np.log(detiP) - 0.5 * np.sum(np.sum(np.multiply(Ydiff, Ydiff @ invP), axis = 0), axis = 0)
        else:
            raise Exception('problem')

    likelihood = likelihood + no_observation_sequences * T * np.log((2 * np.pi) ** (-observation_dim / 2))

    #
    # Backward pass.

    x_hat = np.zeros((no_observation_sequences, state_dim, T))
    V_backward = np.zeros((state_dim, state_dim, T))

    A1 = np.zeros((state_dim, state_dim))

    t = T - 1
    x_hat[:, :, t] = m[:, :, t]
    V_backward[:, :, t] = V[:, :, t]
    Pt = V_backward[:, :, t] + x_hat[:, :, t].T @ x_hat[:, :, t] / no_observation_sequences
    A2 = -Pt
    Ptsum = Pt

    YX = Y[:, :, t].T @ x_hat[:, :, t]

    for t in reversed(range(0, T - 1)):
        J[:, :, t] = V[:, :, t] @ A.T @ np.linalg.inv(P[:, :, t + 1])
        x_hat[:, :, t] = m[:, :, t] + (x_hat[:, :, t + 1] - m[:, :, t] @ A.T) @ J[:, :, t].T

        V_backward[:, :, t] = V[:, :, t] + J[:, :, t] @ (V_backward[:, :, t + 1] - P[:, :, t + 1]) @ J[:, :, t].T
        Pt = V_backward[:, :, t] + x_hat[:, :, t].T @ x_hat[:, :, t] / no_observation_sequences
        Ptsum = Ptsum + Pt
        YX = YX + Y[:, :, t].T @ x_hat[:, :, t]

    A3 = Ptsum - Pt
    A2 = Ptsum + A2

    t = T - 1
    Pcov = (I - K @ C) @ A @ V[:, :, t - 1]
    A1 = A1 + Pcov + x_hat[:, :, t].T @ x_hat[:, :, t - 1] / no_observation_sequences

    for t in reversed(range(1, T - 1)):
        Pcov = (V[:, :, t] + J[:, :, t] @ (Pcov - A @ V[:, :, t])) @ J[:, :, t - 1].T
        A1 = A1 + Pcov + x_hat[:, :, t].T @ x_hat[:, :, t - 1] / no_observation_sequences

    return likelihood, x_hat, V_backward, Ptsum, YX, A1, A2, A3
