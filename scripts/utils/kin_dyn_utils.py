from .math_tools import Math
from forward_kinematics import compute_jacobian, direct_kinematics, geometric2analyticJacobian

import numpy as np

def numericalInverseKinematics(p_d, q0, line_search = False, wrap = False):
    math_utils = Math()

    # hyper-parameters
    epsilon = 1e-06 # Tolerance for stopping criterion
    lambda_ = 1e-08  # Regularization or damping factor (1e-08->0.01)
    max_iter = 200  # Maximum number of iterations
    # For line search only
    #gamma = 0.5
    beta = 0.5 # Step size reduction

    # initialization of variables
    iter = 0
    alpha = 1  # Step size
    log_grad = []
    log_err = []

    # Inverse kinematics with line search
    while True:
        # evaluate  the kinematics for q0
        J = compute_jacobian(q0)
        _, _, _, _, _, T_0e = direct_kinematics(q0)

        p_e = T_0e[:3,3]
        R = T_0e[:3,:3]
        rpy = math_utils.rot2eul(R)
        roll = rpy[0]
        p_e = np.append(p_e,roll)

        # error
        e_bar = p_e - p_d
        J_bar = geometric2analyticJacobian(J,T_0e)
        # take first 4 rows correspondent to our task
        J_bar = J_bar[:4,:]
        # evaluate the gradient
        grad = J_bar.T.dot(e_bar)

        log_grad.append(np.linalg.norm(grad))
        log_err.append(np.linalg.norm(e_bar))

        if np.linalg.norm(grad) < epsilon:
            print("IK Convergence achieved!, norm(grad) :", np.linalg.norm(grad) )
            print("Inverse kinematics solved in {} iterations".format(iter))     
            break
        if iter >= max_iter:                
            print("Warning: Max number of iterations reached, the iterative algorithm has not reached convergence to the desired precision. Error is:  ", np.linalg.norm(e_bar))
            break
        # Compute the error
        JtJ= np.dot(J_bar.T,J_bar) + np.identity(J_bar.shape[1])*lambda_
        JtJ_inv = np.linalg.inv(JtJ)
        P = JtJ_inv.dot(J_bar.T)
        dq = - P.dot(e_bar)

        if not line_search:
            q1 = q0 + dq * alpha
            q0 = q1
        else:
            print("Iter # :", iter)
            # line search loop
            while True:
                #update
                q1 = q0 + dq*alpha
                # evaluate  the kinematics for q1
                _, _, _, _, _, T_0e1 = direct_kinematics(q1)
                p_e1 = T_0e1[:3,3]
                R1 = T_0e1[:3,:3]
                rpy1 = math_utils.rot2eul(R1)
                roll1 = rpy1[0]
                p_e1 = np.append(p_e1,roll1)
                e_bar_new = p_e1 - p_d
                #print "e_bar1", np.linalg.norm(e_bar_new), "e_bar", np.linalg.norm(e_bar)

                error_reduction = np.linalg.norm(e_bar) - np.linalg.norm(e_bar_new)
                threshold = 0.0 # more restrictive gamma*alpha*np.linalg.norm(e_bar)

                if error_reduction < threshold:
                    alpha = beta*alpha
                    print (" line search: alpha: ", alpha)
                else:
                    q0 = q1
                    alpha = 1
                    break

        iter += 1
           

 
    # wrapping prevents from outputs outside the range -2pi, 2pi
    if wrap:
        for i in range(len(q0)):
            while q0[i] >= 2 * np.pi:
                q0[i] -= 2 * np.pi
            while q0[i] < -2 * np.pi:
                q0[i] += 2 * np.pi

    return q0, log_err, log_grad


def fifthOrderPolynomialTrajectory(tf,start_pos,end_pos, start_vel = 0, end_vel = 0, start_acc =0, end_acc = 0):

    # Matrix used to solve the linear system of equations for the polynomial trajectory
    polyMatrix = np.array([[1,  0,              0,               0,                  0,                0],
                           [0,  1,              0,               0,                  0,                0],
                           [0,  0,              2,               0,                  0,                0],
                           [1, tf,np.power(tf, 2), np.power(tf, 3),    np.power(tf, 4),  np.power(tf, 5)],
                           [0,  1,           2*tf,3*np.power(tf,2),   4*np.power(tf,3), 5*np.power(tf,4)],
                           [0,  0,              2,             6*tf, 12*np.power(tf,2),20*np.power(tf,3)]])
    
    polyVector = np.array([start_pos, start_vel, start_acc, end_pos, end_vel, end_acc])
    matrix_inv = np.linalg.inv(polyMatrix)
    polyCoeff = matrix_inv.dot(polyVector)

    return polyCoeff
