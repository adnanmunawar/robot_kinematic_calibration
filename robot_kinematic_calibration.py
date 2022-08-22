import numpy.linalg as la
import numpy as np


class RobotKinematics:
    def __init__(self, l1, l2):
        self.l1 = l1
        self.l2 = l2

        self.num_links = 2
        self.num_dh_params = 2

    def print_params(self):
        print('\tRobot Kinematic Params:\n\t\tl1: ', self.l1, '\n\t\tl2: ', self.l2)


class KinematicCalibration:
    def __init__(self):
        pass

    @staticmethod
    def compute_fk(kinematics, joints):
        t1 = joints[0]
        t2 = joints[1]
        l1 = kinematics.l1
        l2 = kinematics.l2
        x = l1 * np.cos(t1) + l2 * np.cos(t1 + t2)
        y = l1 * np.sin(t1) + l2 * np.sin(t1 + t2)

        return np.hstack((zip(x, y)))

    @staticmethod
    def compute_regressor(kinematics, joints):
        t1 = joints[0]
        t2 = joints[1]
        l1 = kinematics.l1
        l2 = kinematics.l2

        dl1_x = np.cos(t1)
        dl1_y = np.sin(t1)

        dl2_x = np.cos(t1 + t2)
        dl2_y = np.sin(t1 + t2)

        dt1_x = -l1 * np.sin(t1) - l2 * np.sin(t1 + t2)
        dt1_y = l1 * np.cos(t1) + l2 * np.cos(t1 + t2)

        dt2_x = -l2 * np.sin(t1 + t2)
        dt2_y = l2 * np.cos(t1 + t2)

        x_diff = np.vstack([dl1_x, dl2_x, dt1_x, dt2_x]).transpose()
        y_diff = np.vstack([dl1_y, dl2_y, dt1_y, dt2_y]).transpose()

        reg = np.vstack(zip(x_diff, y_diff))
        return reg

    @staticmethod
    def compute_inverse_regressor(kinematics, joints):
        reg = KinematicCalibration.compute_regressor(kinematics, joints)
        reg_transpose = reg.transpose()
        A = np.matmul(reg_transpose, reg)
        invA = la.inv(A)
        reg_pinverse = np.matmul(invA, reg_transpose)
        return reg_pinverse


def main():
    kin_ground_truth_robot = RobotKinematics(1.0, 2.0)
    kin_inaccurate_robot = RobotKinematics(0.88, 2.27)
    kin_inaccurate_robot_copy = RobotKinematics(0.88, 2.27)

    kc = KinematicCalibration()

    num_samples = 100

    theta1_min = -0.6
    theta1_max = 0.2
    theta2_min = -0.3
    theta2_max = 0.4
    theta1_range = theta1_max - theta1_min
    theta2_range = theta2_max - theta2_min
    theta1_step = theta1_range / num_samples
    theta2_step = theta2_range / num_samples

    thetas = np.zeros([2, num_samples])
    t1 = theta1_min
    t2 = theta2_min
    for i in range(num_samples):
        t1 = t1 + theta1_step
        t2 = t2 + theta2_step
        thetas[0, i] = t1
        thetas[1, i] = t2

    r_ground_truth = kc.compute_fk(kin_ground_truth_robot, thetas)

    for i in range(4):
        r_nominal = kc.compute_fk(kin_inaccurate_robot, thetas)
        delta_r = r_ground_truth - r_nominal
        inv_regeressor = kc.compute_inverse_regressor(kin_inaccurate_robot, thetas)
        delta_param = np.matmul(inv_regeressor, delta_r)
        print(delta_param)
        kin_inaccurate_robot.l1 = kin_inaccurate_robot.l1 + delta_param[0]
        kin_inaccurate_robot.l2 = kin_inaccurate_robot.l2 + delta_param[1]

    print('Ground Truth Params')
    kin_ground_truth_robot.print_params()

    print('Original Non Calibrated Params')
    kin_inaccurate_robot_copy.print_params()

    print('Calibrated Params')
    kin_inaccurate_robot.print_params()


if __name__ == "__main__":
    main()

