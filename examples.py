from robot import *
import numpy as np

def psm_example():
    l_pitch2yaw = 0.0091
    l_tool = 0.4162
    l_rcc = 0.4318
    l_yaw2ctrlpoint = 0.0102
    PI = np.pi
    PI_2 = PI / 2.0
    links = []
    links.append(DH(link_num=1, a=0., alpha=PI_2, d=0., theta=0., offset=PI_2, joint_type=JointType.REVOLUTE, convention=Convention.MODIFIED))
    links.append(DH(link_num=2, a=0., alpha=-PI_2, d=0., theta=0., offset=-PI_2, joint_type=JointType.REVOLUTE, convention=Convention.MODIFIED))
    links.append(DH(link_num=3, a=0., alpha=PI_2, d=-l_rcc, theta=0., offset=0., joint_type=JointType.PRISMATIC, convention=Convention.MODIFIED))
    # links.append(DH(link_num=4, a=0., alpha=0., d=l_tool, theta=0., offset=0., joint_type=JointType.REVOLUTE, convention=Convention.MODIFIED))
    # links.append(DH(link_num=5, a=0., alpha=-PI_2, d=0., theta=0., offset=-PI_2, joint_type=JointType.REVOLUTE, convention=Convention.MODIFIED))
    # links.append(DH(link_num=6, a=l_pitch2yaw, alpha=-PI_2, d=0., theta=0., offset=-PI_2, joint_type=JointType.REVOLUTE, convention=Convention.MODIFIED))
    # links.append(DH(link_num=7, a=0.5, alpha=-PI_2, d=l_yaw2ctrlpoint, theta=0., offset=0., joint_type=JointType.REVOLUTE, convention=Convention.MODIFIED))


    rob = Robot(links)
    fk_sym = rob.compute_FK_symbolic()
    # print(fk_sym)
    joint_cmds = [0.0, 0.2, 0.3, 0.0, 0.2, -0.5, 0.0]
    # print(rob.compute_FK_numeric(joint_cmds))
    J_sym = rob._compute_jacobian_symbolic()
    print(J_sym)

    J_num = rob._compute_jacobian_numeric(joint_cmds)
    print(J_num)


if __name__ == '__main__':
    psm_example()