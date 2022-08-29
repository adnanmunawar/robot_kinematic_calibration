import sympy as sp
import numpy as np
from enum import Enum


class JointType(Enum):
    REVOLUTE = 0
    PRISMATIC = 1


class Convention(Enum):
    STANDARD = 0
    MODIFIED = 1


class DH_Symbols:
    def __init__(self, link_num):
        self.a = 'a' + str(link_num)
        self.alpha = 'alpha' + str(link_num)
        self.d = 'd' + str(link_num)
        self.theta = 'theta' + str(link_num)

    def to_str(self):
        return self.a + ' ' + self.alpha + ' ' + self.d + ' ' + self.theta


def get_homogenous_identity_transform():
    f = sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return f


class DH:
    def __init__(self, link_num, a_num, alpha_num, d_num, theta_num, offset_num, joint_type, convention):
        self.T = None
        self.a = a_num
        self.alpha = alpha_num
        self.d = d_num
        self.theta = theta_num
        self.offset = offset_num
        self.joint_type = joint_type
        self.convention = convention
        self.syms = DH_Symbols(link_num)

        a, alpha, d, theta = sp.symbols(self.syms.to_str())

        ct = sp.cos(theta)
        st = sp.sin(theta)
        ca = sp.cos(alpha)
        sa = sp.sin(alpha)

        if self.convention == Convention.STANDARD:
            self.T_symbolic = sp.Matrix([
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0, sa, ca, d],
                [0, 0, 0, 1]
            ])
        elif self.convention == Convention.MODIFIED:
            self.T_symbolic = sp.Matrix([
                [ct, -st, 0, a],
                [st * ca, ct * ca, -sa, -d * sa],
                [st * sa, ct * sa, ca, d * ca],
                [0, 0, 0, 1]
            ])
        else:
            assert self.convention == Convention.STANDARD and self.convention == Convention.MODIFIED
            return

        self.T_numberic = sp.lambdify([a, alpha, d, theta], self.T_symbolic)

    def get_numeric_transform(self, joint_cmd):
        th = 0.
        d = self.d
        if self.joint_type == JointType.REVOLUTE:
            th = joint_cmd + self.offset
        elif self.joint_type == JointType.PRISMATIC:
            d = self.d + joint_cmd + self.offset
        else:
            assert self.joint_type == JointType.REVOLUTE and self.joint_type == JointType.PRISMATIC
            return
        return self.T_numberic(self.a, self.alpha, d, th)

    def get_symbolic_transform(self):
        return self.T_symbolic