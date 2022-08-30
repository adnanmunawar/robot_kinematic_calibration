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
    def __init__(self, link_num, a, alpha, d, theta, offset, joint_type, convention):
        self.T = None
        self.a = a
        self.alpha = alpha
        self.d = d
        self.theta = theta
        self.offset = offset
        self.joint_type = joint_type
        self.convention = convention
        self.link_num = link_num
        self.syms = DH_Symbols(link_num)

        a_sym, alpha_sym, d_sym, theta_sym = sp.symbols(self.syms.to_str())

        ct = sp.cos(theta_sym)
        st = sp.sin(theta_sym)
        ca = sp.cos(alpha_sym)
        sa = sp.sin(alpha_sym)

        if self.convention == Convention.STANDARD:
            self.T_symbolic = sp.Matrix([
                [ct, -st * ca, st * sa, a_sym * ct],
                [st, ct * ca, -ct * sa, a_sym * st],
                [0, sa, ca, d_sym],
                [0, 0, 0, 1]
            ])
        elif self.convention == Convention.MODIFIED:
            self.T_symbolic = sp.Matrix([
                [ct, -st, 0, a_sym],
                [st * ca, ct * ca, -sa, -d_sym * sa],
                [st * sa, ct * sa, ca, d_sym * ca],
                [0, 0, 0, 1]
            ])
        else:
            assert self.convention == Convention.STANDARD and self.convention == Convention.MODIFIED
            return

        self.T_numeric = sp.lambdify([a_sym, alpha_sym, d_sym, theta_sym], self.T_symbolic)

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
        return self.T_numeric(self.a, self.alpha, d, th)

    def get_syms_as_dict(self, joint_cmd):
        th = 0.
        d = self.d
        if self.joint_type == JointType.REVOLUTE:
            th = joint_cmd + self.offset
        elif self.joint_type == JointType.PRISMATIC:
            d = self.d + joint_cmd + self.offset
        else:
            assert self.joint_type == JointType.REVOLUTE and self.joint_type == JointType.PRISMATIC
            return
        syms_dict = dict()
        syms_dict[self.syms.a] = self.a
        syms_dict[self.syms.alpha] = self.alpha
        syms_dict[self.syms.d] = d
        syms_dict[self.syms.theta] = th
        return syms_dict
    def get_symbolic_transform(self):
        return self.T_symbolic