from DH import *


class Robot:
    def __init__(self, links):
        self.links = links
        self.links_trans_numeric = [sp.Matrix() for x in range(len(links))]
        self.links_trans_symbolic = [sp.Matrix() for x in range(len(links))]
        self.T_tip_base_symbolic = get_homogenous_identity_transform()
        self.T_tip_base_numeric = get_homogenous_identity_transform()
        self.Jac_symbolic = sp.Matrix.ones(6, len(links))

    def compute_FK_numeric(self, joint_cmds):
        for i in range(len(self.links)):
            self.links_trans_numeric[i] = self.links[i].get_numeric_transform(joint_cmds[i])

        for i in range(len(self.links)):
            self.T_tip_base_numeric = self.T_tip_base_numeric * self.links_trans_numeric[i]

        return self.T_tip_base_numeric

    def compute_FK_symbolic(self):
        for i in range(len(self.links)):
            self.links_trans_symbolic[i] = self.links[i].get_symbolic_transform()

        for i in range(len(self.links)):
            self.T_tip_base_symbolic = self.T_tip_base_symbolic * self.links_trans_symbolic[i]

        return self.T_tip_base_symbolic

    def _compute_jacobian_symbolic(self):
        num_links = len(self.links)
        T_tip_base_sym = self.compute_FK_symbolic()
        for i in range(num_links):
            self.Jac_symbolic[0:3, i] = sp.diff(T_tip_base_sym[0:3, 3], self.links[i].syms.theta)
        return self.Jac_symbolic


def main():
    link1 = DH(1, 0.5, 0., 0., 0., 0., JointType.REVOLUTE, Convention.MODIFIED)
    link2 = DH(2, 0.5, 0., 0., 0., 0., JointType.REVOLUTE, Convention.MODIFIED)

    print(link1.T_symbolic)
    print(link2.T_symbolic)

    rob = Robot([link1, link2])
    fk_sym = rob.compute_FK_symbolic()
    print(fk_sym)
    print(rob.compute_FK_numeric([2.0, 0.0]))
    J = rob._compute_jacobian_symbolic()
    print(J)

if __name__ == '__main__':
    main()
