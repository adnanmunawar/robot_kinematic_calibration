from DH import *


class Robot:
    def __init__(self, links):
        self.links = links
        self.links_trans_numeric = [sp.Matrix() for x in range(len(links))]
        self.links_trans_symbolic = [sp.Matrix() for x in range(len(links))]
        self.T_tip_base_numeric = get_homogenous_identity_transform()
        self.T_tip_base_symbolic = get_homogenous_identity_transform()

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


link1 = DH(1, 0.5, 0., 0., 0., 0., JointType.REVOLUTE, Convention.MODIFIED)
link2 = DH(2, 0.5, 0., 0., 0., 0., JointType.REVOLUTE, Convention.MODIFIED)

print(link1.T_symbolic)
print(link2.T_symbolic)

rob = Robot([link1, link2])
fk_sym = rob.compute_FK_symbolic()
print(fk_sym)
dm_by_dt = sp.diff(fk_sym, link1.syms.theta)
print(dm_by_dt)
print(dm_by_dt[0, 3].evalf(subs={link1.syms.theta: 0.2}))
print(rob.compute_FK_numeric([2.0, 0.0]))
