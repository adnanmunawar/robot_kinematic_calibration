from DH import *

class Robot:
    def __init__(self, links):
        self.links = links
        self.num_links = len(self.links)
        self._check_links()
        self.links_trans_numeric = [sp.Matrix() for x in range(self.num_links)]
        self.links_trans_symbolic = [sp.Matrix() for x in range(self.num_links)]
        self.T_tip_base_symbolic = sp.ones(4)
        self.T_tip_base_numeric = sp.ones(4)
        self.Jac_symbolic = sp.Matrix.ones(6, len(links))

    def _check_links(self):
        error = False
        for i in range(self.num_links - 1):
            if self.links[i+1].link_num - self.links[i].link_num != 1:
                error = True
                break

        if error:
            print("ERROR! LINKS ARE NOT INDEXED INCREMENTALLY")
            for i in range(self.num_links ):
                print("\t", i, ") Link IDX: ", self.links[i].link_num)
            raise 'ERROR, LINKS ARE NOT INDEXED INCREMENTALLY'
        return True

    def compute_FK_numeric(self, joint_cmds):
        for i in range(self.num_links):
            self.links_trans_numeric[i] = self.links[i].get_numeric_transform(joint_cmds[i])

        for i in range(self.num_links):
            self.T_tip_base_numeric = self.T_tip_base_numeric * self.links_trans_numeric[i]

        return self.T_tip_base_numeric

    def compute_FK_symbolic(self):
        for i in range(self.num_links):
            self.links_trans_symbolic[i] = self.links[i].get_symbolic_transform()

        for i in range(self.num_links):
            self.T_tip_base_symbolic = self.T_tip_base_symbolic * self.links_trans_symbolic[i]

        return self.T_tip_base_symbolic

    def _compute_jacobian_symbolic(self):
        T_tip_base_sym = self.compute_FK_symbolic()
        for i in range(self.num_links):
            self.Jac_symbolic[0:3, i] = sp.diff(T_tip_base_sym[0:3, 3], self.links[i].syms.theta)
            if self.links[i].joint_type == JointType.REVOLUTE:
                self.Jac_symbolic[3:6, i] = sp.Matrix([0., 0., 1.])
            else:
                self.Jac_symbolic[3:6, i] = sp.Matrix([0., 0., 0.])
        return self.Jac_symbolic

    def _compute_jacobian_numeric(self, joint_cmds):
        syms_dict = {}
        for i in range(self.num_links):
            temp_dict = self.links[i].get_syms_as_dict(joint_cmds[i])
            syms_dict.update(temp_dict)
        # print(syms_dict)
        Jac_numeric = self.Jac_symbolic.evalf(subs=syms_dict)
        return Jac_numeric
