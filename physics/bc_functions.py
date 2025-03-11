class get_boundary_functions():
    def __init__(self, bc_type_list, loss_function):
        self.loss_function = loss_function
        self.boundary_function_u = getattr(self, f"{bc_type_list['U']['type']}")
        self.boundary_function_p = getattr(self, f"{bc_type_list['p']['type']}")

    def no_slip(self,component):
        # normal and tangent components are zero
        pass

    def zero_gradient(self,component):
        # normal gradient are zero
        pass

    def fixed_value(self,component):
        pass

    def __call__(self, out, derivatives):

        loss = self.boundary_function_u(out, derivatives)
        return loss