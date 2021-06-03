import cgp
### CUSTOM FUNCTIONS ###

# class Identity(cgp.OperatorNode):
#     _arity = 1
#     _def_output = "x_0"
#     _def_numpy_output = "x_0"

# class Zero(cgp.OperatorNode):
#     _arity = 1
#     _def_output = "0*x_0"
#     _def_numpy_output = "0*x_0"

# class Opposite(cgp.OperatorNode):
#     _arity = 1
#     _def_output = "-x_0"
#     _def_numpy_output = "-x_0"



# Basic
class Sqrt(cgp.OperatorNode):
    _arity = 1
    _def_output = "math.sqrt(abs(x_0))"
    _def_numpy_output = "np.sqrt(np.abs(x_0))"
    _def_torch_output = "torch.sqrt(torch.abs(x_0))"
    _def_sympy_output = "sqrt(abs(x_0))"

class Div(cgp.OperatorNode):
    _arity = 2
    _def_output = "x_0/x_1 if x_1!=0 else x_0"
    _def_numpy_output = "x_0"
    _def_torch_output = "x_0"
    _def_sympy_output = "x_0"

class Double(cgp.OperatorNode):
    _arity = 1
    _def_output = "2*(x_0)"
    _def_numpy_output = "2*(x_0)"
    _def_torch_output = "2*(x_0)"
    _def_sympy_output = "2*(x_0)"



# Power
class Pow2(cgp.OperatorNode):
    _arity = 1
    _def_output = "x_0**2"
    _def_numpy_output = "np.power(x_0,2)"
    _def_torch_output = "x_0"
    _def_sympy_output = "x_0"

class Pow(cgp.OperatorNode):
    _arity = 2
    _def_output = "x_0**x_1 if x_0 > 0 else 0"
    _def_numpy_output = "np.power(x_0,x_1) if x_0.any() != 0 else 0"
    _def_torch_output = "x_0"
    _def_sympy_output = "x_0"

# Min, Max
class Min(cgp.OperatorNode):
    _arity = 2
    _def_output = "x_0 if x_0 <= x_1 else x_1"
    _def_numpy_output = "x_0"
    _def_torch_output = "x_0"
    _def_sympy_output = "x_0"

class Max(cgp.OperatorNode):
    _arity = 2
    _def_output = "x_0 if x_0 >= x_1 else x_1"
    _def_numpy_output = "x_0"
    _def_torch_output = "x_0"
    _def_sympy_output = "x_0"



# IfElse
class Transistor(cgp.OperatorNode):
    _arity = 3
    _def_output = "x_1 if x_0 >= 0 else x_2"
    _def_numpy_output = "x_0"
    _def_torch_output = "x_0"
    _def_sympy_output = "x_0"


class Compare(cgp.OperatorNode):
    _arity = 3
    _def_output = "x_1 if x_0 >= x_2 else -x_1"
    _def_numpy_output = "x_0"
    _def_torch_output = "x_0"
    _def_sympy_output = "x_0"


# Circular
class Cos(cgp.OperatorNode):
    _arity = 1
    _def_output = "math.cos(x_0)"
    _def_numpy_output = "np.cos(x_0)"
    _def_torch_output = "torch.cos(x_0)"
    _def_sympy_output = "cos(x_0)"

class Sin(cgp.OperatorNode):
    _arity = 1
    _def_output = "math.sin(x_0)"
    _def_numpy_output = "np.sin(x_0)"
    _def_torch_output = "torch.sin(x_0)"
    _def_sympy_output = "sin(x_0)"



# Exponential
class Exp(cgp.OperatorNode):
    _arity = 1
    _def_output = "math.exp(x_0)"
    _def_numpy_output = "np.exp(x_0)"
    _def_torch_output = "torch.exp(x_0)"
    _def_sympy_output = "exp(x_0)"

class Gaussian(cgp.OperatorNode):
    _arity = 1
    _def_output = "math.exp(-x_0**2)"
    _def_numpy_output = "np.exp(-x_0**2)"
    _def_torch_output = "torch.exp(-x_0**2)"
    _def_sympy_output = "exp(-x_0**2)"



# Hyperbolic
class Tanh(cgp.OperatorNode):
    _arity = 1
    _def_output = "math.tanh(x_0)"
    _def_numpy_output = "np.tanh(x_0)"
    _def_torch_output = "torch.tanh(x_0)"
    _def_sympy_output = "tanh(x_0)"


class Inv(cgp.OperatorNode):
    _arity = 1
    _def_output = "1/(abs(x_0)+0.01)"
    _def_numpy_output = "1/(np.abs(x_0)+0.01)"
    _def_torch_output = "1/(torch.abs(x_0)+0.01)"
    _def_sympy_output = "1/(abs(x_0)+0.01)"





