from .complex import Complex

# implement the quadratic and Sine maps for complex numbers
def quadratic_map(c, z): return Complex(z.r**2-z.i**2 + c.r, 2*z.r*z.i + c.r)

# implementing the Sine map in PyTorch
def sin_map(c, z): return z.sin().multiply(c)