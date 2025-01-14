import torch

def complex_plane(xrange=(-2,2), yrange=None, res=1000, device='cuda'):
    '''Return a 2-tuple of grids corresponding to the real and imaginary points 
    on the complex plane, respectively.'''
    if yrange == None: yrange = xrange
    if type(res) == int: res = (res,res)
    
    # np.linspace(...) --> torch.linspace(...).cuda()
    x = torch.linspace(xrange[0], xrange[1], res[0]).to(device)
    y = torch.linspace(yrange[1], yrange[0], res[1]).to(device)
    
    # np.meshgrid --> torch.meshgrid
    real_plane, imag_plane = torch.meshgrid(x,y)
    
    cplane = Complex(real_plane.transpose(0,1), imag_plane.transpose(0,1))
    
    return cplane


class Complex(torch.Tensor):

    def __init__(self, r, i, **kwargs):
        super.__init__(kwargs)
        self.r = r
        self.i = i

    def magnitude(self):
        '''returns the magnitude of a complex tensor, given a real component, `r`, and an imaginary component, `i`'''
        return torch.sqrt(self.r**2 + self.i**2)
    
    def multiply(self, other):
        return Complex(
            self.r * other.r - self.i * other.i,
            self.r * other.i + self.i * other.r
        )
    
    def sin(self):
        return Complex(
            torch.sin(self.r) * torch.cosh(self.i),
            torch.sinh(self.i) * torch.cos(self.r)
        )