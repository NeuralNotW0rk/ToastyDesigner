# Based on https://dcato98.github.io/blog/jupyter/visualization/python/cuda/numpy/pytorch/video/2020/04/28/CUDA-Accelerated-Julia-Fractals.html

import torch
import matplotlib.pyplot as plt

def torch_quadratic_method(c, z, n_iterations, divergence_value, device='cuda'):
    '''Iteratively apply the quadratic map `z = z^2 + c` for `n_iterations` times
        c: tuple of the real and imaginary components of the constant value
        z: tuple of the real and imaginary components of the initial z-value
    '''
    # np.zeros_like(...) --> torch.zeros_like(...).cuda()
    stable_steps = torch.zeros_like(z[0]).to(device)
    
    for i in range(n_iterations):
        # numpy handled squaring complex magnitudes, for PyTorch we implement this ourselves
        mask = torch.lt(torch_complex_magnitude(*z), divergence_value).to(device)
        stable_steps += mask.to(torch.float32)
        
        # likewise, we manually implement one iteration of the quadratic map
        z = (z[0]**2-z[1]**2 + c[0], # real
             2*z[0]*z[1] + c[1])     # imaginary
        
    # don't forget to put the array onto the cpu for plotting!
    return stable_steps / n_iterations

# add `transform_func` to our standard plotting function to accept arbitrary image transformations
def torch_plot_julia(julia_img, sz=16, eps=.1, transform_func=None):
    img = -torch.log(julia_img + eps) if transform_func is None else transform_func(julia_img)
    plt.figure(figsize=(sz,sz))
    plt.imshow(img.cpu())
    plt.show()

def torch_quadratic_method(c, z, n_iterations, divergence_value, device='cuda'):
    '''Iteratively apply the quadratic map `z = z^2 + c` for `n_iterations` times
        c: tuple of the real and imaginary components of the constant value
        z: tuple of the real and imaginary components of the initial z-value
    '''
    # add c[0] to get the right shape
    stable_steps = torch.zeros_like(c[0] + z[0]).to(device)
    
    for i in range(n_iterations):
        mask = torch.lt(torch_complex_magnitude(*z), divergence_value).to(device)
        stable_steps += mask.to(torch.float32)
        z = (z[0]**2-z[1]**2 + c[0], # real
             2*z[0]*z[1] + c[1])     # imaginary
    return stable_steps / n_iterations

def torch_plot_julia_grid(images, figsize=(12,12)):
    rows, cols = images.shape[:2]
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    for i in range(rows):
        for j in range(cols):
            axs[i,j].imshow(images[i,j].cpu())
            axs[i,j].get_xaxis().set_visible(False)
            axs[i,j].get_yaxis().set_visible(False)
    plt.show()


# modify our quadratic julia method to generalize to any mapping function
def torch_julia_generator(c=None, z=None, n_iterations=50, divergence_value=50, map_func=torch_quadratic_map, device='cuda'):
    '''A generic julia fractal generator (defaults produce the Mandelbrot set).'''
    if c is None: c = torch_complex_plane(device=device)
    if z is None: z = torch_complex_plane(device=device)
    stable_steps = torch.zeros_like(c[0]+z[0]).to(device)
    for i in range(n_iterations):
        mask = torch.lt(z[1].abs(), divergence_value)
        stable_steps += mask.to(torch.float32)
        z = map_func(c, z)
    return (stable_steps / n_iterations).cpu()