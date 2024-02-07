from typing import List

from PIL import Image
import torch
import torch.nn.functional as F

def permute_decoded_latent(decoded):
    rgb = decoded.detach()
    rgb = rgb.float().cpu().permute(0, 2, 3, 1)
    rgb = rgb.permute(1, 0, 2, 3)
    rgb = rgb.flatten(start_dim=1, end_dim=2)
    return rgb

def clip_image_at_percentiles(image, lower_percentile, upper_percentile):
    """
    Clips the image at the given lower and upper percentiles.
    """
    # Flatten the image to compute percentiles
    flattened_image = image.flatten()

    # Compute the lower and upper bounds
    lower_bound = torch.quantile(flattened_image, lower_percentile)
    upper_bound = torch.quantile(flattened_image, upper_percentile)

    # Clip the image
    clipped_image = torch.clamp(image, lower_bound, upper_bound)

    return clipped_image

def gaussian_kernel(size, sigma):
    """
    Creates a Gaussian Kernel with the given size and sigma
    """
    # Create a tensor with coordinates of a grid
    x = torch.arange(size).float() - size // 2
    y = torch.arange(size).float() - size // 2
    y, x = torch.meshgrid(y, x)

    # Calculate the gaussian kernel
    gaussian_kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))

    # Normalize the kernel so the sum is 1
    gaussian_kernel /= gaussian_kernel.sum()

    return gaussian_kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

def gaussian_blur(image, kernel_size, sigma):
    """
    Applies Gaussian Blur to the given image using the specified kernel size and sigma
    """

    image = image.permute(2, 0, 1)

    # Create the kernel
    kernel = gaussian_kernel(kernel_size, sigma)

    # Repeat the kernel for each channel of the image
    kernel = kernel.repeat(image.size(0), 1, 1, 1)

    # Apply padding
    padding = kernel_size // 2

    # Apply the gaussian kernel, assuming the image is in 'channels first' format
    blurred_image = F.conv2d(image.unsqueeze(0), kernel, padding=padding, groups=image.size(0))

    return blurred_image.squeeze(0).permute(1, 2, 0)

def stack_images_horizontally(images: List, save_path=None):
    widths, heights = list(zip(*(i.size for i in images)))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new("RGBA", (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    if save_path is not None:
        new_im.save(save_path)
    return new_im


def images2gif(
    images: List, save_path, optimize=True, duration=None, loop=0, disposal=2
):
    if duration is None:
        duration = int(5000 / len(images))
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        optimize=optimize,
        duration=duration,
        loop=loop,
        disposal=disposal,
    )


def stack_images_vertically(images: List, save_path=None):
    widths, heights = list(zip(*(i.size for i in images)))
    max_width = max(widths)
    total_height = sum(heights)
    new_im = Image.new("RGBA", (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    if save_path is not None:
        new_im.save(save_path)
    return new_im


def merge_images(images: List):
    if isinstance(images[0], Image.Image):
        return stack_images_horizontally(images)

    images = list(map(stack_images_horizontally, images))
    return stack_images_vertically(images)
