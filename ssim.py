import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray
import cv2

i = 0

def draw_grid_on_image(image, window_size):
    """ Draws a grid on the image with window_size intervals. """
    grid_image = image.copy()
    height, width = grid_image.shape


    # Draw horizontal lines
    for y in range(0, height, window_size):
        cv2.line(grid_image, (0, y), (width, y), (0, 0, 255), 1)  # Red line

    # Draw vertical lines
    for x in range(0, width, window_size):
        cv2.line(grid_image, (x, 0), (x, height), (0, 0, 255), 1)  # Red line

    return grid_image


def ssim(img1, img2, K1=0.01, K2=0.03, L=1, win_size=11, sigma=1.5, weight_luminance=2.0, weight_contrast=1.0, weight_structure=1.0):
    # Convert images to float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Constants for SSIM
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # Mean filters
    mu1 = gaussian_filter(img1, sigma=sigma)
    mu2 = gaussian_filter(img2, sigma=sigma)

    # Variance and covariance filters
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(img1 ** 2, sigma=sigma) - mu1_sq
    sigma2_sq = gaussian_filter(img2 ** 2, sigma=sigma) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sigma=sigma) - mu1_mu2

    # Compute the SSIM components
    luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    contrast = (2 * np.sqrt(sigma1_sq) * np.sqrt(sigma2_sq) + C2) / (sigma1_sq + sigma2_sq + C2)
    structure = (sigma12 + C2 / 2) / (np.sqrt(sigma1_sq * sigma2_sq) + C2 / 2)

    print(luminance.mean(), contrast.mean(), structure.mean())

    # Combine the components with the specified weights to get the SSIM map
    ssim_map = (luminance ** weight_luminance) * (structure ** weight_structure)

    # Return the average SSIM value
    return ssim_map.mean()
"""
def compute_mean_variance_covariance(image1, image2, window_size=11, sigma=1.5, save_mean_images=True):
    half_window = window_size // 2
    np.set_printoptions(threshold=np.inf)

    # Define Gaussian window
    gaussian_window = np.outer(gaussian_filter(np.ones((window_size, window_size)), sigma),
                               gaussian_filter(np.ones((window_size, window_size)), sigma))


    def mean_variance_covariance(img):
        mean = cv2.filter2D(img, -1, gaussian_window)

        mean_sq = mean ** 2

        var = cv2.filter2D(img ** 2, -1, gaussian_window) - mean_sq
        return mean, var

    def covariance(img1, img2):
        print("og shape", img1.shape, img2.shape)
        mean1, var1 = mean_variance_covariance(img1)
        mean2, var2 = mean_variance_covariance(img2)
        cov = cv2.filter2D(img1 * img2, -1, gaussian_window) - (mean1 * mean2)

        #print(mean1.shape, var1.shape, mean2.shape, var2.shape, cov.shape)
        if save_mean_images:
            # Save local means as images with grid
            mean1_grid = draw_grid_on_image(np.uint8(mean1), window_size)
            mean2_grid = draw_grid_on_image(np.uint8(mean2), window_size)

            # Thoughts
            # - what if you typed the mean? like each area in the mean will be a different value, either 1 for p, 2 for btn, etc (maybe this can be a vector)
            # that way you can compare the mean with respect to the type?
            #
            # if it is a vector then you can calculate how far the 2d vector is from each other. if you have something that measures how far it deviates, then the
            # further then the more it penalizes the structure          

            cv2.imwrite('mean1_with_grid.png', mean1_grid)
            cv2.imwrite('mean2_with_grid.png', mean2_grid)

            # save teh covariance image
            cov_grid = draw_grid_on_image(np.uint8(cov / cov.max() * 255), window_size)
            cv2.imwrite('cov_with_grid.png', cov_grid)


        return mean1, var1, mean2, var2, cov

    mean1, var1, mean2, var2, cov = covariance(image1, image2)
    return mean1, var1, mean2, var2, cov

def compute_local_mean_penalty(image1, image2, window_size=11, sigma=1.5):
    half_window = window_size // 2
    
    # Define Gaussian window
    # Create a 1D Gaussian kernel
    gaussian_1d = cv2.getGaussianKernel(ksize=window_size, sigma=sigma)

    # Create a 2D Gaussian kernel by outer product of 1D Gaussian kernels
    gaussian_window = np.outer(gaussian_1d, gaussian_1d)

    # Normalize the kernel to ensure the sum is 1
    gaussian_window /= np.sum(gaussian_window)
    
    # Compute local means
    mean1 = cv2.filter2D(image1, -1, gaussian_window)
    mean2 = cv2.filter2D(image2, -1, gaussian_window)
    
    # Calculate mean difference penalty
    mean_diff = np.abs(mean1 - mean2)
    mean_diff_penalty = mean_diff / (mean1 + mean2 + 1e-10)  # Adding a small constant to avoid division by zero
    
    return mean_diff_penalty

def ssim(image1, image2, window_size=11, sigma=1.5):
    mean1, var1, mean2, var2, cov = compute_mean_variance_covariance(image1, image2, window_size, sigma)
    mean_diff_penalty = compute_local_mean_penalty(image1, image2, window_size, sigma)
    
    C1 = 6.5025  # C1 constant
    C2 = 58.5225e-4  # C2 constant

    # Calculate structural similarity index
    structural_index = (2 * cov + C2) / (var1 + var2 + C2)

    print("mdp", mean_diff_penalty.mean())
    
    # Combine the structural index with the mean difference penalty
    final_ssim = structural_index - mean_diff_penalty
    
    return final_ssim.mean()
"""
"""
def ssim(image1, image2, window_size=11, sigma=1.5):
    # ORIGINAL
    mean1, var1, mean2, var2, cov = compute_mean_variance_covariance(image1, image2, window_size, sigma)
    C2 = 58.5225e-4  # C2 constant

    structural_index = (2 * cov + C2) / (var1 + var2 + C2)
    return structural_index.mean()

"""

