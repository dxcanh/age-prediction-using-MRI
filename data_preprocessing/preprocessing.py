import numpy as np
import scipy.ndimage
from skimage.restoration import denoise_nl_means, estimate_sigma
import nibabel as nib
import logging
from typing import Dict, Tuple, Optional
from scipy.ndimage import median_filter
from scipy.optimize import minimize
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MedicalImageCleaner:
    """A class to preprocess medical images by removing scanner artifacts and preparing clean data for models."""
    
    def __init__(self, config: Dict):
        """
        Initialize the cleaner with a configuration dictionary.
        
        Args:
            config (Dict): Configuration for cleaning steps, e.g., {
                'denoise': {'patch_size': 5, 'patch_distance': 6},
                'bias_correction': {'max_iterations': 100},
                'intensity_normalization': {'method': 'zscore'},
                'smoothing': {'sigma_mm': 0.5}
            }
        """
        self.config = config
        logger.info("Initialized MedicalImageCleaner with provided config.")

    def denoise_image(self, data: np.ndarray) -> np.ndarray:
        """Remove Rician-like noise using Non-Local Means denoising."""
        cfg = self.config.get('denoise', {})
        patch_size = cfg.get('patch_size', 5)
        patch_distance = cfg.get('patch_distance', 6)
        
        # Estimate noise level
        sigma_est = np.mean(estimate_sigma(data, channel_axis=None))
        if sigma_est < 1e-6:
            logger.warning("Estimated noise level too low; skipping denoising.")
            return data
        
        # Apply Non-Local Means denoising
        denoised_data = denoise_nl_means(
            data,
            patch_size=patch_size,
            patch_distance=patch_distance,
            h=0.6 * sigma_est,
            fast_mode=True,
            preserve_range=True
        )
        logger.info(f"Denoised image with estimated sigma={sigma_est:.4f}.")
        return denoised_data

    def correct_bias_field(self, data: np.ndarray) -> np.ndarray:
        """Correct bias field using polynomial fitting."""
        cfg = self.config.get('bias_correction', {})
        max_iterations = cfg.get('max_iterations', 100)
        
        def polynomial_bias_field(coords, coeffs, shape):
            X, Y, Z = shape
            x = np.linspace(-1, 1, X)
            y = np.linspace(-1, 1, Y)
            z = np.linspace(-1, 1, Z)
            poly_x = np.polyval(coeffs[:3], x)
            poly_y = np.polyval(coeffs[3:6], y)
            poly_z = np.polyval(coeffs[6:], z)
            return poly_x[:, np.newaxis, np.newaxis] * poly_y[np.newaxis, :, np.newaxis] * poly_z[np.newaxis, np.newaxis, :]

        def objective_function(coeffs, data, mask):
            bias = polynomial_bias_field(None, coeffs, data.shape)
            corrected = data / (bias + 1e-8)
            return np.var(corrected[mask])

        # Create a mask for non-zero regions
        mask = data > np.percentile(data, 5)
        initial_coeffs = np.ones(9)  # 3 coefficients per axis
        try:
            result = minimize(
                objective_function,
                initial_coeffs,
                args=(data, mask),
                method='Powell',
                options={'maxiter': max_iterations}
            )
            bias_field = polynomial_bias_field(None, result.x, data.shape)
            corrected_data = data / (bias_field + 1e-8)
            logger.info("Bias field corrected successfully.")
            return corrected_data
        except Exception as e:
            logger.warning(f"Bias correction failed: {str(e)}. Returning original data.")
            return data

    def normalize_intensity(self, data: np.ndarray) -> np.ndarray:
        """Normalize intensity to standardize data for model input."""
        cfg = self.config.get('intensity_normalization', {})
        method = cfg.get('method', 'zscore')
        
        if method == 'zscore':
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val < 1e-8:
                logger.warning("Standard deviation too low; skipping normalization.")
                return data
            normalized_data = (data - mean_val) / std_val
            logger.info("Applied Z-score normalization.")
        elif method == 'minmax':
            min_val = np.min(data)
            max_val = np.max(data)
            if (max_val - min_val) < 1e-8:
                logger.warning("Intensity range too small; skipping normalization.")
                return data
            normalized_data = (data - min_val) / (max_val - min_val)
            logger.info("Applied Min-Max normalization.")
        else:
            logger.warning(f"Unknown normalization method {method}; skipping.")
            return data
        return normalized_data

    def smooth_image(self, data: np.ndarray, pixdim: Tuple[float, float, float]) -> np.ndarray:
        """Apply light Gaussian smoothing to reduce minor artifacts."""
        cfg = self.config.get('smoothing', {})
        sigma_mm = cfg.get('sigma_mm', 0.5)
        sigma_voxels = [sigma_mm / dim for dim in pixdim]
        smoothed_data = scipy.ndimage.gaussian_filter(data, sigma=sigma_voxels)
        logger.info(f"Applied Gaussian smoothing with sigma_mm={sigma_mm:.2f}.")
        return smoothed_data

    def clean(self, data: np.ndarray, pixdim: Tuple[float, float, float]) -> np.ndarray:
        """
        Apply all cleaning steps to produce model-ready data.
        
        Args:
            data (np.ndarray): Input 3D image data.
            pixdim (Tuple[float, float, float]): Pixel dimensions (mm) in x, y, z directions.
        
        Returns:
            np.ndarray: Cleaned image data.
        """
        logger.info("Starting cleaning pipeline.")
        try:
            # Ensure data is float32 for numerical stability
            cleaned_data = data.astype(np.float32)
            
            # Apply cleaning steps in logical order
            cleaned_data = self.denoise_image(cleaned_data)
            cleaned_data = self.correct_bias_field(cleaned_data)
            cleaned_data = self.normalize_intensity(cleaned_data)
            cleaned_data = self.smooth_image(cleaned_data, pixdim)
            
            logger.info("Cleaning pipeline completed successfully.")
            return cleaned_data
        except Exception as e:
            logger.error(f"Cleaning failed: {str(e)}")
            raise

def main():
    """Example usage of the MedicalImageCleaner."""
    # Sample configuration
    config = {
        'denoise': {'patch_size': 5, 'patch_distance': 6},
        'bias_correction': {'max_iterations': 100},
        'intensity_normalization': {'method': 'zscore'},
        'smoothing': {'sigma_mm': 0.5}
    }
    
    # Sample data (replace with actual NIfTI loading in practice)
    sample_data = np.random.rand(64, 64, 64).astype(np.float32)
    pixdim = (1.0, 1.0, 1.0)  # Example pixel dimensions in mm
    
    # Initialize cleaner
    cleaner = MedicalImageCleaner(config)
    
    # Apply cleaning
    cleaned_data = cleaner.clean(sample_data, pixdim)
    
    print("Cleaning completed. Shape of cleaned data:", cleaned_data.shape)

if __name__ == "__main__":
    main()