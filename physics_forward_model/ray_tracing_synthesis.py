import torch
import scipy.io
import numpy as np
from PIL import Image
from torchvision import transforms


class SoSToTimeShiftTransformer:
    def __init__(self, bf_sos: float = 1540.0, device: str = None):
        """
        Load fixed forward model matrices and initialize the transformer.

        Args:
            bf_sos (float): Background speed of sound in m/s (default = 1540).
            device (str or None): 'cuda', 'cpu', or None for auto-detect.
        """
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print('Class initializing, device is', self.device)
        self.bf_sos = bf_sos
        self.matrices = self._load_fixed_matrices()

    def _load_fixed_matrices(self):
        """
        Load fixed set of forward model matrices.

        Returns:
            dict: Dictionary of matrix_key to matrix_tensor.
        """
        file_paths = {
            '0psf': 'd11_11_psf0_forward_modelmatrix.mat',
            '7p5psf': 'd11_11_7p5_forward_modelmatrix.mat',
            'minus7p5psf': 'd11_11_minus7p5_forward_modelmatrix.mat'
        }

        matrices = {}
        for key, path in file_paths.items():
            mat = scipy.io.loadmat(path)
            if 'L' not in mat:
                raise KeyError(f"Expected key 'L' in {path}, but not found.")
            matrices[key] = torch.tensor(mat['L'], dtype=torch.float32, device=self.device)
        return matrices

    def transform(self, sos_map_np: np.ndarray, matrix_key: str) -> np.ndarray:
        """
        Convert a SoS map to a time-shift map using the specified forward model matrix.

        Args:
            sos_map_np (np.ndarray): 2D SoS input map.
            matrix_key (str): Key of the matrix to use (e.g. '0psf').

        Returns:
            np.ndarray: 2D time-shift map.
        """
        if matrix_key not in self.matrices:
            raise ValueError(f"Matrix key '{matrix_key}' not found. Available keys: {list(self.matrices.keys())}")

        forward_matrix = self.matrices[matrix_key]
        sos_tensor = torch.tensor(sos_map_np.flatten(order='F'), dtype=torch.float32, device=self.device)
        delta_inv_sos = 1.0 / sos_tensor - 1.0 / self.bf_sos
        time_shift_tensor = forward_matrix @ delta_inv_sos
        H, W = sos_map_np.shape
        return time_shift_tensor.view(W, H).T.cpu().numpy()


def image_to_sos_map(image: Image.Image) -> np.ndarray:
    """
    Convert a grayscale image to a SoS map.

    Args:
        image (PIL.Image.Image): Input grayscale image.

    Returns:
        np.ndarray: Scaled SoS map of shape (73, 89)
    """
    gray_transform = transforms.Grayscale()
    gray_image = gray_transform(image)
    gray_image = gray_image.resize((89, 73))  # (width, height)
    gray_np = np.array(gray_image, dtype=np.float32) / 255.0
    sos_map = gray_np * 200 + 1400
    return sos_map