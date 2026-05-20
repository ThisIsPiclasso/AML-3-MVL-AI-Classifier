import numpy as np

from features.base_processor import BasePreprocessor
from features.rgb_normalization_pipeline import RGBNormalizationPreprocessor
from features.rgb_gray_pipeline import RGBToGrayPreprocessor


class GLCMPreprocessor(BasePreprocessor):
    """Compute normalized Gray-Level Co-occurrence Matrices from an RGB patch.

    Four directional spatial offsets are computed:
        - Horizontal:     ``(dx=+1, dy= 0)``
        - Vertical:       ``(dx= 0, dy=+1)``
        - Diagonal:       ``(dx=+1, dy=+1)``
        - Anti-diagonal:  ``(dx=-1, dy=+1)``

    Pixel intensities are quantized from 256 levels down to ``n_levels``
    discrete bins using quantization over the [0, 255] range.
    """

    _OFFSETS = [
        (1, 0),  # horizontal
        (0, 1),  # vertical
        (1, 1),  # diagonal
        (-1, 1),  # anti-diagonal
    ]

    def __init__(
        self,
        n_levels: int = 32,
        symmetric: bool = True,
    ):
        """Initialize the GLCM preprocessor.

        Args:
            n_levels: Number of discrete intensity bins for quantization.
                Default 32 gives a good balance of sensitivity and robustness
            symmetric: If True, GLCM is symmetrized via ``G = G + G.T``,
                treating co-occurrence of ``(i, j)`` and ``(j, i)`` as
                equivalent. 
        """

        # Maps pixel values [0, 255] → [0, n_levels)
        self.n_levels = int(n_levels)
        self.symmetric = symmetric
        self._scale = np.float32(self.n_levels / 255.0)

        self._normalization = RGBNormalizationPreprocessor()
        self._to_gray = RGBToGrayPreprocessor()

    def _compute_glcm(
        self,
        quantized_image: np.ndarray,
        offset_x: int,
        offset_y: int,
    ) -> np.ndarray:
        """Compute one normalized GLCM for a single spatial offset.

        For every valid pixel pair (reference, neighbor) at the given
        offset, counts co-occurrences of intensity levels and normalizes
        to a probability distribution.

        Args:
            quantized_image: Integer array of shape ``(H, W)`` with
                values in ``[0, n_levels - 1]``.
            offset_x: Horizontal displacement from reference to neighbor.
            offset_y: Vertical displacement from reference to neighbor.

        Returns:
            Normalized co-occurrence matrix of shape
            ``(n_levels, n_levels)`` as float32. Values are
            probabilities that sum to 1.0.
        """
        height, width = quantized_image.shape

        # Compute valid index ranges for reference and neighbor pixels.
        # max(0, -offset) and max(0, offset) handle both positive and
        # negative offsets to exclude out-of-bounds pairs.
        #
        # Example for offset_y=1 (vertical, looking down):
        #   reference rows: [0, height-1)  — all rows except the last
        #   neighbor rows:  [1, height)    — all rows except the first
        #   Each ref[y] pairs with nb[y] = ref[y] + 1
        y_ref = slice(max(0, -offset_y), height - max(0, offset_y))
        y_nb = slice(max(0, offset_y), height - max(0, -offset_y))
        x_ref = slice(max(0, -offset_x), width - max(0, offset_x))
        x_nb = slice(max(0, offset_x), width - max(0, -offset_x))


        # Extract aligned reference and neighbor pixel arrays.
        # ravel() flattens to 1D
        reference_levels = quantized_image[y_ref, x_ref].ravel()
        neighbor_levels = quantized_image[y_nb, x_nb].ravel()

        # Build co-occurrence count matrix.
        # G[i, j] = number of times intensity level i appears adjacent
        # to intensity level j at the specified spatial offset.
        co_occurence_matrix = np.zeros((self.n_levels, self.n_levels), dtype=np.float32)
        
        #duplicate handling
        np.add.at(co_occurence_matrix, (reference_levels, neighbor_levels), np.float32(1.0))

        # Symmetrize: treat (i→j) and (j→i) as equivalent.
        # This doubles all counts and makes G[i,j] = G[j,i].
        if self.symmetric:
            co_occurence_matrix = co_occurence_matrix + co_occurence_matrix.T

        # normalization to probabilities
        total = co_occurence_matrix.sum()
        if total > 0:
            co_occurence_matrix /= total

        return co_occurence_matrix

    def __call__(self, image_patch: np.ndarray) -> np.ndarray:
        """Compute normalized GLCMs for four spatial directions.

        Args:
            image_patch: RGB image of shape ``(PATCH_SIZE, PATCH_SIZE, 3)``

        Returns:
            GLCM tensor of shape ``(4, n_levels, n_levels)`` as float32.
            Axis 0 order: [horizontal, vertical, diagonal, anti_diagonal].
            Each ``(n_levels, n_levels)`` slice is a normalized probability
            matrix.
        """
        image = self._normalization(image_patch)
        gray = self._to_gray(image)
        gray_q = np.floor(gray * self._scale).astype(np.int32)
        gray_q = np.clip(gray_q, 0, self.n_levels - 1) # Clip handles the edge case where value=255 produces bin index 16.

        glcms = np.stack(
            [self._compute_glcm(gray_q, offset_x, offset_y) for offset_x, offset_y in self._OFFSETS],
            axis=0,
        )  # (4, n_levels, n_levels), float32

        return glcms
