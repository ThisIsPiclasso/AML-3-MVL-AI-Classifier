class GLCMPreprocessor(BasePreprocessor):
    """
    Compute normalized Gray-Level Co-occurrence Matrices (GLCMs) from a
    256x256 RGB image patch.
    Output order:
        [horizontal, vertical, diagonal, anti_diagonal]
    """

    def __init__(
        self,
        n_levels=16,
        epsilon=1e-8,
        symmetric=True
    ):
        if not isinstance(n_levels, (int, np.integer)) or n_levels <= 0:
            raise ValueError("n_levels must be a positive integer.")
        self.n_levels = int(n_levels)
        self.epsilon = epsilon
        self.symmetric = symmetric
        self.RGB_normalization_preprocessor = RGBNormalizationPreprocessor( 
                epsilon=epsilon
        )
        self.grayscale_preprocessor = RGBToGrayPreprocessor(
                epsilon=epsilon,
                normalize=True,
            )

    def _compute_glcm(self, qimg, dx, dy):
        """
          reference pixel: (y, x)
          neighbor pixel:   (y + dy, x + dx)
          only use valid in-bounds pixel pairs
        """
        h, w = qimg.shape #height and width of the quantized grayscale image, should be (256, 256)

        if dy >= 0:
            y_ref = slice(0, h - dy)
            y_nb = slice(dy, h)
        else:
            y_ref = slice(-dy, h)
            y_nb = slice(0, h + dy)

        if dx >= 0:
            x_ref = slice(0, w - dx)
            x_nb = slice(dx, w)
        else:
            x_ref = slice(-dx, w)
            x_nb = slice(0, w + dx)

        ref = qimg[y_ref, x_ref].ravel() #flatten the 2D array of reference pixel levels into a 1D array
        nb = qimg[y_nb, x_nb].ravel() #flatten the 2D array of neighbor pixel levels into a 1D array, aligned with ref

        G = np.zeros((self.n_levels, self.n_levels), dtype=np.float64) #2D array of shape (n_levels, n_levels) initialized to zero
        np.add.at(G, (ref, nb), 1.0)  # G[i, j] counts how often level i is adjacent to level j.

        # symmetrize to ignore pair order.
        if self.symmetric:
            G = G + G.T

        # Convert counts into a probability distribution.
        total = G.sum()
        if total > 0:
            G = G / (total + self.epsilon)

        return G

    def __call__(self, image_patch):
        image = self.RGB_normalization_preprocessor(image_patch)
        gray = self.grayscale_preprocessor(image)
        # quantize grayscale intensities into discrete bins
        gray_q = np.floor(gray * self.n_levels).astype(np.int32)
        gray_q = np.clip(gray_q, 0, self.n_levels - 1)
        # compute directional co-occurrence matrices
        glcm_horizontal = self._compute_glcm(gray_q, dx=1, dy=0)
        glcm_vertical = self._compute_glcm(gray_q, dx=0, dy=1)
        glcm_diagonal = self._compute_glcm(gray_q, dx=1, dy=1)
        glcm_antidiagonal = self._compute_glcm(gray_q, dx=-1, dy=1)

        return [
            glcm_horizontal,
            glcm_vertical,
            glcm_diagonal,
            glcm_antidiagonal,
        ]
