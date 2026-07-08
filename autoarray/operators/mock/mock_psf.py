import numpy as np


class MockPSF:
    def __init__(self, operated_mapping_matrix=None):
        self.operated_mapping_matrix = operated_mapping_matrix
        self.convolve_over_sample_size = 1

    def convolved_mapping_matrix_from(
        self, mapping_matrix, mask, use_mixed_precision=False, xp=np
    ):
        return self.operated_mapping_matrix
