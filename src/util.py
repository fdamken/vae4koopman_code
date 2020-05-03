from sacred.utils import SacredInterrupt



class LikelihoodDroppingInterrupt(SacredInterrupt):
    STATUS = 'LIKELIHOOD_DROPPED'



class InvalidCovarianceMatrixInterrupt(SacredInterrupt):
    STATUS = 'COVARIANCE_MATRIX_NOT_POSITIVE_SEMIDEFINITE'


    def __init__(self, invalid_matrices):
        self._invalid_matrices = invalid_matrices


    def __str__(self):
        return 'Invalid matrices: ' + str(self._invalid_matrices)


    def __repr__(self):
        return '_invalid_matrices: ' + str(self._invalid_matrices)
