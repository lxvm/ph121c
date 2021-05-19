"""Interface with the Fortran module evolve
"""

from ..fortran import evolve


def Pauli_ev (L, Nstp, which, cevecs, tevals, k=0, num_threads=1):
    """Evolve the expectation value of a 1-site Pauli operator.
    
    Arguments: (see notebooks for examples)
    L -- int -- number of qubits
    Nstp -- int -- Number of iterations to evolve
    which -- str -- x, y, or z: the Pauli operator to evaluate
    cevecs -- np.ndarray -- shape (2 ** L, 2 ** L) eigenvector per row scaled by coef
    tevals -- np.ndarray -- shape (2 ** L) the exp(-i * evals * dt) vectors
    k -- int -- from 0 to L - 1 the site to evaluate the operator
    num_threads -- int -- the number of threads to parallelize over (default 1)
    """
    assert which in ['x', 'y', 'z'], 'only Pauli x, y, z operators supported'
    try:
        evolve.set_threads(num_threads)
        assert num_threads == evolve.get_threads()
        return evolve.pauli_ev(
            l=L, k=k, nstp=Nstp, which=which, cevecs=cevecs, tevals=tevals
        )
    except AssertionError:
        raise UserWarning(f'{num_threads} threads requested, {evolve.get_threads()} returned')