import unittest
import numpy as np


def _assert_numpy_close(a, b, atol=None, rtol=None, err_msg=''):
  assert a.shape == b.shape
  atol = max(tolerance(a.dtype, atol), tolerance(b.dtype, atol))
  rtol = max(tolerance(a.dtype, rtol), tolerance(b.dtype, rtol))
  _assert_numpy_allclose(a, b, atol=atol * a.size, rtol=rtol * b.size,
                         err_msg=err_msg)


class DistmlTestCase(unittest.TestCase):
  """Base class for JAX tests including numerical checks and boilerplate."""

  def setUp(self):
    super(DistmlTestCase, self).setUp()
    # config.update('jax_enable_checks', True)
    # # We use the adler32 hash for two reasons.
    # # a) it is deterministic run to run, unlike hash() which is randomized.
    # # b) it returns values in int32 range, which RandomState requires.
    # self._rng = npr.RandomState(zlib.adler32(self._testMethodName.encode()))


  def assertArraysEqual(self, x, y, *, check_dtypes=True):
    """Assert that x and y arrays are exactly equal."""
    if check_dtypes:
      self.assertDtypesMatch(x, y)
    np.testing.assert_array_equal(x, y)

  def assertArraysAllClose(self, x, y, *, check_dtypes=True, atol=None,
                           rtol=None):
    """Assert that x and y are close (up to numerical tolerances)."""
    self.assertEqual(x.shape, y.shape)
    atol = max(tolerance(_dtype(x), atol), tolerance(_dtype(y), atol))
    rtol = max(tolerance(_dtype(x), rtol), tolerance(_dtype(y), rtol))

    _assert_numpy_allclose(x, y, atol=atol, rtol=rtol)

    if check_dtypes:
      self.assertDtypesMatch(x, y)

  def assertDtypesMatch(self, x, y, *, canonicalize_dtypes=True):
    if not config.x64_enabled and canonicalize_dtypes:
      self.assertEqual(_dtypes.canonicalize_dtype(_dtype(x)),
                       _dtypes.canonicalize_dtype(_dtype(y)))
    else:
      self.assertEqual(_dtype(x), _dtype(y))

  def assertAllClose(self, x, y, *, check_dtypes=True, atol=None, rtol=None,
                     canonicalize_dtypes=True):
    """Assert that x and y, either arrays or nested tuples/lists, are close."""
    if isinstance(x, dict):
      self.assertIsInstance(y, dict)
      self.assertEqual(set(x.keys()), set(y.keys()))
      for k in x.keys():
        self.assertAllClose(x[k], y[k], check_dtypes=check_dtypes, atol=atol,
                            rtol=rtol, canonicalize_dtypes=canonicalize_dtypes)
    elif is_sequence(x) and not hasattr(x, '__array__'):
      self.assertTrue(is_sequence(y) and not hasattr(y, '__array__'))
      self.assertEqual(len(x), len(y))
      for x_elt, y_elt in zip(x, y):
        self.assertAllClose(x_elt, y_elt, check_dtypes=check_dtypes, atol=atol,
                            rtol=rtol, canonicalize_dtypes=canonicalize_dtypes)
    elif hasattr(x, '__array__') or np.isscalar(x):
      self.assertTrue(hasattr(y, '__array__') or np.isscalar(y))
      if check_dtypes:
        self.assertDtypesMatch(x, y, canonicalize_dtypes=canonicalize_dtypes)
      x = np.asarray(x)
      y = np.asarray(y)
      self.assertArraysAllClose(x, y, check_dtypes=False, atol=atol, rtol=rtol)
    elif x == y:
      return
    else:
      raise TypeError((type(x), type(y)))

