import numpy as np

def buffered_axis_limits(amin, amax, buffer_factor=1.0):
	"""
	Increases the range (amin, amax) by buffer_factor on each side
	and then rounds to precision of 1/10th min or max.
	Used for generating good plotting limits.
	For example (0, 100) with buffer factor 1.1 is buffered to (-10, 110)
	and then rounded to the nearest 10.
	"""
	diff = amax - amin
	amin -= (buffer_factor-1)*diff
	amax += (buffer_factor-1)*diff
	magnitude = np.floor(np.log10(np.amax(np.abs((amin, amax)) + 1e-100)))
	precision = np.power(10, magnitude-1)
    amin = np.floor(amin/precision) * precision
    amax = np.ceil (amax/precision) * precision
    return (amin, amax)
