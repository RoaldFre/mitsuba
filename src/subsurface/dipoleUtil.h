#if !defined(__DIPOLE_UTIL_H)
#define __DIPOLE_UTIL_H

MTS_NAMESPACE_BEGIN

inline Float dEon_C1(const Float n) {
	Float r;
	if (n > 1.0) {
		r = -9.23372 + n * (22.2272 + n * (-20.9292 + n * (10.2291 + n * (-2.54396 + 0.254913 * n))));
	} else {
		r = 0.919317 + n * (-3.4793 + n * (6.75335 + n *  (-7.80989 + n *(4.98554 - 1.36881 * n))));
	}
	return r / 2.0;
}
inline Float dEon_C2(const Float n) {
	Float r = -1641.1 + n * (1213.67 + n * (-568.556 + n * (164.798 + n * (-27.0181 + 1.91826 * n))));
	r += (((135.926 / n) - 656.175) / n + 1376.53) / n;
	return r / 3.0;
}

inline Float dEon_A(const Float eta) {
	return (1 + 3*dEon_C2(eta)) / (1 - 2*dEon_C1(eta));
}

MTS_NAMESPACE_END

#endif /* __DIPOLE_UTIL_H */
