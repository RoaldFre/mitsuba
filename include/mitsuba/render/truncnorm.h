#ifndef __MITSUBA_RENDER_TRUNCNORM_H_
#define __MITSUBA_RENDER_TRUNCNORM_H_
/*
 * Code directly based on work of Jonathan Olmsted:
 * https://github.com/olmjo/RcppTN
 * http://olmjo.com/computing/RcppTN/
 */

#include <mitsuba/core/platform.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/sampler.h>

// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; tab-width: 4 -*-

// This C++ code implements an Accept/Reject sampler for a single
// Truncated Normal random variable with a mixture of algorithms
// depending on distributional parameters.

MTS_NAMESPACE_BEGIN

inline Float truncnormPdf(const Float _mean,
            const Float _sd,
            const Float _lo,
            const Float _hi,
            const Float _z) {
    /* We do everything explicitly in doubles here, because the
     * exponentials are too prone to over/underflow with single precision
     * */
    double mean(_mean);
    double sd(_sd);
    double lo(_lo);
    double hi(_hi);
    double z(_z);

    SAssert(lo <= hi);
    SAssert(sd >= 0);

    if (z < lo || z > hi)
        return 0.0f;

    if (lo == hi)
        return 1.0f;

    if (sd == 0) {
        double scale = hi - lo;
        if (!std::isfinite(scale))
            SLog(EError, "I currently only support finite intervals when sd==0");
        double acceptedError = Epsilon * scale;
        if (lo <= mean && mean <= hi)
            return math::abs(z - mean) < acceptedError ? 1.0 : 0.0;
        if (mean > hi)
            return math::abs(z - hi) < acceptedError ? 1.0 : 0.0;
        return math::abs(z - lo) < acceptedError ? 1.0 : 0.0;
    }

    if (std::isinf(sd)) {
        return 1.0/(hi - lo);
    }

    SAssert(sd > 0);

    /* If both erfs in the denominator go to one (meaning both
     * bounds are [far] to the right of the mean), we simply use the
     * mirror property to get both bounds to the left of the mean
     * where each erf is something small and there is less catastrophic
     * cancellation */
    if (lo >= mean && hi > mean) {
        SAssert(z >= mean);
        double loFlipped = mean - (hi - mean);
        double hiFlipped = mean - (lo - mean);
        double zFlipped  = mean - (z - mean);
        lo = loFlipped;
        hi = hiFlipped;
        z = zFlipped;
        SAssert(lo < mean && hi <= mean && z <= mean);
    }
    SAssert(lo < mean);


    double pdf;
    double c_stdhi = (hi - mean) / sd; // standarized bound
    double c_stdlo = (lo - mean) / sd; // standarized bound
    double c_stdz  = (z - mean) / sd; // standarized sample
    if (c_stdhi > -8.) { // in this case: full erf expression should be sufficiently stable
        double absoluteExpArgument = 0.5 * pow((z - mean) / sd, 2);
        double erfDiff = math::erf((hi - mean)/(SQRT_TWO*sd))
                       - math::erf((lo - mean)/(SQRT_TWO*sd));
        //SAssert(absoluteExpArgument < LOG_REDUCED_PRECISION); // this can underflow if pdf becomes 0, which is OK...
        SAssert(erfDiff > 0);
        pdf = 2.0*exp(-absoluteExpArgument)
                / ((sqrt(TWO_PI) * sd) * erfDiff);
        if (!std::isfinite(pdf))
            SLog(EWarn, "full pdf %e: stdlo:%e stdhi:%e stdz:%e sd:%e | m:%e lo:%e hi:%e z:%e",
                    pdf, c_stdlo, c_stdhi, c_stdz, sd,
                    mean, lo, hi, z);
    } else {
        /* Our bounds are *waaaay* to the left of the mean, so the exponential
         * and erfs can potentially underflow. Expand the erfs and cancel
         * exp(lo^2/2) factors [note that exp(lo^2/2) > exp(hi^2/2), because
         * lo and hi are both <0, and lo<hi, so lo is bigger in absolute
         * value] */
        SAssert(c_stdlo < 0);
        SAssert(c_stdhi < 0);
        SAssert(c_stdz < 0);
        // Expansion up to 7th order in |c_stdhi| and |c_stdlo|
        double c_stdlo2 = c_stdlo*c_stdlo;     double c_stdhi2 = c_stdhi*c_stdhi;
        double c_stdlo4 = c_stdlo2*c_stdlo2;   double c_stdhi4 = c_stdhi2*c_stdhi2;
        double c_stdlo6 = c_stdlo4*c_stdlo2;   double c_stdhi6 = c_stdhi4*c_stdhi2;
        double c_stdlo7 = c_stdlo6*c_stdlo;    double c_stdhi7 = c_stdhi6*c_stdhi;

        double pdfDenom = (c_stdhi6 - c_stdhi4 + 3*c_stdhi2 - 15); /* assuming c_stdlo = -infinity */
        if (std::isfinite(c_stdlo7)) /* add seperately only when finite to avoid NaNs */
              pdfDenom += (c_stdlo6 - c_stdlo4 + 3*c_stdlo2 - 15) /* correction for c_stdlo != -infinity */
                            * c_stdhi7/c_stdlo7 * exp(0.5*(c_stdhi*c_stdhi - c_stdlo*c_stdlo));
        pdf = -c_stdhi7 * exp(0.5*(c_stdhi*c_stdhi - c_stdz*c_stdz)) / pdfDenom;
        pdf /= sd; // transform back to non-standard setting
        SAssert(!(pdf < 0));
        if (!std::isfinite(pdf)) {
            SLog(EWarn, "expanded pdf %e: stdlo:%e stdhi:%e stdz:%e sd:%e | m:%e lo:%e hi:%e z:%e",
                    pdf, c_stdlo, c_stdhi, c_stdz, sd,
                    mean, lo, hi, z);
            return 0;
        }
    }
    SAssert(std::isfinite(pdf));
    SAssert(pdf >= 0);
    return pdf;
}


// TODO: wasteful...
inline Float stdnorm(Sampler *sampler) {
    return warp::squareToStdNormal(sampler->next2D()).x;
}


/// Check if simpler subalgorithm is appropriate.
inline bool CheckSimple(const Float low, ///< lower bound of distribution
                        const Float high ///< upper bound of distribution
                        ) {
  // Init Values Used in Inequality of Interest
  Float val1 = (2 * sqrt(exp(1))) / (low + sqrt(pow(low, 2) + 4));
  Float val2 = exp((pow(low, 2) - low * sqrt(pow(low, 2) + 4)) / (4)) ;
  //

  // Test if Simple is Preferred
  if (high > low + val1 * val2) {
    return true ;
  } else {
    return false ;
  }
}


/// XXX This check was missing from:
/// https://github.com/olmjo/RcppTN
/// http://olmjo.com/computing/RcppTN/
inline bool CheckRejectFromUniformInsteadOfNormal(
        const Float low, const Float high) {
    if (low * high > 0)
        return false;
    return high - low < sqrt(TWO_PI);
}

/// Draw using algorithm 1.

///
/// Naive Accept-Reject algorithm.
///
/// Samples z from gaussian and rejects when out of bounds

inline Float UseAlg1(const Float low, ///< lower bound of distribution
                      const Float high, ///< upper bound of distribution
                      Sampler *sampler
                      ) {
  // Init Valid Flag
  int valid = 0 ;
  //

  // Init Draw Storage
  Float z = 0.0 ;
  //

  // Loop Until Valid Draw
  while (valid == 0) {
    z = stdnorm(sampler);

    if (z <= high && z >= low) {
      valid = 1 ;
    }
  }
  //

  // Returns
  return z ;
  //
}

/// Draw using algorithm 2.

///
///  Accept-Reject Algorithm
///
/// Samples from exponential distribution and rejects to transform to
/// 'one-sided' bounded Gaussian.

inline Float UseAlg2(const Float low, ///< lower bound of distribution
                      Sampler *sampler
                      ) {
  // Init Values
  const Float alphastar = (low +
                sqrt(pow(low, 2) + 4.0)
                ) / (2.0) ;
  const Float alpha = alphastar ;
  Float z;
  Float rho;
  Float u;
  //

  // Init Valid Flag
  int valid = 0 ;
  //

  // Loop Until Valid Draw
  while (valid == 0) {
    Float e = -log(sampler->next1D());
    z = low + e / alpha ;

    rho = exp(-pow(alpha - z, 2) / 2) ;
    u = sampler->next1D() ;
    if (u <= rho) {
      // Keep Successes
      valid = 1 ;
    }
  }
  //

  // Returns
  return z ;
  //
}

/// Draw using algorithm 3.

///
/// Accept-Reject Algorithm
///
/// Samples z uniformly within lo..hi and rejects based on gaussian weight

inline Float UseAlg3(const Float low, ///< lower bound of distribution
                      const Float high, ///< upper bound of distribution
                      Sampler *sampler
                      ) {
  // Init Valid Flag
  int valid = 0 ;
  //

  // Declare Qtys
  Float rho;
  Float z;
  Float u;
  //

  // Loop Until Valid Draw
  while (valid == 0) {
    z = low + sampler->next1D() * (high - low);
    if (0 < low) {
      rho = exp((pow(low, 2) - pow(z, 2)) / 2) ;
    } else if (high < 0) {
      rho = exp((pow(high, 2) - pow(z, 2)) / 2) ;
    } else {
      SAssert(0 <= high && low <= 0);
      rho = exp(- pow(z, 2) / 2) ;
    }

    u = sampler->next1D();
    if (u <= rho) {
      valid = 1 ;
    }
  }
  //

  // Returns
  return z ;
  //
}


/// Draw from an arbitrary truncated normal distribution.

///
/// See Robert (1995): <br />
/// Reference Type: Journal Article <br />
/// Author: Robert, Christian P. <br />
/// Primary Title: Simulation of truncated normal variables <br />
/// Journal Name: Statistics and Computing <br />
/// Cover Date: 1995-06-01 <br />
/// Publisher: Springer Netherlands <br />
/// Issn: 0960-3174 <br />
/// Subject: Mathematics and Statistics <br />
// Start Page: 121 <br />
// End Page: 125 <br />
/// Volume: 5 <br />
/// Issue: 2 <br />
/// Url: http://dx.doi.org/10.1007/BF00143942 <br />
/// Doi: 10.1007/BF00143942 <br />
///

/* TODO: sampler generates only a few set of discrete samples for, eg,:
 * mean = 0, sd = 1, lo = 10, hi = 10.001
 * -> numerical cancellation issue + should simply sample uniformly here,
 *  because pdf is nearly constant anyway
 */

inline Float truncnorm(const Float mean,
            const Float sd,
            const Float low,
            const Float high,
            Sampler *sampler
            ) {

    SAssert(low <= high);
    SAssert(sd >= 0);

    if (low == high)
        return low;

    if (sd == 0) {
        if (low <= mean && mean <= high)
            return mean;
        if (mean > high)
            return high;
        return low;
    }

    if (std::isinf(sd)) {
        return low + sampler->next1D() * (high - low);
    }

  // Init Useful Values
  Float draw = 0;
  int type = 0 ;
  int valid = 0 ; // used only when switching to a simplified version
          // of Alg 2 within Type 4 instead of the less
          // efficient Alg 3

  // Set Current Distributional Parameters
   const Float c_mean = mean ;
   Float c_sd = sd ;
   const Float c_low = low ;
   const Float c_high = high ;
   Float c_stdlow = (c_low - c_mean) / c_sd ;
   Float c_stdhigh = (c_high - c_mean) / c_sd ; // bounds are standardized

   const Float INF = std::numeric_limits<Float>::infinity();


  // Map Conceptual Cases to Algorithm Cases
  // Case 1 (Simple Deterministic AR)
  // mu \in [low, high]
  if (0 <= c_stdhigh &&
      0 >= c_stdlow
      ) {
    type = 1 ;
  }

  // Case 2 (Robert 2009 AR)
  // mu < low, high = Inf
  if (0 < c_stdlow &&
      c_stdhigh == INF
      ) {
    type = 2 ;
  }

  // Case 3 (Robert 2009 AR)
  // high < mu, low = -Inf
  if (0 > c_stdhigh &&
      c_stdlow == -INF
      ) {
    type = 3 ;
  }

  // Case 4 (Robert 2009 AR)
  // mu -\in [low, high] & (abs(low) =\= Inf =\= high)
  if ((0 > c_stdhigh || 0 < c_stdlow) &&
      !(c_stdhigh == INF || c_stdlow == -INF)
      ) {
    type = 4 ;
  }

  ////////////
  // Type 1 //
  ////////////
  if (type == 1) {
    if (CheckRejectFromUniformInsteadOfNormal(c_stdlow, c_stdhigh))
      draw = UseAlg3(c_stdlow, c_stdhigh, sampler) ;
    else
      draw = UseAlg1(c_stdlow, c_stdhigh, sampler) ;
  }

  ////////////
  // Type 3 //
  ////////////
  if (type == 3) {
    c_stdlow = -1 * c_stdhigh ;
    c_stdhigh = INF ;
    c_sd = -1 * c_sd ; // hack to get two negative signs to cancel out

    // Use Algorithm #2 Post-Adjustments
    type = 2 ;
  }

  ////////////
  // Type 2 //
  ////////////
  if (type == 2) {
    draw = UseAlg2(c_stdlow, sampler) ;
  }

    ////////////
    // Type 4 //
    ////////////

    if (type == 4) {
        // Flip to make the standardized bounds positive if they aren't
        // already, or else Alg2 fails
        SAssert(c_stdlow * c_stdhigh > 0); // double check that both have same sign
        if (c_stdlow < 0) {
            double tmp = c_stdlow;
            c_stdlow = -c_stdhigh ;
            c_stdhigh = -tmp;
            c_sd = -1 * c_sd ; // hack to get two negative signs to cancel out
        }

        if (CheckSimple(c_stdlow, c_stdhigh)) {
            while (valid == 0) {
                draw = UseAlg2(c_stdlow, sampler) ;
                // use the simple
                // algorithm if it is more
                // efficient
                if (draw <= c_stdhigh) {
                    valid = 1 ;
                }
            }
        } else {
            draw = UseAlg3(c_stdlow, c_stdhigh, sampler) ; // use the complex
            // algorithm if the simple
            // is less efficient
        }
    }

    if (draw < c_stdlow || draw > c_stdhigh) {
        SLog(EWarn, "Generated out of bounds draw: %f not in [%f .. %f]",
                draw, c_stdlow, c_stdhigh);
    }
    return math::clamp(c_mean + c_sd * draw, low, high); // to protect against round-off
}

MTS_NAMESPACE_END

#endif // __MITSUBA_RENDER_TRUNCNORM_H_
