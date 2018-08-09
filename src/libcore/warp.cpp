/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <mitsuba/core/warp.h>

MTS_NAMESPACE_BEGIN

namespace warp {

Vector squareToUniformSphere(const Point2 &sample) {
    Float z = 1.0f - 2.0f * sample.y;
    Float r = math::safe_sqrt(1.0f - z*z);
    Float sinPhi, cosPhi;
    math::sincos(2.0f * M_PI * sample.x, &sinPhi, &cosPhi);
    return Vector(r * cosPhi, r * sinPhi, z);
}

Vector squareToUniformHemisphere(const Point2 &sample) {
    Float z = sample.x;
    Float tmp = math::safe_sqrt(1.0f - z*z);

    Float sinPhi, cosPhi;
    math::sincos(2.0f * M_PI * sample.y, &sinPhi, &cosPhi);

    return Vector(cosPhi * tmp, sinPhi * tmp, z);
}

Vector squareToCosineHemisphere(const Point2 &sample) {
    Point2 p = squareToUniformDiskConcentric(sample);
    Float z = math::safe_sqrt(1.0f - p.x*p.x - p.y*p.y);

    /* Guard against numerical imprecisions */
    if (EXPECT_NOT_TAKEN(z == 0))
        z = 1e-10f;

    return Vector(p.x, p.y, z);
}

Vector squareToUniformCone(Float cosCutoff, const Point2 &sample) {
    Float cosTheta = (1-sample.x) + sample.x * cosCutoff;
    Float sinTheta = math::safe_sqrt(1.0f - cosTheta * cosTheta);

    Float sinPhi, cosPhi;
    math::sincos(2.0f * M_PI * sample.y, &sinPhi, &cosPhi);

    return Vector(cosPhi * sinTheta,
        sinPhi * sinTheta, cosTheta);
}

Point2 squareToUniformDisk(const Point2 &sample) {
    Float r = std::sqrt(sample.x);
    Float sinPhi, cosPhi;
    math::sincos(2.0f * M_PI * sample.y, &sinPhi, &cosPhi);

    return Point2(
        cosPhi * r,
        sinPhi * r
    );
}

Point2 squareToUniformTriangle(const Point2 &sample) {
    Float a = math::safe_sqrt(1.0f - sample.x);
    return Point2(1 - a, a * sample.y);
}

Point2 squareToUniformDiskConcentric(const Point2 &sample) {
    Float r1 = 2.0f*sample.x - 1.0f;
    Float r2 = 2.0f*sample.y - 1.0f;

    /* Modified concencric map code with less branching (by Dave Cline), see
       http://psgraphics.blogspot.ch/2011/01/improved-code-for-concentric-map.html */
    Float phi, r;
    if (r1 == 0 && r2 == 0) {
        r = phi = 0;
    } else if (r1*r1 > r2*r2) {
        r = r1;
        phi = (M_PI/4.0f) * (r2/r1);
    } else {
        r = r2;
        phi = (M_PI/2.0f) - (r1/r2) * (M_PI/4.0f);
    }

    Float cosPhi, sinPhi;
    math::sincos(phi, &sinPhi, &cosPhi);

    return Point2(r * cosPhi, r * sinPhi);
}

Point2 uniformDiskToSquareConcentric(const Point2 &p) {
    Float r   = std::sqrt(p.x * p.x + p.y * p.y),
          phi = std::atan2(p.y, p.x),
          a, b;

    if (phi < -M_PI/4) {
        /* in range [-pi/4,7pi/4] */
        phi += 2*M_PI;
    }

    if (phi < M_PI/4) { /* region 1 */
        a = r;
        b = phi * a / (M_PI/4);
    } else if (phi < 3*M_PI/4) { /* region 2 */
        b = r;
        a = -(phi - M_PI/2) * b / (M_PI/4);
    } else if (phi < 5*M_PI/4) { /* region 3 */
        a = -r;
        b = (phi - M_PI) * a / (M_PI/4);
    } else { /* region 4 */
        b = -r;
        a = -(phi - 3*M_PI/2) * b / (M_PI/4);
    }

    return Point2(0.5f * (a+1), 0.5f * (b+1));
}

Point2 squareToStdNormal(const Point2 &sample) {
    Float r   = std::sqrt(-2 * math::fastlog(1-sample.x)),
          phi = 2 * M_PI * sample.y;
    Point2 result;
    math::sincos(phi, &result.y, &result.x);
    return result * r;
}

Float lineToStdNormalPdf(Float pos) {
    return INV_SQRT_TWOPI * math::fastexp(-pos*pos/2.0f);
}

Float squareToStdNormalPdf(const Point2 &pos) {
    return INV_TWOPI * math::fastexp(-(pos.x*pos.x + pos.y*pos.y)/2.0f);
}

Float cubeToStdNormalPdf(const Point3 &pos) {
    return INV_SQRT_TWOPI_CUBED * math::fastexp(
            -(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z)/2.0f);
}

static Float intervalToTent(Float sample) {
    Float sign;

    if (sample < 0.5f) {
        sign = 1;
        sample *= 2;
    } else {
        sign = -1;
        sample = 2 * (sample - 0.5f);
    }

    return sign * (1 - std::sqrt(sample));
}

Point2 squareToTent(const Point2 &sample) {
    return Point2(
        intervalToTent(sample.x),
        intervalToTent(sample.y)
    );
}

Float intervalToNonuniformTent(Float a, Float b, Float c, Float sample) {
    Float factor;

    if (sample * (c-a) < b-a) {
        factor = a-b;
        sample *= (a-c)/(a-b);
    } else {
        factor = c-b;
        sample = (a-c)/(b-c) * (sample - (a-b)/(a-c));
    }

    return b + factor * (1-math::safe_sqrt(sample));
}

static void uniformToTruncatedExponentialImpl(Float original_lambda,
        Float original_lo, Float original_hi, const Float *u_, Float &original_x, Float *pdf) {
    Float u, x;
    if (u_) {
        u = *u_;
        SAssert(u >= 0 && u <= 1);
    } else {
        u = 0.0f/0.0f;
    }
    Float thePdf, lambda, lo, hi;
    if (original_lambda > 0) {
        lambda = original_lambda;
        lo = original_lo;
        hi = original_hi;
        x = original_x;
    } else {
        lambda = -original_lambda; // make it positive
        lo = -original_hi;
        hi = -original_lo;
        x = -original_x;
    }

    if ((hi - lo)*lambda < Epsilon) {
        /* Expansion in small lambda, up to second order. Expansion is
         * neccesary to avoid catastrophic cancellation. */
        Float d = hi - lo;
        Float d2 = d*d;
        Float d3 = d*d2;
        if (u_) {
            /* The expanded x is still guaranteed to stay within bounds, but
             * clamp to protect against roundoff */
            x = math::clamp(lo + d*u - 0.5f*u*(u-1.f)*d2*lambda
                    + 1.f/6.f*(2.f*u-1.f)*(u-1.f)*u*d3*lambda*lambda, lo, hi);
        }
        /* The expanded thePdf is still guaranteed to be >= 0 */
        thePdf = (1 + 0.5*(2*x - lo - hi) * lambda
                + 1./12.*(math::square(hi) + math::square(lo)
                        + 4*lo*hi
                        + 6*x*(x - lo - hi))
                                * lambda*lambda) / d;
    } else if (lambda*hi > LOG_REDUCED_PRECISION / 2) {
        /* Expansion in large lambda */

        // TODO: code below implicitly assumes that (hi-lo)*lambda is LARGE!!
        // if that is not the case, then the warning below could be triggered... FIXME!

        if (u_) {
            x = math::clamp(hi + std::log(u) / lambda,
                    lo, hi);
            if (x < lo) {
                /* *INSANELY* unlikely (pdf below would probably cut off to
                 * zero anyway, but universe would die of heat death first) */
                SLog(EWarn, "Woah! Universe should have encountered heat death, "
                        "or code is bugged -- x: %f < lo %f", x, lo);
                x = lo;
            }
        }
        thePdf = lambda * exp(lambda*(x - hi));
        if (!std::isfinite(thePdf) || thePdf < 0 || (u_ && thePdf <= 0))
            SLog(EWarn, "Something fishy happened, pdf %f, x: %f (min %f max %f lambda %f)", thePdf, x, lo, hi, lambda);
    } else {
        if (u_) {
            x = math::clamp(std::log((1.f-u)*math::fastexp(lo*lambda) + u*math::fastexp(hi*lambda)) / lambda,
                    lo, hi);
        }
        thePdf = lambda/(exp(hi*lambda) - exp(lo*lambda)) * exp(lambda * x);
    }

    if (thePdf < 0) {
        SLog(EWarn, "Got negative pdf! %e", thePdf);
        thePdf = 0;
    }
    if (pdf)
        *pdf = thePdf;

    if (u_) {
        if (original_lambda < 0)
            original_x = -x;
        else
            original_x = x;
    }

    SAssert(original_lo <= original_x);
    SAssert(original_x <= original_hi);
}

Float uniformToTruncatedExponential(Float lambda,
        Float lo, Float hi, Float u, Float *pdf) {
    Float z;
    uniformToTruncatedExponentialImpl(lambda, lo, hi, &u, z, pdf);
    return z;
}

Float uniformToTruncatedExponentialPdf(Float lambda,
        Float lo, Float hi, Float z) {
    Float pdf;
    uniformToTruncatedExponentialImpl(lambda, lo, hi, NULL, z, &pdf);
    return pdf;
}


};

MTS_NAMESPACE_END
