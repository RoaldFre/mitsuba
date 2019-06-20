#include "fwdscat.h"
#include "dipoleUtil.h"

#include <mitsuba/core/warp.h>
#include <mitsuba/render/truncnorm.h>

#include <boost/math/tools/roots.hpp>
#include <boost/math/special_functions/erf.hpp>

MTS_NAMESPACE_BEGIN

/* Reject incoming directions that come from within the actual geometry
 * (i.e. w.r.t. the actual local normal at the incoming point instead of,
 * for instance, the modified tangent plane normal)? */
#define MTS_FWDSCAT_DIPOLE_REJECT_INCOMING_WRT_TRUE_SURFACE_NORMAL true

#define MTS_FWDSCAT_GIVE_REAL_AND_VIRTUAL_SOURCE_EQUAL_SAMPLING_WEIGHT false

static constexpr Float directionSampler_origWeight = 0.5; // TODO

/* Sample the dipole direction as a simple cosine weighted hemisphere with
 * this weight. This improves robustness in case we would severely
 * undersample the transport with the dedicated importance samplers (e.g.
 * by underestimating the width of a sharp peak). */
static constexpr Float directionSampler_dipoleHemiWeight = 0.05;

#ifdef MTS_FWDSCAT_DEBUG
# define FSAssert(x)      Assert(x)
# define FSAssertWarn(x)  AssertWarn(x)
# define SFSAssert(x)     SAssert(x)
# define SFSAssertWarn(x) SAssertWarn(x)
#else /* This removes the side-effects from the functions! */
# define FSAssert(x)      ((void) 0)
# define FSAssertWarn(x)  ((void) 0)
# define SFSAssert(x)     ((void) 0)
# define SFSAssertWarn(x) ((void) 0)
#endif

FINLINE Float _reducePrecisionForCosTheta(Float x) {
    /* Turns out not to help too much -- or even make things worse! So
     * don't round. TODO: Test some more at some point... */
    return x;
    //return roundFloatForStability(x);
    //return roundToSignificantDigits(x, 3);
}

FINLINE void roundCosThetaBoundsForStability(
        Float &minCosTheta, Float &maxCosTheta) {
    minCosTheta = _reducePrecisionForCosTheta(minCosTheta);
    maxCosTheta = _reducePrecisionForCosTheta(maxCosTheta);
}
FINLINE Float roundCosThetaForStability(Float cosTheta,
        Float minCosTheta, Float maxCosTheta) {
    cosTheta = math::clamp(cosTheta, minCosTheta, maxCosTheta);
    return _reducePrecisionForCosTheta(cosTheta);
}



/// Already has the 3/2 winkeler correction and assumes D*(1-dot(u0,uL)) (and 'C=0') instead of -D*dot(u0*uL) (and nonzero C to keep N from exploding)
FINLINE double FwdScat::absorptionAndNormalizationConstant2(Float theLength) const {
    const double sig = sqrt(1.5) * sigma_s * mu; // \bar{sigma}
    const double ss = sig * theLength; // \bar{sigma} * s
    double result;
    if (ss < 0.04) {
        // protect against overflows in the exp()'s
        const double c0 = 18.*sqrt(3.)*std::pow(6.,0.25);
        const double c1 = 33.*sqrt(3.)*pow(6.,0.75)/10.;
        const double c2 = 303.*sqrt(3.)*pow(6.,0.25)/200.;
        const double c3 = -25483.*sqrt(3.)*pow(6.,0.75)/84000.;
        result = 0.25 / std::pow(M_PI_DBL, 2.5) * exp(-sigma_a*theLength)
                * std::pow(ss, -11./2) * (c0 + c1*ss + c2*ss*ss + c3*ss*ss*ss);
        result *= sig*sig*sig; // from \bar{sigma}=1 back to the real value
    } else {
        double C, D, E, F, CminD, Z;
        calcValues2(theLength, C, D, E, F, &CminD, &Z);
        double ZoverExpMinOne; // = Z / (exp(Z) - 1)
        if (Z < 0.01) {
            // small Z corresponds to limit of large ss
            ZoverExpMinOne = 1. + 0.5*Z + 1./12.*Z*Z;
        } else {
            ZoverExpMinOne = Z / (exp(Z) - 1);
        }
        result = 0.25 / std::pow(M_PI_DBL, 2.5) * exp(CminD - sigma_a*theLength)
                * sqrt(F) * F * ZoverExpMinOne;
    }

#ifdef MTS_FWDSCAT_DEBUG
    if (!std::isfinite(result) || result < 0) {
        Log(EWarn, "problem with analytical normalization at ss %e: %e",
                sqrt(1.5) * sigma_s * mu * theLength, result);
    }
#endif
    FSAssert(result >= 0);

    return result;
}

/// XXX the limits have a wrong constant scaling ... (compare with v2 for m_winklerCorrection)
FINLINE double FwdScat::absorptionAndNormalizationConstant(Float theLength) const {
    double p = 0.5 * sigma_s * mu;
    if (m_winklerCorrection) {
#if 1
        return absorptionAndNormalizationConstant2(theLength);
#else
        // or 'hack' to use our code below:
        p *= sqrt(1.5); // TODO to counter typo in Winkeler formula... -- XXX expansions below still inconsistent!
#endif
    }
    double ps = p * theLength;

    double result;
    if (ps < 0.03) {
        double ps2 = ps*ps;
        double ps3 = ps2*ps;
        result = sqrt(2.0) * std::pow(M_PI_DBL, -2.5)
                * std::pow(ps, -11./2.) * (
                    81./32. + 243./64.*ps + 3429./1280.*ps2 - 243./2560.*ps3);
        result *= p*p*p; // from p=1 back to the real p value;
        result *= math::fastexp(-sigma_a*theLength);
    } else if (ps > 11) { // quickly to avoid cancellation in denominator of full expression!
        double psi1 = 1.0/ps;
        double psi2 = psi1*psi1;
        double psi3 = psi2*psi1;
        double psi4 = psi3*psi1;
        result = std::exp(1.5)*sqrt(6.0)*3./2048. * std::pow(M_PI_DBL, -2.5)
                * std::pow(ps, -3./2.) * (
                    128 + 129*psi1 + 240*psi2 + 280*psi3 + 315*psi4);
        result *= p*p*p; // from p=1 back to the real p value;
        result *= math::fastexp(-sigma_a*theLength);
        result *= 0.25; // XXX this factor was missing for some time!
    } else {
        double C, D, E, F;
        calcValues(theLength, C, D, E, F);
        double denomExpArg = E*E/F - 2*D; /* cancellation problem, so quickly go to large ps
                                             expansion TODO: rewrite in terms of A&B to avoid
                                             cancellation */
        double denom = math::abs(denomExpArg) > 1e-3
                ? exp(denomExpArg) - 1
                : denomExpArg*(1 + denomExpArg*(0.5 + denomExpArg*1./6.));
        result = 0.25 / std::pow(M_PI_DBL, 2.5)
                * sqrt(F) * (E*E - 2*D*F) * exp(C - D - sigma_a*theLength)
                / denom;
#ifdef MTS_FWDSCAT_DEBUG
        FSAssertWarn(std::isfinite(exp(D + C)));
        FSAssertWarn(std::isfinite(exp(2*D)));
        FSAssertWarn(std::isfinite(exp(E*E/F)));
        // denom
        // XXX disabled for now, as this is on the edge and triggers often.....
        //CancellationCheck(E*E/F, -2*D); /* need to switch to large ps expansion
        //                                   sufficiently fast, see above */
        // exp in result:
        CancellationCheck(C, -D);
#endif
    }

#ifdef MTS_FWDSCAT_DEBUG
    if (!std::isfinite(result)) {
        Log(EWarn, "problem with analytical normalization at ps %e: %e",
                ps, result);
    }
#endif
    FSAssert(result >= 0);

    return result;
}

FINLINE void FwdScat::calcValues(double length, double &C, double &D, double &E, double &F) const {
    FSAssert(length >= 0);
    FSAssert(mu > 0 && mu <= 1);
    FSAssert(sigma_s > 0);
    FSAssert(length >= 0);

    double p = 0.5 * mu * sigma_s;
    if (m_winklerCorrection) {
#if 0
        calcValues2(length, C, D, E, F);
        return;
#else
        // hack to get the old code below to work with the correction
        p *= sqrt(1.5); // TODO to counter typo in Winkeler formula...
#endif
    }
    double s = length;
    double ps = p*s;

    /* TODO/NOTE: C is independent of R,u0,uL eventually, so purely
     * a normalization problem!
     * We could drop C, but that has the effect of exploding the
     * normalization constant exponential beyond double precision range for
     * small p*s. So we currently keep it as a help for numerical stability.
     * (Maybe a reduced form leads to something that is still managable and
     * allows for a simpler normalization function fit?) Either way, the
     * nu0 term in C is simply dropped */

    if (ps < 0.001) {
        /* Expansion accurate up to a range of 6 orders of ps */
        C = 3./ps; // + 0.4*ps - 11./525.*ps*ps*ps;
        D = 1.5/ps - 0.1*ps + 13./1050.*ps*ps*ps;
        E = p * (4.5/(ps*ps) + 0.3 - 3./350*ps*ps);
        F = p*p * (4.5/(ps*ps*ps) + 1.8/ps - 3./350*ps);
    } else if (ps > 1.0/0.001) {
        /* Expansion accurate up to a range of 'all' orders of 1/ps (exact
         * geometric series) */
        double series = 1.0 / (ps - 1.0); // = 1/ps + 1/ps^2 + 1/ps^3 + ...
        C = 3./ps; //1.5 + 0.75 * series;
        D = 0.75 * series;
        E = p * 1.5 * series;
        F = p*p * 1.5 * series;
    } else {
        /* Exact solutions, in a ps range that is safe from numerical problems */
        double TH = tanh(p*s);
        double SH = sinh(2*p*s);
        double TH2 = tanh(2*p*s); /* Note: SH/CH with CH = sqrt(1 + SH*SH) is
                                     unstable for large arguments (inf/inf) */

        double A=1/(s/p - TH/(p*p));
        double B=TH/(2*p);

        CancellationCheck(3*A*B*B,  3/(2*TH2)); // C
        CancellationCheck(3*A*B*B, -3/(2*SH));  // D
        C=3/ps; //3*A*B*B + 3/(2*TH2);
        D=3*A*B*B - 3/(2*SH);
        E=3*A*B;
        F=3*A/2;
    }

    if (m_winklerCorrection) {
        // TODO to counter typo in Winkeler formula...
        C *= sqrt(2./3.);
        D *= sqrt(2./3.);
        E *= sqrt(2./3.);
        F *= sqrt(2./3.);


        double ss = sqrt(1.5) * sigma_s * mu * s; // \bar{sigma} * s
        double D_exp, E_exp, F_exp;
        if (ss < 1e-3) {
            D_exp =   sqrt(1.5)   * 2/ss - 1./30*ss + 13./12600*ss*ss*ss;
            E_exp =   sqrt(1.5)   * 6/ss + 0.1*ss - 1./1400*ss*ss*ss;
            F_exp = 0.5*sqrt(1.5) * 12/ss + 6./5.*ss - 1./700*ss*ss*ss;
        } else if (ss > 1e3) {
            double series = 0.5 / (0.5*ss - 1);
            D_exp =   sqrt(1.5)         * series;
            E_exp =   sqrt(1.5)   * ss  * series;
            F_exp = 0.5*sqrt(1.5)*ss*ss * series;
        } else {
            double t = exp(-ss);
            double t2 = t*t;
            D_exp = sqrt(1.5) * (1 - t2 - 2*ss*t) / (ss - 2 + 4*t - (ss + 2)*t2);
            E_exp =     sqrt(1.5) * ss  * (1 - t) / (ss - 2 + (ss + 2)*t);
            F_exp = 0.5*sqrt(1.5)*ss*ss * (1 + t) / (ss - 2 + (ss + 2)*t);
        }
        // We are dimensionful here (e.g. displacement vector R instead of r)
        E_exp /= s;
        F_exp /= s*s;


        if (false) {
            if (math::abs((D - D_exp)/(D + D_exp)) > 1e-4) {
                cout << "ss: "<< ss << endl;
                cout << "D: " << D / D_exp << endl;
                cout << "D: " << D << " " << D_exp << endl;
            }
            if (math::abs((E - E_exp)/(E + E_exp)) > 1e-4) {
                cout << "ss: "<< ss << endl;
                cout << "E: " << E / E_exp << endl;
                cout << "E: " << E << " " << E_exp << endl;
            }
            if (math::abs((F - F_exp)/(F + F_exp)) > 1e-4) {
                cout << "ss: "<< ss << endl;
                cout << "F: " << F / F_exp << endl;
                cout << "F: " << F << " " << F_exp << endl;
            }
        }
    }

    FSAssert(C >= 0);
    FSAssert(D >= 0);
    FSAssert(E >= 0);
    FSAssert(F >= 0);
}

/**
 * Already contains the 3/2 winkler correction
 * CminD = C - D
 * Z = E^2/F - 2*D (which is > 0)
 */
FINLINE void FwdScat::calcValues2(double length, double &C, double &D, double &E, double &F,
        double *CminDptr, double *Zptr) const {
    FSAssert(length >= 0);
    FSAssert(mu > 0 && mu <= 1);
    FSAssert(sigma_s > 0);
    FSAssert(length >= 0);

    double s = length;
    double ss = sqrt(1.5) * sigma_s * mu * s; // \bar{sigma} * s
    double t = exp(-ss);
    double t2 = t*t;
    double Z, CminD;
    C = 2 * sqrt(6.) / ss;
    if (ss < 0.3) {
        double ss3 = ss*ss*ss;
        double ss5 = ss3*ss*ss;
        D =   sqrt(1.5)   * (2/ss  - 1./30*ss + 13./12600*ss3);
        E =   sqrt(1.5)   * (6/ss  +  0.1*ss - 1./1400*ss3);
        F = 0.5*sqrt(1.5) * (12/ss + 6./5*ss -  1./700*ss3);
        Z =    sqrt(6)    * (1/ss  - 1./6*ss +  7./360*ss3 + 31./15120*ss5);
    } else if (ss > 30) {
        double series = 0.5 / (0.5*ss - 1);
        D =   sqrt(1.5)         * series;
        E =   sqrt(1.5)   * ss  * series;
        F = 0.5*sqrt(1.5)*ss*ss * series;
        Z = 2*sqrt(6.)  *  t / (1 - t2); // full expression is stable for large ss
    } else {
        // TODO: Some kind of cheaper approximation? (Chebychev? Rational?)
        D = sqrt(1.5) * (1 - t2 - 2*ss*t) / (ss - 2 + 4*t - (ss + 2)*t2);
        E =     sqrt(1.5) * ss  * (1 - t) / (ss - 2 + (ss + 2)*t);
        F = 0.5*sqrt(1.5)*ss*ss * (1 + t) / (ss - 2 + (ss + 2)*t);
        Z = 2*sqrt(6.)  *  t / (1 - t2);
    }
    CminD = C - D; // stable throughout the range
    if (CminDptr)
        *CminDptr = CminD;
    if (Zptr)
        *Zptr = Z;

    // We are dimensionful here (e.g. displacement vector R instead of r):
    E /= s;
    F /= s*s;

    FSAssert(D >= 0);
    FSAssert(E >= 0);
    FSAssert(F >= 0);
    FSAssert(CminD >= 0);
    FSAssert(Z >= 0);

#if 0
    double Ccheck, Dcheck, Echeck, Fcheck;
    calcValues(length, Ccheck, Dcheck, Echeck, Fcheck);
    //SLog(EInfo, "%f %f %f %f", C/Ccheck, D/Dcheck, E/Echeck, F/Fcheck);
    //SLog(EInfo, "%f", Z/(Echeck*Echeck/Fcheck-2*Dcheck));
    //SLog(EInfo, "%f", CminD/(Ccheck-Dcheck));
#endif
}



/// if rejectInternalIncoming is requested: returns false if we should stop
FINLINE bool FwdScat::getVirtualDipoleSource(
        Normal n0, Vector u0,
        Normal nL, Vector uL,
        Vector R, Float length,
        bool rejectInternalIncoming,
        TangentPlaneMode tangentMode,
        ZvMode zvMode,
        Vector &u0_virt, Vector &R_virt,
        Vector *optional_n0_effective) const {
    Normal n0_effective;
    switch (tangentMode) {
    case EFrisvadEtAl:
        /* Use the modified tangent plane of the directional dipole model
         * of Frisvad et al */
        if (R.length() == 0) {
            n0_effective = n0;
        } else {
            if (cross(n0,R).length() == 0)
                return false;
            n0_effective = cross(normalize(R), normalize(cross(n0, R)));
            FSAssert(dot(n0_effective, n0) > -Epsilon);
        }
        break;
    case EFrisvadEtAlWithMeanNormal: {
        /* Like the tangent plane of Frisvad et al, but based on an
         * 'average' normal at incoming and outgoing point instead of on
         * the incoming normal. This should immediately give reciprocity as
         * a bonus. */
        Vector sumNormal = n0 + nL;
        if (R.length() == 0) {
            n0_effective = n0;
        } else {
            if (cross(sumNormal,R).length() == 0)
                return false;
            n0_effective = cross(normalize(R), normalize(cross(sumNormal, R)));
        }
        break; }
    case EUnmodifiedIncoming:
        n0_effective = n0; break;
    case EUnmodifiedOutgoing:
        n0_effective = nL; break;
    default:
        Log(EError, "Unknown tangentMode: %d", tangentMode);
        return false; // keep compiler happy
    }
    if (!n0_effective.isFinite()) {
        Log(EWarn, "Non-finite n0_effective: %s", n0_effective.toString().c_str());
        return false;
    }

    if (rejectInternalIncoming && dot(n0_effective, u0) > 0)
        return false;

    FSAssert(math::abs(n0_effective.length() - 1) < Epsilon);

    Float zv;
    Float sigma_sp = sigma_s * mu;
    Float sigma_tp = sigma_sp + sigma_a;


    switch (zvMode) {
    case EFrisvadEtAlZv: {
        if (sigma_tp == 0 || sigma_sp == 0)
            return false;
        Float D = 1./(3.*sigma_tp);
        Float alpha_p = sigma_sp / sigma_tp;
        Float d_e = 2.131 * D / sqrt(alpha_p);
        Float A = dEon_A(m_eta);
        zv = 2*A*d_e;
        break; }
    case EBetterDipoleZv: {
        if (sigma_tp == 0)
            return false;
        Float D = (2*sigma_a + sigma_sp)/(3*math::square(sigma_tp));
        Float A = dEon_A(m_eta);
        zv = 4*A*D;
        break; }
    case EClassicDiffusion: {
        if (sigma_tp == 0)
            return false;
        Float Fdr = fresnelDiffuseReflectance(1 / m_eta);
        Float A = (1 + Fdr) / (1 - Fdr);
        Float D = 1./(3*sigma_tp);
        zv = 4*A*D;
        break; }
    default:
        Log(EError, "Unknown VirtSourceHeight mode %d", zvMode);
        return false;
    }

    /* If not rejectInternalIncoming -> virtual source will point *INTO*
     * the half space!! (and 'cross' the actual real source "beam" if we
     * elongate it).
     * Maybe flip the normal? (to get the half space on the other side...) */
    R_virt = R - zv * n0_effective;
    u0_virt = u0  -  2*dot(n0_effective, u0) * n0_effective;
    if (optional_n0_effective)
        *optional_n0_effective = n0_effective;
    return true;
}

FINLINE bool FwdScat::getTentativeIndexMatchedVirtualSourceDisp(
        Normal n0,
        Normal nL, Vector uL,
        Vector R,
        Float s, // not always required
        TangentPlaneMode tangentMode,
        Vector &R_virt,
        Vector *optional_n0_effective,
        Float *optional_realSourceRelativeWeight) const {
    Vector _u0_virt, n0_effective;
    Vector _u0(0.0f/0.0f);
    bool rejectInternalIncoming = false; //u0 not sensible yet!
    ZvMode zvMode = EClassicDiffusion; //only one that does not depend on u0
    if (!getVirtualDipoleSource(n0, _u0, nL, uL, R, s,
            rejectInternalIncoming, tangentMode, zvMode,
            _u0_virt, R_virt, &n0_effective)) {
        return false; // Won't be able to evaluate bssrdf transport anyway!
    } else {
        FSAssert(R_virt.isFinite());
    }
    if (optional_n0_effective)
        *optional_n0_effective = n0_effective;
    if (!optional_realSourceRelativeWeight)
        return true;
    double C, D, E, F;
    calcValues(s, C, D, E, F);
    double ratio = exp(E*dot(R-R_virt,uL) - F*(R.lengthSquared()-R_virt.lengthSquared()));
    Float realSourceWeight = (std::isinf(ratio + 1) ? 1.0 : ratio/(ratio + 1));
    // TODO: clamp the extremes of 0 and 1 to something slightly more 'centered'?
    FSAssert(realSourceWeight >= 0 && realSourceWeight <= 1);
#if MTS_FWDSCAT_GIVE_REAL_AND_VIRTUAL_SOURCE_EQUAL_SAMPLING_WEIGHT
    *optional_realSourceRelativeWeight = 0.5;
#else
    *optional_realSourceRelativeWeight = realSourceWeight;
#endif
    return true;
}




FINLINE Float FwdScat::evalDipole(
        Normal n0, Vector u0_external,
        Normal nL, Vector uL_external,
        Vector R, Float length,
        bool rejectInternalIncoming,
        bool reciprocal,
        TangentPlaneMode tangentMode,
        ZvMode zvMode,
        bool useEffectiveBRDF,
        DipoleMode dipoleMode) const {

    /* If reciprocal is requested, nL should be finite and uL_external should point
     * along nL. */
    FSAssert(!reciprocal || nL.isFinite());
    FSAssert(!reciprocal || dot(uL_external,nL) >= -Epsilon); // positive with small margin for roundoff errors
    if (nL.isFinite() && dot(uL_external,nL) <= 0) // clamp to protect against roundoff errors
        return 0.0f;

#if MTS_FWDSCAT_DIPOLE_REJECT_INCOMING_WRT_TRUE_SURFACE_NORMAL
    if (dot(u0_external, n0) >= 0)
        return 0.0f;
#endif


    /* Handle eta != 1 case by 'refracting' the 'external' directions
     * u0_external and uL_external to 'internal' directions u0 and uL. We
     * keep the directions pointing along the propagation direction of
     * light (i.e. not the typical refract as in BSDFs, for instance, which
     * flips to the other side of the boundary). */
    Float _cosThetaT, F0, FL;
    Vector u0 = refract(-u0_external, n0, m_eta, _cosThetaT, F0);
    Vector uL = -refract(uL_external, nL, m_eta, _cosThetaT, FL);
    Float fresnelTransmittance = (1-F0)*(1-FL);

    if (m_eta == 1)
        FSAssert(u0 == u0_external  &&  uL == uL_external);

    if (u0.isZero() || uL.isZero()) {
        if (m_eta > 1)
            Log(EWarn, "Could not refract, which is weird because we have a "
                    "higher ior! (eta=%f)", m_eta);
        return 0.0f;
    }


    Vector R_virt;
    Vector u0_virt;
    if (!getVirtualDipoleSource(n0, u0, nL, uL, R, length,
            rejectInternalIncoming, tangentMode, zvMode,
            u0_virt, R_virt))
        return 0.0f;

    // Effective BRDF?
    if (useEffectiveBRDF) {
        FSAssert((n0 - nL).length() < Epsilon); // same point -> same normal
        Float Rv_z = dot(R_virt, nL);
#ifdef MTS_FWDSCAT_DEBUG
        Float lRvl = R_virt.length();
        FSAssert((n0 - nL).length() < Epsilon); // same point -> same normal
        FSAssert(Rv_z <= 0); // pointing from virtual point towards xL -> into medium
        // the only displacement should be in the normal direction:
        FSAssertWarn(lRvl == 0 || math::abs((lRvl - math::abs(Rv_z))/lRvl) < Epsilon);
#endif

        return fresnelTransmittance * (
                  evalPlaneSource(u0,      uL, nL, 0.0f, length)
                - evalPlaneSource(u0_virt, uL, nL, Rv_z, length));
    }

    // Full BSSRDF
    Float real = 0, virt = 0;
    if (dipoleMode & EReal)
        real = evalMonopole(u0,      uL, R,      length);
    if (dipoleMode & EVirt)
        virt = evalMonopole(u0_virt, uL, R_virt, length);
    Float transport;
    switch (dipoleMode) {
        case ERealAndVirt: transport = real - virt; break;
        case EReal:        transport = real; break;
        case EVirt:        transport = virt; break; // note: positive sign
        default: Log(EError, "Unknown dipoleMode: %d", dipoleMode); return 0;
    }
    if (reciprocal) {
        Float transportRev = evalDipole(
                nL, -uL, n0, -u0, -R, length,
                rejectInternalIncoming, false,
                tangentMode, zvMode, useEffectiveBRDF, dipoleMode);
        return 0.5 * (transport + transportRev) * fresnelTransmittance;
    } else {
        return transport * fresnelTransmittance;
    }
}



FINLINE Float FwdScat::evalMonopole(Vector u0, Vector uL, Vector R, Float length) const {
    FSAssert(math::abs(u0.length() - 1) < 1e-6);
    FSAssert(math::abs(uL.length() - 1) < 1e-6);
    
    double C, D, E, F;
    calcValues(length, C, D, E, F);

    /* We regularized the sampling of u0, so we should be consistent here.
     * NOTE: but E can still blow up in the final expression for G (TODO
     * does this happen?) */
    Vector3d H = E*Vector3d(R) - D*Vector3d(uL);
    double lHl = H.length();
    Vector Hnorm = Vector(H / lHl);
    Float lHlreg = (lHl > 1./MTS_FWDSCAT_DIRECTION_MIN_MU) ?
            1./MTS_FWDSCAT_DIRECTION_MIN_MU : lHl;
    Float cosTheta = roundCosThetaForStability(dot(u0, Hnorm), -1, 1);

    double N = absorptionAndNormalizationConstant(length); // XXX DEBUG
    double G = N * exp(-C + E*dot(R,uL) + lHlreg*cosTheta - F*R.lengthSquared());
    //Non-regularized:
    //G = N * exp(-C - D*dot(u0,uL) + E*(dot(R,u0) + dot(R,uL)) - F*R.lengthSquared());

    // Note: fastmath compiler flags may change the order of the operations...
    /* We only care for cancellations if the result is sufficiently large
     * (otherwise exp(epsilon) ~= 1 anyway) */
    if (math::abs(E*dot(R,uL)) > 1e3)
        CancellationCheck(-C,  E*dot(R,uL));
    if (math::abs(lHlreg*cosTheta) > 1e3)
        CancellationCheck(-C + E*dot(R,uL),  lHlreg*cosTheta);
    if (math::abs(F*R.lengthSquared()) > 1e3)
        CancellationCheck(-C + E*dot(R,uL) + lHlreg*cosTheta, -F*R.lengthSquared());

#ifdef MTS_FWDSCAT_DEBUG
    if (!std::isfinite(G) || G < 0) {
        Log(EWarn, "Invalid G in evalMonopole(): "
                "%e; ss %e C %e D %e E %e F %e Rsq %e u0dotuL %e\n"
                "%e %e %e %e %e\n"
                "%e %e",
                G, length*sqrt(1.5)*sigma_s*mu, C, D, E, F, R.lengthSquared(), dot(u0,uL),

                N, -C, E*dot(R,uL), lHlreg*cosTheta, -F*R.lengthSquared(),

                -C + E*dot(R,uL) + lHlreg*cosTheta - F*R.lengthSquared(),
                exp(-C + E*dot(R,uL) + lHlreg*cosTheta - F*R.lengthSquared()));
        return 0;
    }
#endif
    return G;
}

FINLINE Float FwdScat::evalPlaneSource(Vector u0, Vector uL,
        Vector n, Float Rz, Float length) const {
    FSAssert(math::abs(u0.length() - 1) < 1e-6);
    FSAssert(math::abs(uL.length() - 1) < 1e-6);

    double C, D, E, F;
    calcValues(length, C, D, E, F);

    Float u0z = dot(u0,n);
    Float uLz = dot(uL,n);

    double result = absorptionAndNormalizationConstant(length)
            * M_PI_DBL / F * exp(
                E*E/4/F*(2 + 2*dot(u0,uL) - math::square(u0z + uLz))
                - D*dot(u0,uL)
                - C
                + E*Rz * (u0z + uLz)
                - F*Rz*Rz);

    if (!std::isfinite(result)) {
        Log(EWarn, "non-finite result %lf", result);
        return 0;
    }
    return result;
}



// Strategy weights, must sum to one
static constexpr Float lengthSample_w1 = 0.5; /* short length limit */
static constexpr Float lengthSample_w2 = 0.5; /* long length limit */
static constexpr Float lengthSample_w3 = 0.0; /* absorption */

// If d_in is unknown, it is set to NULL
FINLINE Float FwdScat::sampleLengthDipole(
        const Vector &uL, const Vector &nL, const Vector &R,
        const Vector *u0, const Vector &n0,
        TangentPlaneMode tangentMode, Float &s, Sampler *sampler) const {

    Vector R_virt;
    if (!getTentativeIndexMatchedVirtualSourceDisp(
            n0, nL, uL, R, 0./0., tangentMode, R_virt))
        return 0.0;

    /* For R-dependent functions that don't take the dipole into account
     * themselves.
     * TODO: Smart MIS weight? (Need length-marginalized 'realSourceWeight'
     * from getTentativeIndexMatchedVirtualSourceDisp then.) */
    Vector R_effective, R_other;
    if (sampler->next1D() < 0.5) {
        R_effective = R;
        R_other = R_virt;
    } else {
        R_effective = R_virt;
        R_other = R;
    }

    Float p1, p2, p3;
    p1 = p2 = p3 = -1;
    const Float u = sampler->next1D();
    if (u < lengthSample_w1) {
        p1 = sampleLengthShortLimit(R, u0, uL, s, sampler);
        if (p1 == 0)
            return 0.0f;
    } else if (u < lengthSample_w1 + lengthSample_w2) {
        p2 = sampleLengthLongLimit(R_effective, uL, s, sampler);
        if (p2 == 0)
            return 0.0f;
    } else if (u < lengthSample_w1 + lengthSample_w2 + lengthSample_w3) {
        p3 = sampleLengthAbsorption(s, sampler);
        if (p3 == 0)
            return 0.0f;
    }

    if (p1 == -1)
        p1 = (lengthSample_w1 == 0 ? 0 : pdfLengthShortLimit(R, u0, uL, s));
    if (p2 == -1)
        p2 = (lengthSample_w2 == 0 ? 0 : pdfLengthLongLimit(R_effective, uL, s));
    if (p3 == -1)
        p3 = (lengthSample_w3 == 0 ? 0 : pdfLengthAbsorption(s));

    // Handle the MIS probabilities of having sampled based on R_other
    if (lengthSample_w2 != 0)
        p2 = 0.5 * (p2 + pdfLengthLongLimit(R_other, uL, s));

    return 1.0 / (lengthSample_w1 * p1
                + lengthSample_w2 * p2
                + lengthSample_w3 * p3);
}

FINLINE Float FwdScat::pdfLengthDipole(
        const Vector &uL, const Vector &nL, const Vector &R,
        const Vector *u0, const Vector &n0,
        TangentPlaneMode tangentMode, Float s) const {
    FSAssert(s >= 0);
    Vector R_virt;
    if (!getTentativeIndexMatchedVirtualSourceDisp(
            n0, nL, uL, R, 0./0., tangentMode, R_virt))
        return 0.0;

    Float p1 = (lengthSample_w1 == 0 ? 0 :
            pdfLengthShortLimit(R, u0, uL, s));
    Float p2 = (lengthSample_w2 == 0 ? 0 :
            0.5 * (pdfLengthLongLimit(R, uL, s)
                 + pdfLengthLongLimit(R_virt, uL, s)));
    Float p3 = (lengthSample_w3 == 0 ? 0 :
            pdfLengthAbsorption(s));
    return lengthSample_w1 * p1
         + lengthSample_w2 * p2
         + lengthSample_w3 * p3;
}



/**
 * \brief Sample based purely on the absorption factor
 *
 * This is the safest bet 'at infinity' (the tail is certainly more heavy
 * than the target distribution), but extremely high variance is possible
 * for high albedo materials. */
FINLINE Float FwdScat::sampleLengthAbsorption(
        Float &s, Sampler *sampler) const {
    if (sigma_a == 0)
        return 0.0;
    s = -log(sampler->next1D())/sigma_a;
    Float pdf = sigma_a*exp(-sigma_a*s);
    FSAssert(std::isfinite(s));
    FSAssert(s >= 0);
    FSAssert(std::isfinite(pdf));
    return pdf;
}

FINLINE Float FwdScat::pdfLengthAbsorption(
        Float s) const {
    if (sigma_a == 0)
        return 0.0;
    Float pdf = sigma_a*exp(-sigma_a*s);
    FSAssert(std::isfinite(pdf));
    return pdf;
}


FINLINE Float FwdScat::sampleLengthShortLimit(
        Vector R, const Vector *u0, Vector uL, Float &s, Sampler *sampler) const {
    Float pdf;
    implLengthShortLimit(R, u0, uL, s, sampler, &pdf);
    return pdf;
}

FINLINE Float FwdScat::pdfLengthShortLimit(
        Vector R, const Vector *u0, Vector uL, Float s) const {
    Float pdf;
    implLengthShortLimit(R, u0, uL, s, NULL, &pdf);
    return pdf;
}

FINLINE void FwdScat::implLengthShortLimit(
        Vector R, const Vector *u0, Vector uL, Float &s, Sampler *sampler, Float *pdf) const {
    if (u0 == NULL) {
        implLengthShortLimitMargOverU0(R, uL, s, sampler, pdf);
    } else {
        implLengthShortLimitKnownU0(R, *u0, uL, s, sampler, pdf);
    }
}

FINLINE void FwdScat::implLengthShortLimitKnownU0(
        Vector R, Vector u0, Vector uL, Float &s, Sampler *sampler, Float *pdf) const {
    double p = 0.5*sigma_s*mu;
    double lRl = R.length();
    double r = lRl * p;
    if (r == 0)  {
        if (sampler) s = 0;
        if (pdf) *pdf = 0;
        return;
    }
    double cosTheta0L = math::clamp(dot(R, u0) / lRl, -1.0, 1.0)
                      + math::clamp(dot(R, uL) / lRl, -1.0, 1.0);
    double u0dotuL = dot(u0, uL);

    double mean;
    if (r > 1e-4) { // full expression is sufficiently stable
        // transformation t = (ps)^(-3)
        // compute mean of gaussian in t: (root of a cubic polynomial)
        // Based on Maple codegen...
        double t1 = 0.1e1 / r;
        double t2 = cosTheta0L * cosTheta0L;
        double t3 = t2 * cosTheta0L;
        double t5 = sqrt(0.3e1);
        double t8 = u0dotuL * u0dotuL;
        double t18 = r * r;
        double t25 = -108 * r * u0dotuL * cosTheta0L + 96 * t3 * r
                - 216 * r * cosTheta0L - 4 * t2 * t8 - 16 * t2 * u0dotuL
                + 4 * t8 * u0dotuL + 243 * t18 - 16 * t2 + 24 * t8
                + 48 * u0dotuL + 32;
        double t26 = sqrt(t25);
        double t34 = cbrt(12 * t26 * t5 - (72 * cosTheta0L * u0dotuL)
                + (324 * r) + (64 * t3) - (144 * cosTheta0L));
        double t35 = t34 * t1;
        double t42 = 1 / t34 * t1 * (-4 * t2 + 3 * u0dotuL + 6);
        double t44 = cosTheta0L * t1;
        double t46 = t35 / 18 + 2.0/9.0 * (t44 - t42);
        double t47 = t46 * t46;
        mean = 1.0/9.0 / t18 * (6*cosTheta0L * t47 * r - u0dotuL * t46
                - t35/9.0 + 4.0/9.0*(t42 - t44) + 1);
    } else { // short r limit
        // first nontrivial order expansion:
        double t1 = sqrt(3.0);
        double t3 = (u0dotuL + 2) * (u0dotuL + 2);
        double t4 = cosTheta0L * cosTheta0L;
        double t7 = sqrt(t3 * (-t4 + u0dotuL + 2));
        double t14 = 24 * t1 * t7 - 72 * cosTheta0L * (-8.0/9.0 * t4 + u0dotuL + 2);
        double t15 = cbrt(t14);
        double t16 = t15 * t15;
        double t28 = -8.0 / 3.0 * t4 + u0dotuL + 2.0;
        double t35 = t4 * t4;
        double t41 = u0dotuL * u0dotuL;
        double t48 = r * r;
        mean = ((48 * t4 * cosTheta0L + (-36 * u0dotuL - 72) * cosTheta0L) * t16
                + 36 * (-4.0 / 3.0 * t4 + u0dotuL + 2) * t28 * t15
                - 72 * t1 * t28 * t7
                + cosTheta0L * ((768 * t35) + ((-1152 * u0dotuL - 2304) * t4)
                    + t15 * t14 + (360 * t41) + (1440 * u0dotuL) + 1440))
                / (t16 * t48 * r * 486);
    }
    if (!std::isfinite(mean) || mean <= 0)  {
        /* This usually happens for small to negative u0dotuL and
         * cosTheta0L -- at which point there is no large ballistic peak
         * anyway!
         * Any choice is better than no choice, so set it as: */
        mean = 1./(r*r*r); // 'pushing s to r'
    }
    FSAssert(std::isfinite(mean));
    FSAssert(mean > 0);

    double mean113 = pow(mean, 11./3.);
    double mean53 = pow(mean, 5./3.);
    double mean73 = pow(mean, 7./3.);
    double mean2 = mean*mean;
    double realStddev;
    if (r < 1e-4) {
        // short r limit expansion
        realStddev = sqrt((-54 * r * cosTheta0L + 12 * u0dotuL * u0dotuL + 48 * u0dotuL + 48) * pow(mean, 8./3.) / 27
                + (18 * u0dotuL + 36) * mean73 / 27 + (8 * u0dotuL*u0dotuL*u0dotuL + 48 * u0dotuL * u0dotuL
                + (-72 * r * cosTheta0L + 96) * u0dotuL - 144 * r * cosTheta0L + 64) * mean*mean*mean / 27 + mean * mean);
    } else {
        realStddev = sqrt((3*mean113)
                / (3*mean53 + 6*mean73 * r * cosTheta0L - (2*u0dotuL + 4)*mean2));
    }
    double stddevSafetyFactor = 2;
    double stddev = stddevSafetyFactor * realStddev;
    if (!std::isfinite(stddev) || stddev <= 0)  {
        stddev = mean; // heurstic!
    }
    FSAssert(std::isfinite(stddev));
    FSAssert(stddev>0);

    Float t, ps;
    if (sampler) {
        do {
            t = truncnorm(mean, stddev, 0.0, 1.0/0.0, sampler);
        } while (t == 0);
        ps = std::pow(t, -1./3.);
        s = ps / p;
    } else {
        ps = p*s;
        t = 1 / (ps*ps*ps);
    }
    FSAssert(std::isfinite(s));
    FSAssert(s > 0);

    if (pdf) {
        Float tPdf = truncnormPdf(mean, stddev, 0.0, 1.0/0.0, t);

        // transform from pdf(t = (ps)^(-3)) to pdf(ps) [factor 3*(ps)^-4] & go back to p!=1 [factor p]
        *pdf = tPdf * 3 / (ps*ps*ps*ps) * p;
    }
}

FINLINE void FwdScat::implLengthShortLimitMargOverU0(
        Vector R, Vector uL, Float &s, Sampler *sampler, Float *pdf) const {
    // Working in p=1, transforming back at the end
    Float p = 0.5*sigma_s*mu;
    Float lRl = R.length();
    Float r = lRl * p;
    Float r2 = r*r;
    Float cosTheta = math::clamp(dot(R, uL) / lRl, (Float)-1, (Float)1);

    /* TODO:
     *
     * (1) This is not very sensible for r > 1 (set this strategy's MIS
     * weight to 0 then?)
     *
     * (2) The case r=0 can happen for an effective BRDF -> handle that
     * better by relaxing the cosTheta=1 assumption in derivation? Even
     * better: make dedicated sampler (which will have different s
     * behaviour, presumably) */
    //if (r == 0 || r > 1) {
    if (r == 0) {
        if (pdf) *pdf = 0;
        if (sampler) s = 0;
        return;
    }


    Float invps_mean; // the critical point (t* in the suppl. mat. of the paper)
#if 1 /* Exact, fully cosTheta-dependent solution from Maple codegen (true), or
         crude, easy to evaluate approximation (false) */
    // Maple codegen for (the real part of) the root that we want
    double t1 = 1. / r;
    double t5 = cosTheta * cosTheta;
    double t6 = cosTheta * t5;
    double t7 = t6 * r;
    double t9 = t5 * t5;
    double t11 = t5 * r;
    double t14 = r * r;
    double t16 = r * cosTheta;
    double t20 = 96 * t7 - 4 * t9 + 180 * t11 - 20 * t6 + 243 * t14
            - 36 * t16 - 28 * t5 - 120 * r + 16;
    double t21 = abs(t20);
    double t22 = sqrt(t21);
    double t23 = (t20 > 0) - (t20 < 0); // =signum(t20): (t20>0 ? 1 : (t20<0 ? -1 : 0));
    double t24 = t23 * t22;
    double t25 = sqrt(30);
    double t34 = t25 * t22;
    double t45 = -288 * cosTheta * t25 * t24 + 3888 * r * t34 - 960 * t23 * t34
            + 1440 * t5 * t34 + 768 * t6 * t34 - 288 * cosTheta * t34 + 77760 * t11
            - 15552 * t16 + 216 * t21 + 41472 * t7 + 3840 * cosTheta + 6400;
    double t65 = t23 * t23;
    double t68 = 3888 * r * t25 * t24 + 1440 * t5 * t25 * t24
            + 768 * t6 * t25 * t24 + 216 * t65 * t21 + 4096 * t5 * t9
            + 15360 * cosTheta * t9 - 51840 * r + 104976 * t14
            - 960 * t34 - 18624 * t5 - 16000 * t6 + 11328 * t9;
    double t70 = pow(t45 + t68, 1.0 / 6.0);
    double t83 = atan2(6 * (1 - t23) * t34, (64 * t6) + 6 * (1 + t23) * t34
            + (120 * t5) + (324 * r) - 24 * cosTheta - 80);
    double t85 = cos(t83 / 3);
    invps_mean = 2. / 9. * cosTheta * t1 + 2. / 9. * t1 + t85 * t70 * t1 / 18
            + t85 / t70 * t1 * (16 * t5 + 20 * cosTheta - 8) / 18;
    // Note: t70 can give problems for negative values!
    if (!std::isfinite(invps_mean) || invps_mean <= 0) {
        invps_mean=1/r; // Heuristic 'guess' to not 'waste' a sample
    }
#else
    /* Simplest possible choice. This is accurate up to 10% relative
     * accuracy (and becomes more accurate for r->0). */
    Float invps_mean = 1/r;
#endif

    FSAssert(std::isfinite(invps_mean));
    FSAssert(invps_mean > 0);
    Float t2 = invps_mean*invps_mean;
    Float t3 = invps_mean*t2;
    Float var = t2 / (3 + 54*t3*r2 - 18*r*(cosTheta + 1)*t2);

    if (!std::isfinite(var) || var <= 0) {
        /* This happens when we aren't in a local maximum, but a
         * *minimum* (var < 0)!
         * We can bail, or set the stddev to something 'safe' ... e.g. set
         * stddev=t just to sample *something* at least.
         * Probably better: determine suitability beforehand and don't use
         * this technique if it doesn't make sense. */
        var = invps_mean*invps_mean; // just some heuristic 'guess'
    }

    const Float stddevSafetyFactor = 2.5;
    Float stddev = stddevSafetyFactor * sqrt(var);

    Float invps;
    Float ps;
    if (sampler) {
        do {
            invps = truncnorm(invps_mean, stddev, 0.0, 1.0/0.0, sampler);
        } while (invps == 0);
        ps = 1/invps;
        s = ps / p;
    } else {
        ps = p*s;
        invps = 1/ps;
    }

    if (pdf) {
        Float invpsPdf = truncnormPdf(invps_mean, stddev, 0.0, 1.0/0.0, invps);

        // transform from pdf(1/(ps)) to pdf(ps) [factor (ps)^-2] & go back to p!=1 [factor p]
        *pdf = invpsPdf / (ps*ps) * p;
    }
}




// TODO: approximation that does not require a numerical cdf inversion?
FINLINE Float FwdScat::sampleLengthLongLimit(
        Vector R, Vector uL, Float &s, Sampler *sampler) const {
    Float p = 0.5*sigma_s*mu;
    if (p == 0)
        return 0;
    Vector R_p1 = R*p;
    Float R2minusRdotUL_p1 = R_p1.lengthSquared() - dot(R_p1, uL);
    Float beta = 3./2. * R2minusRdotUL_p1;
    if (beta <= 0)
        return sampleLengthAbsorption(s, sampler);
    double B = beta;
    double A = sigma_a / p;
    FSAssert(A>0);
    FSAssert(B>0);
    double sA = sqrt(A);
    double sB = sqrt(B);
    double C = exp(4*sA*sB);
    auto cdf = [=] (double ps) {
                double erfDiffArg = (sA*ps + sB)/sqrt(ps);
                double erfSumArg  = (sA*ps - sB)/sqrt(ps);
                double erfDiff, erfSum;
                // expansion
                if (erfDiffArg > 3) {
                    double x = erfDiffArg;
                    double x2 = x*x;
                    double x3 = x2*x;
                    double x5 = x3*x2;
                    erfDiff = (1/x - 0.5/x3 + .75/x5)*exp(4*sA*sB - x2)/sqrt(M_PI_DBL);
                } else {
                    erfDiff = C * (1 - boost::math::erf(erfDiffArg));
                }
                if (erfSumArg < -3) {
                    double x = erfSumArg;
                    double x2 = x*x;
                    double x3 = x2*x;
                    double x5 = x3*x2;
                    erfSum = (-1/x + 0.5/x3 - .75/x5)/exp(x2)/sqrt(M_PI_DBL);
                } else {
                    erfSum = 1 + boost::math::erf(erfSumArg);
                }
                double theCdf = 0.5*(erfDiff + erfSum);
                if (theCdf <= -Epsilon || theCdf >= 1+Epsilon) {
                    SLog(EWarn, "invalid cdf: %e %e %e %e", theCdf, erfDiff, erfSum, C);
                };
                theCdf = math::clamp(theCdf, 0., 1.);
                return theCdf;
    };
    double u = sampler->next1D();
    auto target = [=] (double ps) { return cdf(ps) - u; };

    // Bracket the root
    double lo = 0;
    if (!std::isfinite(target(lo)) || target(lo) > 0) {
        Log(EWarn, "target(lo) did something weird: %f", target(lo));
        return 0;
    }
    double hi = 1000/A;
    if (!std::isfinite(target(hi))) {
        Log(EWarn, "target(hi) not finite: %f", target(hi));
        return 0;
    }
    while (target(hi) < 0 && hi < 1e4*1000/A)
        hi *= 3; // look further if we don't have the zero crossing bracketed
    if (!std::isfinite(target(hi)) || target(hi) < 0) {
        Log(EWarn, "could not find suitable target(hi): %f", target(hi));
        return 0;
    }

    size_t max_iter = 1000;
    try {
        std::pair<double, double> Rvnsol = boost::math::tools::toms748_solve(
                target, lo, hi,
                boost::math::tools::eps_tolerance<double>(15),
                max_iter);
        Float s_p1 = 0.5*(Rvnsol.first + Rvnsol.second);
        s = s_p1 / p;
        if (!std::isfinite(s)) {
            Log(EWarn, "FIXME %f", s);
            return 0;
        }
    } catch (const std::exception& e) {
        Log(EWarn, "root finding failed (sA %e, sB %e): %s",
                sA, sB, e.what());
        return 0;
    }
    return pdfLengthLongLimit(R, uL, s);
}

FINLINE Float FwdScat::pdfLengthLongLimit(
        Vector R, Vector uL, Float s) const {
    Float p = 0.5*sigma_s*mu;
    if (p == 0)
        return 0;
    Float s_p1 = s * p;
    Vector R_p1 = R*p;
    Float R2minusRdotUL_p1 = R_p1.lengthSquared() - dot(R_p1, uL);
    Float beta = 3./2. * R2minusRdotUL_p1;
    if (beta <= 0)
        return pdfLengthAbsorption(s);
    Float a_p1 = sigma_a/p;
    Float pdf_p1 = sqrt(beta/M_PI) / (s_p1*sqrt(s_p1))
            * math::fastexp(-beta/s_p1 - a_p1*s_p1 + 2*sqrt(beta*a_p1));
    if (!std::isfinite(pdf_p1)) {
        //Log(EWarn, "FIXME %f %e %e %e", pdf_p1, beta, a_p1, s_p1);
        return 0;
    }
    return pdf_p1 * p;
}





FINLINE Float _sampleHemisphere(const Vector &n_in, Vector &d_in, Sampler *sampler) {
    /* Sample an incoming direction (on our side of the medium) on the
     * cosine-weighted hemisphere */
    Vector hemiSamp = warp::squareToCosineHemisphere(sampler->next2D());
    Float pdf = warp::squareToCosineHemispherePdf(hemiSamp);
    hemiSamp.z = -hemiSamp.z; // pointing inwards
    d_in = Frame(n_in).toWorld(hemiSamp); // pointing inwards
    return pdf;
}

FINLINE Float _pdfHemisphere(const Vector &n_in, const Vector &d_in) {
    return INV_PI * math::abs(dot(d_in, n_in));
}




FINLINE Float FwdScat::sampleDirectionDipole(
        Vector &u0, const Vector &n0, const Vector &uL, const Vector &nL,
        const Vector &R, Float s, TangentPlaneMode tangentMode,
        bool useEffectiveBRDF, Sampler *sampler) const {
    Vector R_virt, n0_effective;
    Float realSourceWeight;
    if (!getTentativeIndexMatchedVirtualSourceDisp(n0, nL, uL, R, s,
            tangentMode, R_virt, &n0_effective, &realSourceWeight)) {
        return 0.0f; // Won't be able to evaluate bssrdf transport anyway!
    } else {
        FSAssert(R_virt.isFinite());
    }

    Float pReal = -1;
    Float pVirt = -1;
    Float u = sampler->next1D();
    if (u <= (1 - directionSampler_dipoleHemiWeight) * realSourceWeight) {
        pReal = sampleDirectionBoundaryAwareMonopole(
                u0, n0, uL, R, s, useEffectiveBRDF, sampler);
        if (pReal == 0)
            return 0.0f;
    } else if (u <= (1 - directionSampler_dipoleHemiWeight)) {
        Vector u0_virt;
        Vector n0_virt = n0  -  2*dot(n0_effective, n0) * n0_effective;
        pVirt = sampleDirectionBoundaryAwareMonopole(
                u0_virt, n0_virt, uL, R_virt, s, useEffectiveBRDF, sampler);
        if (pVirt == 0)
            return 0.0f;
        /* Don't forget: we have to transform back to the real u0! */
        u0 = u0_virt  -  2*dot(n0_effective, u0_virt) * n0_effective;
    } else {
        _sampleHemisphere(n0, u0, sampler);
    }

    if (pReal == -1)
        pReal = pdfDirectionBoundaryAwareMonopole(
                u0, n0, uL, R, s, useEffectiveBRDF);

    if (pVirt == -1) {
        /* Don't forget: we have to transform to the virtual u0 to get the
         * corresponding pdf! We also need to transform to get a 'virtual'
         * normal n0, so that, upon transforming u0_virt to its
         * corresponding u0, that u0 is on the correct side of the actual
         * boundary as determined by n0. */
        Vector u0_virt = u0  -  2*dot(n0_effective, u0) * n0_effective;
        Vector n0_virt = n0  -  2*dot(n0_effective, n0) * n0_effective;
        pVirt = pdfDirectionBoundaryAwareMonopole(
                u0_virt, n0_virt, uL, R_virt, s, useEffectiveBRDF);
    }

    Float pHemi = _pdfHemisphere(n0, u0);

    return (1 - directionSampler_dipoleHemiWeight)
                * (realSourceWeight * pReal + (1.0 - realSourceWeight) * pVirt)
            + directionSampler_dipoleHemiWeight * pHemi;
}

FINLINE Float FwdScat::pdfDirectionDipole(
        const Vector &u0, const Vector &n0, const Vector &uL, const Vector &nL,
        const Vector &R, Float s, TangentPlaneMode tangentMode,
        bool useEffectiveBRDF) const {
    Vector R_virt, n0_effective;
    Float realSourceWeight;
    if (!getTentativeIndexMatchedVirtualSourceDisp(n0, nL, uL, R, s,
            tangentMode, R_virt, &n0_effective, &realSourceWeight)) {
        return 0.0f; // Won't be able to evaluate bssrdf transport anyway!
    } else {
        FSAssert(R_virt.isFinite());
    }

    Float pReal, pVirt, pHemi;

    pReal = pdfDirectionBoundaryAwareMonopole(
            u0, n0, uL, R, s, useEffectiveBRDF);

    Vector u0_virt = u0  -  2*dot(n0_effective, u0) * n0_effective;
    Vector n0_virt = n0  -  2*dot(n0_effective, n0) * n0_effective;
    pVirt = pdfDirectionBoundaryAwareMonopole(
            u0_virt, n0_virt, uL, R_virt, s, useEffectiveBRDF);

    pHemi = _pdfHemisphere(n0, u0);

    return (1 - directionSampler_dipoleHemiWeight)
                * (realSourceWeight * pReal + (1.0 - realSourceWeight) * pVirt)
            + directionSampler_dipoleHemiWeight * pHemi;
}



FINLINE Float FwdScat::pdfDirectionBoundaryAwareMonopole_BRDF(
        const Vector &u0, const Vector &n0, const Vector &uL, const Vector &R,
        Float s) const {
    Float pdf;
    Vector u0nonConst(u0);
    implDirectionBoundaryAwareMonopole_BRDF(
            u0nonConst, n0, uL, R, s, NULL, &pdf);
    return pdf;
}

FINLINE Float FwdScat::sampleDirectionBoundaryAwareMonopole_BRDF(
        Vector &u0, const Vector &n0, const Vector &uL, const Vector &R,
        Float s, Sampler *sampler) const {
    Float pdf;
    implDirectionBoundaryAwareMonopole_BRDF(
            u0, n0, uL, R, s, sampler, &pdf);
#ifdef MTS_FWDSCAT_DEBUG
    if (pdf == 0)
        return 0;
    Float pdfCheck = pdfDirectionBoundaryAwareMonopole_BRDF(
            u0, n0, uL, R, s);
    if (math::abs(pdf-pdfCheck)/pdf > 1e-3) {
        Log(EWarn, "Inconsistent pdfs: %e %e, rel %f",
                pdf, pdfCheck, (pdf-pdfCheck)/pdf);
    }
#endif
    return pdf;
}

// if sampler is NULL: read u0 and set the pdf (should not be NULL)
// if sampler is not NULL: sample u0 and set the pdf (if it isn't NULL)
FINLINE void FwdScat::implDirectionBoundaryAwareMonopole_BRDF(
        Vector &u0, const Vector &n0, const Vector &uL, const Vector &R,
        Float s, Sampler *sampler, Float *pdf) const {
    /* Note to self: n0==nL is no longer guaranteed here, because if we are
     * sampling u0_virt, n0 will be mirrored (it will equal
     * '-n0_real = -nL') */

    if (pdf)
        *pdf = 0; // set to zero already so we can simply return on error

    if (sampler == NULL)
        FSAssert(math::abs(u0.length() - 1) < Epsilon);
    FSAssert(math::abs(uL.length() - 1) < Epsilon);
    FSAssert(R.isFinite());
    FSAssert(std::isfinite(s));
    FSAssert(s >= 0);

    // frame:
    const Vector z = n0; /* (= nL for a BSRDF if we are sampling a real
                            direction, and '-n0_real' = -nL if we are
                            sampling a virtual direction) */
    Vector x_unnorm = uL - z*dot(z,uL);
    if (x_unnorm.length() <= Epsilon) {
        /* any frame will do; a will go to 0 and the sampling will be
         * uniform where needed (e.g. phi sampling) */
        Frame f(z);
        x_unnorm = f.s;
    }
    const Vector x = normalize(x_unnorm);
    const Vector y = cross(x, z);
    FSAssert(math::abs(dot(x,y)) < Epsilon);
    FSAssert(math::abs(dot(x,z)) < Epsilon);
    FSAssert(math::abs(dot(y,z)) < Epsilon);

    Vector woi = -uL; // outgoing direction in 'incident' orientation
    /* BRDF consistency check:
     *   R == 0       if we are samping a real direction
     *   R == -|R|nL  if we are samping a virt direction
     *                note: in that case, the u0 that we get here, is
     *                actually '-u0_real', and -uL!
     */
    FSAssert(R.isZero() || math::abs(dot(R,n0)) > 0.999 * R.length());

    double a, b, c;
    if (m_winklerCorrection) {
        double C, D, E, F, Z;
        calcValues2(s, C, D, E, F, NULL, &Z);

        a = 0.5 * Z * dot(woi,x);
        CancellationCheck(D * dot(woi,z), E*dot(R,z));
        b = D * dot(woi,z) + E*dot(R,z);
        c = 0.25*E*E/F;
    } else {
        double C, D, E, F;
        calcValues(s, C, D, E, F);

        /* TODO: D - 0.5E^2/F has cancellation problem and should be rewritten.
         * It's barely hanging by its teeth in double precision. */
        //CancellationCheck(D, -0.5*E*E/F);
        a = (D - 0.5*E*E/F) * dot(woi,x);
        CancellationCheck(D * dot(woi,z), E*dot(R,z));
        b = D * dot(woi,z) + E*dot(R,z);
        c = 0.25*E*E/F;
    }

    if (math::abs(a) < 1e-4) {
        a = 0;
        /* This makes the standard deviations go to infinity (i.e. simply
         * uniform sampling) and helps with stability. There are pdf
         * inconsistencies otherwise. */
    }


    /* Sample cos(theta) */
    double cosThetaSd = 1/sqrt(2*c + math::abs(a));
    FSAssert(cosThetaSd>=0);
    if (cosThetaSd == 0)
        return;
    double cosThetaMean = b * math::square(cosThetaSd);
    double cosTheta;
    if (sampler) {
        cosTheta = truncnorm(cosThetaMean, cosThetaSd, -1.0, 0.0, sampler);
    } else {
        cosTheta = dot(u0,z);
        FSAssert(-1-Epsilon <= cosTheta && cosTheta <= Epsilon);
        cosTheta = math::clamp(cosTheta, -1.0, 0.0);
    }
    double cosThetaPdf = truncnormPdf(cosThetaMean, cosThetaSd, -1.0, 0.0, cosTheta);
    double sinTheta = math::safe_sqrt(1 - math::square(cosTheta));


    /* Sample phi:
     * weight: exp(a*sin(theta) * cos(phi))
     * -> expand cos(phi) up to second order:
     *       - around phi=0  if a>0 (i.e. cos(phi) -> +1  => phi->0)
     *       - around phi=pi if a<0 (i.e. cos(phi) -> -1  => phi->pi)
     */
    double phiSd = 1.0 / sqrt(math::abs(a) * sinTheta);
    if (phiSd == 0)
        return;
    double phiMean, phiLo, phiHi;
    if (a > 0) {
        phiMean = 0;
        phiLo = -M_PI_DBL;
        phiHi =  M_PI_DBL;
    } else {
        phiMean = M_PI_DBL;
        phiLo = 0;
        phiHi = TWO_PI_DBL;
    }
    double phi;
    if (sampler) {
        phi = truncnorm(phiMean, phiSd, phiLo, phiHi, sampler);
    } else {
        phi = atan2(dot(u0,y), dot(u0,x));
        if (phi < phiLo)
            phi += TWO_PI_DBL;
        FSAssert(phiLo <= phi && phi <= phiHi);
    }
    double phiPdf = truncnormPdf(phiMean, phiSd, phiLo, phiHi, phi);
    double sinPhi, cosPhi;
    math::sincos(phi, &sinPhi, &cosPhi);

    Vector constructed_u0 = x * cosPhi*sinTheta  +  y * sinPhi*sinTheta  +  z * cosTheta;
    FSAssert(math::abs(constructed_u0.length() - 1) < Epsilon);

    if (sampler) {
        u0 = constructed_u0;
    } else {
        FSAssert((u0-constructed_u0).length() < ShadowEpsilon);
    }

    double thePdf = cosThetaPdf * phiPdf;
    if (!std::isfinite(thePdf) || thePdf < 0) {
        Log(EWarn, "problematic pdf: %f", thePdf);
        return;
    }

    if (pdf)
        *pdf = thePdf;
}





FINLINE Float FwdScat::sampleDirectionBoundaryAwareMonopole(
        Vector &u0, const Vector &n0, const Vector &uL, const Vector &R,
        Float s, bool useEffectiveBRDF, Sampler *sampler) const {

    if (useEffectiveBRDF)
        return sampleDirectionBoundaryAwareMonopole_BRDF(
                u0, n0, uL, R, s, sampler);

    Float p1,p2;
    if (sampler->next1D() < directionSampler_origWeight) {
        p1 = sampleDirectionBoundaryAwareMonopole_orig(
                u0, n0, uL, R, s, sampler);
        if (p1 == 0)
            return 0;
        p2 = pdfDirectionBoundaryAwareMonopole_bis(
                u0, n0, uL, R, s);
    } else {
        p2 = sampleDirectionBoundaryAwareMonopole_bis(
                u0, n0, uL, R, s, sampler);
        if (p2 == 0)
            return 0;
        p1 = pdfDirectionBoundaryAwareMonopole_orig(
                u0, n0, uL, R, s);
    }
    return p1 * directionSampler_origWeight + p2 * (1 - directionSampler_origWeight);
}

FINLINE Float FwdScat::pdfDirectionBoundaryAwareMonopole(
        const Vector &u0, const Vector &n0, const Vector &uL, const Vector &R,
        Float s, bool useEffectiveBRDF) const {

    if (useEffectiveBRDF)
        return pdfDirectionBoundaryAwareMonopole_BRDF(
                u0, n0, uL, R, s);

    Float p1,p2;
    p1 = pdfDirectionBoundaryAwareMonopole_orig(
            u0, n0, uL, R, s);
    p2 = pdfDirectionBoundaryAwareMonopole_bis(
            u0, n0, uL, R, s);
    return p1 * directionSampler_origWeight + p2 * (1 - directionSampler_origWeight);
}


FINLINE Float FwdScat::sampleDirectionBoundaryAwareMonopole_orig(
        Vector &u0, const Vector &n0, const Vector &uL, const Vector &R,
        Float s, Sampler *sampler) const {
    FSAssert(math::abs(uL.length() - 1) < Epsilon);
    FSAssert(R.isFinite());
    FSAssert(std::isfinite(s));
    FSAssert(s >= 0);
    /* The relevant factor in the propagator to sample u0 is:
     *    pdf(u0) ~ exp(dot(H, u0))
     * where
     *    H = E*R - D*uL
     * i.e. a simple exponential distribution on the cosine of the angle
     * theta between H and u0:
     *    pdf(u0) ~ exp(|H| * cos(theta))
     * going to spherical coordinates:
     *    pdf(theta,phi) ~ sin(theta) * exp(|H| * cos(theta))
     * or
     *    pdf(cos(theta),phi) ~ exp(|H| * cos(theta))
     *
     * ====================================================================
     * TAKING INTO ACCOUNT THE INCOMING NORMAL SO WE DON'T GENERATE INVALID
     * INCOMING DIRECTIONS THAT COME FROM WITHIN OUR OWN MEDIUM
     * ====================================================================
     * The derivation above ignores the position of the incoming tangent
     * plane (i.e. the incoming normal). We cannot sample *incoming*
     * directions that don't come in from outside of our medium (i.e. we
     * must have have dot(u0,n0) < 0).
     *
     * Cutting off the part dot(u0,n0) > 0 from the integration space and
     * setting up a proper pdf for this case is not analytically tractable.
     * We just clamp the cosTheta range to the extremal values, and then
     * sample phi only within the allowed wedge. Normally, the cosTheta pdf
     * should be weighted to reflect the fact that the size of the allowed
     * phi wedge changes depending on cosTheta, but that's mostly a lower
     * order effect that would get swamped by the exponential factor in the
     * weight anyway (for large |H| -- for small |H| we could essentially
     * just be sampling the hemisphere uniformly/cos-weighted).
     * And besides, we're only using to sample a dipole direction, so not
     * getting the monopole exactly right is probably not the most
     * important source of error.
     *
     * REMARK: This is the version from the SIGGRAPH2017 paper. See the
     * '_bis' version for an alternative approach that works in spherical
     * coordinates about the normal of the admissible hemisphere and
     * expands the trigonometric functions within the exponential weight to
     * obtain simple (truncated) gaussian and/or exponential distributions.
     * */
    double C, D, E, F;
    calcValues(s, C, D, E, F);

    Vector3d H = E*Vector3d(R) - D*Vector3d(uL);
    double lHl = H.length();
    FSAssert(std::isfinite(lHl));
    Vector Hnorm = Vector(H / lHl);

    /* Regularization */
    lHl = (lHl > 1./MTS_FWDSCAT_DIRECTION_MIN_MU) ? 1./MTS_FWDSCAT_DIRECTION_MIN_MU : lHl;

    /* If we are badly conditioned: pick coordinates around n0 instead of
     * trying to set up a Hnorm frame. */
    bool badlyConditioned = math::abs(dot(n0, Hnorm)) > 1-Epsilon;

    Float minCosTheta, maxCosTheta;
    Float phiCutoffSlope = 0./0.;
    Vector projectionDir;
    if (badlyConditioned) {
        projectionDir = n0;
        minCosTheta = -1;
        maxCosTheta = 0; // only incoming directions
    } else {
        Float exactSin = dot(n0,Hnorm);
        Float tmpCos = roundCosThetaForStability(math::safe_sqrt(1 - math::square(exactSin)), (Float)-1, (Float)1);
        Float tmpSin = math::safe_sqrt(1 - math::square(tmpCos)); // detour because of potential rounding
        if (dot(Hnorm, n0) < 0) {
            // H points to the correct (=incoming) side of the boundary -> maxCos=1
            minCosTheta = -tmpCos;
            maxCosTheta = 1.0f;
            roundCosThetaBoundsForStability(minCosTheta, maxCosTheta);
            phiCutoffSlope = tmpSin / minCosTheta;
        } else {
            minCosTheta = -1.0f;
            maxCosTheta = tmpCos;
            roundCosThetaBoundsForStability(minCosTheta, maxCosTheta);
            phiCutoffSlope = tmpSin / maxCosTheta;
        }
        projectionDir = Hnorm;
    }
    FSAssert(minCosTheta >= -1 && minCosTheta <= 0);
    FSAssert(maxCosTheta >=  0 && maxCosTheta <= 1);

    Float cosTheta;
    Float cosThetaPdf;
    if (lHl < Epsilon) {
        /* Expansion in small |H|, up to second order */
        Float d = maxCosTheta - minCosTheta;
        Float d2 = d*d;
        Float d3 = d*d2;
        Float u = sampler->next1D();
        /* The expanded cosTheta is still guaranteed to stay within bounds */
        cosTheta = roundCosThetaForStability(minCosTheta + d*u - 0.5*u*(u-1)*d2*lHl
                + 1./6.*(2*u-1)*(u-1)*u*d3*lHl*lHl, minCosTheta, maxCosTheta);
        /* The expanded pdf is still guaranteed to be >= 0 */
        cosThetaPdf = (1 + 0.5*(2*cosTheta - minCosTheta - maxCosTheta) * lHl
                + 1./12.*(math::square(maxCosTheta) + math::square(minCosTheta)
                        + 4*minCosTheta*maxCosTheta
                        + 6*cosTheta*(cosTheta - minCosTheta - maxCosTheta))
                                * lHl*lHl) / d;
    } else if (lHl > LOG_REDUCED_PRECISION / 2) {
        /* Expansion in large |H| */

        cosTheta = roundCosThetaForStability(maxCosTheta + log(sampler->next1D()) / lHl,
                minCosTheta, maxCosTheta);
        if (cosTheta < minCosTheta) {
            /* *INSANELY* unlikely (pdf below would probably cut off to
             * zero anyway, but universe would die of heat death first) */
            Log(EWarn, "Woah! Universe should have encountered heat death, "
                    "or code is bugged -- cosTheta: %f < minCosTheta %f",
                    cosTheta, minCosTheta);
            cosTheta = roundCosThetaForStability(minCosTheta, minCosTheta, maxCosTheta);
        }
        cosThetaPdf = lHl * exp(lHl*(cosTheta - maxCosTheta));
        if (!std::isfinite(cosThetaPdf) || cosThetaPdf <= 0)
            Log(EWarn, "Something fishy happened, cosThetaPdf %f, "
                    "cosTheta: %f (min %f max %f), lRlregularized %e",
                    cosThetaPdf, cosTheta, minCosTheta, maxCosTheta, lHl);
    } else {
        Float u = sampler->next1D();
        cosTheta = roundCosThetaForStability(log((1-u)*exp(minCosTheta*lHl) + u*exp(maxCosTheta*lHl)) / lHl,
                minCosTheta, maxCosTheta);
        cosThetaPdf = lHl/(exp(maxCosTheta*lHl) - exp(minCosTheta*lHl)) * exp(lHl * cosTheta);
    }
    FSAssert(minCosTheta - ShadowEpsilon <= cosTheta  &&  cosTheta <= maxCosTheta + ShadowEpsilon);
    FSAssert(std::isfinite(cosThetaPdf) && cosThetaPdf > 0);
    Float sinTheta = math::safe_sqrt(1 - cosTheta * cosTheta);

    Float minPhi, maxPhi;
    if (badlyConditioned) {
        minPhi = -HALF_PI;
        maxPhi = M_PI+HALF_PI;
    } else {
        /* height of the cutoff, when looking at the phi slice circle */
        Float h = phiCutoffSlope * cosTheta;
        Float hUnitCircle; // to a height in a unit circle
        if (sinTheta == 0) {
            hUnitCircle = -1;
        } else {
            hUnitCircle = h / sinTheta;
        }
        FSAssert(std::isfinite(hUnitCircle));
        /* phi frame: phi = 0 corresponds to the direction perpendicular to H
         * and n0 (at h=0), whith 'down' (negative h) being in the direction of
         * the normal, so that 'up' points towards the incoming directions. */
        FSAssertWarn(hUnitCircle <= 1+ShadowEpsilon);
        /* if hUnitCircle < -1: the full 2pi range of phi is permitted ->
         * safe_asin clamps for us */
        minPhi = math::safe_asin(hUnitCircle);
        maxPhi = M_PI - minPhi;
        FSAssert(math::abs(sin(minPhi)-sin(maxPhi)) < Epsilon);
    }
    FSAssert(minPhi >= -HALF_PI && minPhi <= HALF_PI);
    FSAssert(maxPhi >=  HALF_PI && maxPhi <= M_PI + HALF_PI);
    Float phi = minPhi + (maxPhi - minPhi) * sampler->next1D();
    if (maxPhi == minPhi)
        return 0.0f;
    Float phiPdf = 1.0 / (maxPhi - minPhi);
    FSAssert(std::isfinite(phiPdf) && phiPdf > 0);
    /* Note: for a perfect sampling, phiPdf should have been a constant
     * (independent of cosTheta) [And ideally the dot(n_in,d_ni) should
     * also have been taken into account] */

    Vector upDir, zeroPhiDir;
    if (badlyConditioned) { // Hnorm approximately equal to n0
        /* any frame perpendicular to H will do (min and max cosTheta
         * are set to -1 and 1 above anyway) */
        Frame f(projectionDir);
        upDir = f.s;
        zeroPhiDir = f.t;
    } else {
        FSAssert(projectionDir == Hnorm);
        upDir = -normalize(n0 - Hnorm*dot(n0, Hnorm)); // point in opposite direction than normal
        zeroPhiDir = cross(upDir, Hnorm);
    }
    FSAssertWarn(math::abs(upDir.length() - 1) < Epsilon);
    FSAssertWarn(math::abs(zeroPhiDir.length() - 1) < Epsilon);
    FSAssertWarn(math::abs(dot(zeroPhiDir, projectionDir) < Epsilon));
    FSAssertWarn(math::abs(dot(zeroPhiDir, upDir) < Epsilon));
    FSAssertWarn(math::abs(dot(projectionDir, upDir) < Epsilon));
    FSAssertWarn(badlyConditioned || math::abs(dot(zeroPhiDir, n0) < Epsilon));
    FSAssertWarn(badlyConditioned || dot(upDir, n0) <= Epsilon); // negative with safety epsilon

#if MTS_FWDSCAT_DEBUG
    /* This can become bad when roundCosThetaForStability is too agressive... */
    // The point at the extremal cosine should lie exactly in the plane
    if (minCosTheta != -1)
        FSAssertWarn(ShadowEpsilon > math::abs(dot(n0,
                minCosTheta*projectionDir + math::safe_sqrt(1-minCosTheta*minCosTheta)*upDir)));
    if (maxCosTheta != 1)
        FSAssertWarn(ShadowEpsilon > math::abs(dot(n0,
                maxCosTheta*projectionDir + math::safe_sqrt(1-maxCosTheta*maxCosTheta)*upDir)));
    // The point at the (non-trivial) extremal phi values should lie exactly in the plane
    Float minCosPhi, minSinPhi;
    math::sincos(minPhi, &minSinPhi, &minCosPhi);
    if (minPhi != -HALF_PI)
        FSAssertWarn(ShadowEpsilon > math::abs(dot(n0,
                sinTheta * (minSinPhi*upDir + minCosPhi*zeroPhiDir) + cosTheta * projectionDir)));
    Float maxCosPhi, maxSinPhi;
    math::sincos(maxPhi, &maxSinPhi, &maxCosPhi);
    if (maxPhi != M_PI+HALF_PI)
        FSAssertWarn(ShadowEpsilon > math::abs(dot(n0,
                sinTheta * (maxSinPhi*upDir + maxCosPhi*zeroPhiDir) + cosTheta * projectionDir)));
#endif

    Float sinPhi, cosPhi;
    math::sincos(phi, &sinPhi, &cosPhi);
    u0 = sinTheta * (sinPhi*upDir + cosPhi*zeroPhiDir) + cosTheta * projectionDir;
    FSAssertWarn(math::abs(u0.length() - 1) < Epsilon);
#ifdef MTS_FWDSCAT_DEBUG
    if (dot(u0,n0) > ShadowEpsilon) // We *aren't* an incoming direction (with some epsilon margin)
        Log(EWarn, "Generated non-incoming direction: cosine %f (should be < 0) -- badlyConditioned: %d",
                dot(u0,n0), badlyConditioned);
#endif
    if (dot(u0,n0) >= 0) { // can happen due to roundoff and roundCosThetaForStability
        Log(EWarn, "Incorrect incoming direction in sampleDirectionBoundaryAwareMonopole()!");
        return 0.0f;
    }

    Float pdf = cosThetaPdf * phiPdf;
    FSAssert(pdf >= 0);
    if (pdf == 0) {
        Log(EWarn, "Underflow occured in the pdf of sampleDirectionBoundaryAwareMonopole");
        FSAssertWarn(0 == pdfDirectionBoundaryAwareMonopole_orig(u0, n0, uL, R, s));
        return 0.0f;
    }
#ifdef MTS_FWDSCAT_DEBUG
    Float pdfCheck = pdfDirectionBoundaryAwareMonopole_orig(u0, n0, uL, R, s);
    if (math::abs((pdf - pdfCheck) / pdf) > 1e-3)
        Log(EWarn, "Inconsistent pdfs: %e %e, rel %f; costheta %e, |H| %e, E %e, D %e", pdf, pdfCheck, (pdf-pdfCheck)/pdf, cosTheta, lHl, E, D);
#endif
    FSAssertWarn(cosTheta == 0 || math::abs((dot(u0,projectionDir) - cosTheta) / cosTheta) < 1e-3);
    return pdf;
}

// TODO combine pdf and sampler with an 'impl' style function
FINLINE Float FwdScat::pdfDirectionBoundaryAwareMonopole_orig(
        const Vector &u0, const Vector &n0, const Vector &uL, const Vector &R,
        Float s) const {

    if (dot(u0,n0) >= 0)
        return 0.0;
    // Now, cosTheta and phi should lie correctly within their bounds!

    double C, D, E, F;
    calcValues(s, C, D, E, F);

    Vector3d H = E*Vector3d(R) - D*Vector3d(uL);
    double lHl = H.length();
    FSAssert(std::isfinite(lHl));
    Vector Hnorm = Vector(H / lHl);

    /* Regularization */
    lHl = (lHl > 1./MTS_FWDSCAT_DIRECTION_MIN_MU) ? 1./MTS_FWDSCAT_DIRECTION_MIN_MU : lHl;
    bool badlyConditioned = math::abs(dot(n0, Hnorm)) > 1-Epsilon;

    Float minCosTheta, maxCosTheta, minPhi, maxPhi;
    Vector projectionDir;
    Float cosTheta;
    if (badlyConditioned) {
        minCosTheta = -1;
        maxCosTheta = 0;
        minPhi = -HALF_PI;
        maxPhi = M_PI+HALF_PI;
        projectionDir = n0;

        cosTheta = roundCosThetaForStability(dot(projectionDir, u0),
                -1, 1); /* Don't artificially clamp, we should be OK already! */
        FSAssertWarn(minCosTheta - Epsilon <= cosTheta);
        FSAssertWarn(cosTheta <= maxCosTheta + Epsilon);
    } else {
        Float phiCutoffSlope;
        Float exactSin = dot(n0,Hnorm);
        Float tmpCos = roundCosThetaForStability(math::safe_sqrt(1 - math::square(exactSin)), (Float)-1, (Float)1); // z* in paper
        Float tmpSin = math::safe_sqrt(1 - math::square(tmpCos)); // detour because of potential rounding
        if (dot(Hnorm, n0) < 0) {
            // H points to the correct (=incoming) side of the boundary -> maxCos=1
            minCosTheta = -tmpCos;
            maxCosTheta = 1.0f;
            roundCosThetaBoundsForStability(minCosTheta, maxCosTheta);
            phiCutoffSlope = tmpSin / minCosTheta;
        } else {
            minCosTheta = -1.0f;
            maxCosTheta = tmpCos;
            roundCosThetaBoundsForStability(minCosTheta, maxCosTheta);
            phiCutoffSlope = tmpSin / maxCosTheta;
        }
        projectionDir = Hnorm;
        cosTheta = roundCosThetaForStability(dot(projectionDir, u0),
                -1, 1); /* Don't artificially clamp, we should be OK already! */
        FSAssertWarn(minCosTheta - Epsilon <= cosTheta);
        FSAssertWarn(cosTheta <= maxCosTheta + Epsilon);

        Float h = phiCutoffSlope * cosTheta;
        Float sinTheta = math::safe_sqrt(1 - cosTheta * cosTheta);
        Float hUnitCircle;
        if (sinTheta == 0) {
            hUnitCircle = -1;
        } else {
            hUnitCircle = h / sinTheta;
        }
        FSAssert(std::isfinite(hUnitCircle));
        FSAssertWarn(hUnitCircle <= 1+ShadowEpsilon);
        minPhi = math::safe_asin(hUnitCircle);
        maxPhi = M_PI - minPhi;
    }
    FSAssert(minCosTheta >= -1 && minCosTheta <= 0);
    FSAssert(maxCosTheta >=  0 && maxCosTheta <= 1);
    FSAssert(minPhi >= -HALF_PI && minPhi <= HALF_PI);
    FSAssert(maxPhi >=  HALF_PI && maxPhi <= M_PI + HALF_PI);

    FSAssert(minCosTheta - Epsilon <= cosTheta  &&  cosTheta <= maxCosTheta + Epsilon);

    Float cosThetaPdf;
    // expansion
    if (lHl < Epsilon) {
        Float d = maxCosTheta - minCosTheta;
        cosThetaPdf = (1 + 0.5*(2*cosTheta - minCosTheta - maxCosTheta) * lHl
                + 1./12.*(math::square(maxCosTheta) + math::square(minCosTheta)
                        + 4*minCosTheta*maxCosTheta
                        + 6*cosTheta*(cosTheta - minCosTheta - maxCosTheta))
                                * lHl*lHl) / d;
    } else if (lHl > LOG_REDUCED_PRECISION / 2) {
        cosThetaPdf = lHl * exp(lHl*(cosTheta - maxCosTheta));
        if (!std::isfinite(cosThetaPdf))
            Log(EWarn, "Something fishy happened, cosThetaPdf %f, cosTheta: %f (min %f max %f), lRlregularized %e", cosThetaPdf, cosTheta, minCosTheta, maxCosTheta, lHl);
    } else {
        cosThetaPdf = lHl/(exp(maxCosTheta*lHl) - exp(minCosTheta*lHl)) * exp(lHl * cosTheta);
    }
    FSAssert(std::isfinite(cosThetaPdf) && cosThetaPdf >= 0);

    Float phiPdf = 1.0 / (maxPhi - minPhi);
    return cosThetaPdf * phiPdf;
}






/**
 * Samples exp(a*sin(theta) + b*cos(theta)) on d(cos(theta)) if sampler is
 * given, otherwise simply returns pdf of given cosTheta.
 * Assumption: a >= 0 and the returned cosine is constrained within [-1..0]
 * Returns pdf(cos(theta)). */
FINLINE double sampleExpSinCos_dCos(double a, double b, double &cosTheta, Sampler *sampler) {
    SFSAssert(a >= -Epsilon);
    // TODO; better blend based on relative magnitudes of a and b...
    Float laplaceWeight = (a < Epsilon) ? 0.00 : 0.49;
    Float expWeight     = (a < Epsilon) ? 0.98 : 0.49;
    Float uniformWeight = 0.02;
    SFSAssert(math::abs(laplaceWeight + expWeight + uniformWeight - 1) < Epsilon);

    enum {ELaplace, EExp, EUniform, ENone} strategy;
    if (sampler) {
        Float u = sampler->next1D();
        if (u < laplaceWeight) {
            strategy = ELaplace;
        } else if (u < laplaceWeight + expWeight) {
            strategy = EExp;
        } else {
            strategy = EUniform;
        }
    } else {
        strategy = ENone;
    }

    /* Laplace approximation */
    double mean = b / sqrt(a*a + b*b);
    double var = 0.5 * pow(1 - mean*mean, 1.5) / a;
    double stddevSafetyFactor = 2;
    double stddev = stddevSafetyFactor * sqrt(var);
    if (sampler && strategy == ELaplace) {
        cosTheta = truncnorm(mean, stddev, -1.0, 0.0, sampler);
    }

    /* Exponential approximation: |b| >> a */
    if (sampler && strategy == EExp) {
        cosTheta = warp::uniformToTruncatedExponential(
                b, -1.0, 0.0, sampler->next1D());
    }

    if (sampler && strategy == EUniform) {
        cosTheta = -sampler->next1D();
    }

    double laplacePdf = laplaceWeight == 0 ? 0 : truncnormPdf(
            mean, stddev, -1.0, 0.0, cosTheta);
    double expPdf = warp::uniformToTruncatedExponentialPdf(
            b, -1.0, 0.0, cosTheta);
    double uniformPdf = 1;

    return laplaceWeight * laplacePdf
            + expWeight * expPdf
            + uniformWeight * uniformPdf;
}


/* Sample phi with weight: exp(a * cos(phi))
 * Returns: pdf(phi) */
FINLINE double sampleExpCos_dPhi(double a, double &phi, Sampler *sampler) {
    /* Sample phi:
     * weight: exp(a * cos(phi))
     * -> expand cos(phi) up to second order:
     *       - around phi=0  if a>0 (i.e. cos(phi) -> +1  => phi->0)
     *       - around phi=pi if a<0 (i.e. cos(phi) -> -1  => phi->pi)
     */
    const double uniformWeight   = 0.10;
    const double truncnormWeight = 0.90;
    bool doUniformSamplingInstead = false;
    if (sampler && sampler->next1D() < uniformWeight) {
        doUniformSamplingInstead = true;
    }

    double phiOrigSd = 1.0 / sqrt(math::abs(a));
    double stddevSafetyFactor = phiOrigSd > 1.5 ? 1.8 : 1.1; // we are less precise for high stddev
    double phiSd = stddevSafetyFactor * phiOrigSd;
    if (phiSd == 0)
        return 0;
    double phiMean, phiLo, phiHi;
    if (a > 0) {
        phiMean = 0;
        phiLo = -M_PI_DBL;
        phiHi =  M_PI_DBL;
    } else {
        phiMean = M_PI_DBL;
        phiLo = 0;
        phiHi = TWO_PI_DBL;
    }
    if (sampler && !doUniformSamplingInstead) {
        phi = truncnorm(phiMean, phiSd, phiLo, phiHi, sampler);
    }

    if (sampler && doUniformSamplingInstead) {
        phi = phiLo + sampler->next1D() * (phiHi - phiLo);
    }

    double phiForPdf;
    phiForPdf = phi;
    if (phiForPdf < phiLo) {
        /* can happen when we did not sample ourself, but got fed something
         * from atan2 (NOTE: more robust: full mod 2pi...) */
        phiForPdf += TWO_PI_DBL;
    }
    SFSAssert(phiLo <= phiForPdf && phiForPdf <= phiHi);

    return truncnormWeight * truncnormPdf(phiMean, phiSd, phiLo, phiHi, phiForPdf)
            + uniformWeight * INV_TWOPI;
}








FINLINE Float FwdScat::sampleDirectionBoundaryAwareMonopole_bis(
        Vector &u0, const Vector &n0, const Vector &uL, const Vector &R,
        Float s, Sampler *sampler) const {

    Float pdf;
    implDirectionBoundaryAwareMonopole_bis(
            u0, n0, uL, R, s, sampler, &pdf);

#ifdef MTS_FWDSCAT_DEBUG
    if (pdf == 0)
        return 0;
    Float pdfCheck = pdfDirectionBoundaryAwareMonopole_bis(
            u0, n0, uL, R, s);
    if (math::abs(pdf-pdfCheck)/pdf > 1e-3) {
        Log(EWarn, "Inconsistent pdfs: %e %e, rel %f",
                pdf, pdfCheck, (pdf-pdfCheck)/pdf);
    }
#endif

    return pdf;
}

FINLINE Float FwdScat::pdfDirectionBoundaryAwareMonopole_bis(
        const Vector &u0, const Vector &n0, const Vector &uL, const Vector &R,
        Float s) const {

    Float pdf;
    Vector u0copy(u0);
    implDirectionBoundaryAwareMonopole_bis(
            u0copy, n0, uL, R, s, NULL, &pdf);
    return pdf;
}

/**
 * Direct sampling on the admissible hemisphere by expanding the
 * trigonometric functions in the exponential, giving a simple (truncated)
 * gaussian or exponential.
 *
 * Note: the difference between this version and the '_orig' version of the
 * direction sampler seems negligible in practise, so either one will
 * probably suffice on its own (without the need for an MIS combination).
 * (The '_orig' version is the one that was described in the SIGGRAPH2017
 * paper)
 * */
FINLINE void FwdScat::implDirectionBoundaryAwareMonopole_bis(
        Vector &u0, const Vector &n0, const Vector &uL, const Vector &R,
        Float s, Sampler *sampler, Float *pdf) const {
    if (pdf)
        *pdf = 0; // set to zero already so we can simply return on error

    if (sampler == NULL)
        FSAssert(math::abs(u0.length() - 1) < Epsilon);

    FSAssert(math::abs(uL.length() - 1) < Epsilon);
    FSAssert(R.isFinite());
    FSAssert(std::isfinite(s));
    FSAssert(s >= 0);

    double C, D, E, F;
    calcValues(s, C, D, E, F);

    // frame:
    const Vector3d z(n0);
    Vector3d H = E*Vector3d(R) - D*Vector3d(uL);
    Vector3d x_unnorm = H - z*dot(z,H);
    if (x_unnorm.length() <= Epsilon * H.length()) {
        /* any frame will do; a will go to 0 and the sampling will be
         * uniform where needed (e.g. phi sampling) */
        Frame f(n0); // n0 = 'Vector(z)'
        /* TODO when compiled for single precision: this will not be
         * orthogonal up to double precision (single not tested...)!: */
        x_unnorm = Vector3d(f.s);
    }
    const Vector3d x = normalize(x_unnorm);
    const Vector3d y = cross(x, z);
    FSAssert(math::abs(dot(x,y)) < Epsilon);
    FSAssert(math::abs(dot(x,z)) < Epsilon);
    FSAssert(math::abs(dot(y,z)) < Epsilon);

    // TODO: regularize length of H!

    /* Sample cos(theta) */
    double a = dot(H,x);
    double b = dot(H,z);
    FSAssert(a >= -10*Epsilon*H.length());
    FSAssert(std::isfinite(b));
    if (math::abs(a) < 1e-4) {
        a = 0;
        /* This makes the standard deviations go to infinity (i.e. simply
         * uniform sampling) and helps with stability. There are pdf
         * inconsistencies otherwise. */
    }
    double cosTheta, cosThetaPdf;
    const Vector3d u0d(u0);
    if (sampler) {
        cosThetaPdf = sampleExpSinCos_dCos(a, b, cosTheta, sampler);
    } else {
        cosTheta = dot(u0d,z);
        FSAssert(-1-Epsilon <= cosTheta && cosTheta <= Epsilon);
        cosTheta = math::clamp(cosTheta, -1.0, 0.0);
        cosThetaPdf = sampleExpSinCos_dCos(a, b, cosTheta, NULL);
    }
    double sinTheta = math::safe_sqrt(1 - math::square(cosTheta));


    /* Sample phi:
     * weight: exp(a*sin(theta) * cos(phi)) */
    double phi, phiPdf;
    double phiCte = math::abs(a) * sinTheta;
    if (sampler) {
        phiPdf = sampleExpCos_dPhi(phiCte, phi, sampler);
    } else {
        phi = atan2(dot(u0d,y), dot(u0d,x));
        phiPdf = sampleExpCos_dPhi(phiCte, phi, NULL);
    }
    if (phiPdf == 0)
        return;

    double sinPhi, cosPhi;
    math::sincos(phi, &sinPhi, &cosPhi);

    Vector constructed_u0 = Vector(x * cosPhi*sinTheta  +  y * sinPhi*sinTheta  +  z * cosTheta);
    FSAssert(math::abs(constructed_u0.length() - 1) < Epsilon);

    if (sampler) {
        u0 = constructed_u0;
    } else {
        FSAssert((u0-constructed_u0).length() < ShadowEpsilon);
    }

    double thePdf = cosThetaPdf * phiPdf;
    if (!std::isfinite(thePdf) || thePdf < 0) {
        Log(EWarn, "problematic pdf: %f", thePdf);
        return;
    }

    if (pdf)
        *pdf = thePdf;
}

MTS_NAMESPACE_END
