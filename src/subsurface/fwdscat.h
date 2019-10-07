#pragma once
#if !defined(__MITSUBA_MEDIUM_FWDSCAT_H_)
#define __MITSUBA_MEDIUM_FWDSCAT_H_

#include <mitsuba/mitsuba.h>

MTS_NAMESPACE_BEGIN

/* This ensures that the pdf calculation from a set of ('numerically
 * rounded') directions doesn't become badly conditioned when compared to
 * the calculation of the pdf during sampling. */
// WARNING: Single precision has not been tested! Value is just a guess...
#ifdef SINGLE_PRECISION
# define MTS_FWDSCAT_DIRECTION_MIN_MU 1e-3
#else
# define MTS_FWDSCAT_DIRECTION_MIN_MU 1e-4
#endif


class MTS_EXPORT FwdScat : public Object {
public:
    MTS_DECLARE_CLASS();

    FwdScat(Float g, Float sigma_s, Float sigma_a, Float eta) :
                mu(1 - g), sigma_s(sigma_s), sigma_a(sigma_a), m_eta(eta) {
        if (g < 0 || g >= 1) {
            Log(EError, "Valid values for g are in [0,1). "
                    "Sensible values are close to 1.");
        }
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "FwdScat[mu="<<mu
                <<", sigma_s="<<sigma_s
                <<", sigma_a="<<sigma_a
                <<", eta="<<m_eta
                <<"]";
        return oss.str();
    }

    enum TangentPlaneMode {
        EUnmodifiedIncoming,
        EUnmodifiedOutgoing,
        EFrisvadEtAl,
        EFrisvadEtAlWithMeanNormal,
    };

    enum DipoleMode {
        EReal = 1,
        EVirt = 2,
        ERealAndVirt = EReal | EVirt,
    };

    enum ZvMode {
        EClassicDiffusion, /// As in the original Jensen et al. dipole
        EBetterDipoleZv,   /// As in the better dipole model of d'Eon
        EFrisvadEtAlZv,    /// As in the directional dipole model of Frisvad et al.
    };

    Float evalDipole(
            Normal n0, Vector u0, Normal nL, Vector uL, Vector R, Float length,
            bool rejectInternalIncoming, bool reciprocal,
            TangentPlaneMode tangentMode, ZvMode zvMode,
            bool useEffectiveBRDF = false,
            DipoleMode dipoleMode = ERealAndVirt) const;

    /// Returns the sample weight
    Float sampleLengthDipole(
            const Vector &uL, const Vector &nL, const Vector &R,
            const Vector *u0, const Vector &n0,
            TangentPlaneMode tangentMode, Float &s, Sampler *sampler) const;
    Float pdfLengthDipole(
            const Vector &uL, const Vector &nL, const Vector &R,
            const Vector *u0, const Vector &n0,
            TangentPlaneMode tangentMode, Float s) const;

    /// Returns the pdf
    Float sampleDirectionDipole(
            Vector &u0, const Vector &n0, const Vector &uL, const Vector &nL,
            const Vector &R, Float s, TangentPlaneMode tangentMode,
            bool useEffectiveBRDF, Sampler *sampler) const;
    Float pdfDirectionDipole(
            const Vector &u0, const Vector &n0, const Vector &uL, const Vector &nL,
            const Vector &R, Float s, TangentPlaneMode tangentMode,
            bool useEffectiveBRDF) const;

    Float evalMonopole(Vector u0, Vector uL, Vector R, Float length) const;

    Float evalPlaneSource(Vector u0, Vector uL,
            Vector n, Float Rz, Float length) const;

protected:
    void calcValues(double length, double &C, double &D, double &E, double &F,
            double *Z=NULL) const;
    double absorptionAndNormalizationConstant(Float theLength) const;

    bool getVirtualDipoleSource(
            Normal n0, Vector u0,
            Normal nL, Vector uL,
            Vector R, Float length,
            bool rejectInternalIncoming,
            TangentPlaneMode tangentMode,
            ZvMode zvMode,
            Vector &u0_virt, Vector &R_virt,
            Vector *optional_n0_effective = NULL) const;

    bool getTentativeIndexMatchedVirtualSourceDisp(
            Normal n0,
            Normal nL, Vector uL,
            Vector R,
            Float s,
            TangentPlaneMode tangentMode,
            Vector &R_virt,
            Vector *optional_n0_effective = NULL,
            Float *optional_realSourceRelativeWeight = NULL) const;


    /// Returns the pdf
    Float sampleLengthShortLimit(
            Vector R, const Vector *u0, Vector uL, Float &s, Sampler *sampler) const;
    Float pdfLengthShortLimit(
            Vector R, const Vector *u0, Vector uL, Float s) const;
    void implLengthShortLimit(
            Vector R, const Vector *u0, Vector uL, Float &s, Sampler *sampler, Float *pdf) const;
    void implLengthShortLimitKnownU0(
            Vector R, Vector u0, Vector uL, Float &s, Sampler *sampler, Float *pdf) const;
    void implLengthShortLimitMargOverU0(
            Vector R, Vector uL, Float &s, Sampler *sampler, Float *pdf) const;
    void implLengthShortLimitMargOverU0_internal(
            Vector R, Vector uL, Float &s, Sampler *sampler, Float *pdf, Float safetyFac) const;
    void implLengthShortLimitMargOverU0_oldVersion(
            Vector R, Vector uL, Float &s, Sampler *sampler, Float *pdf) const;

    /// Returns the pdf
    Float sampleLengthLongLimit(
            Vector R, Vector uL, Float &s, Sampler *sampler) const;
    Float pdfLengthLongLimit(
            Vector R, Vector uL, Float s) const;

    /// Returns the pdf
    Float sampleLengthAbsorption(
            Float &s, Sampler *sampler) const;
    Float pdfLengthAbsorption(
            Float s) const;


    /// Returns the pdf
    Float sampleDirectionBoundaryAwareMonopole_BRDF(
            Vector &u0, const Vector &n0, const Vector &uL, const Vector &R,
            Float s, Sampler *sampler) const;
    Float pdfDirectionBoundaryAwareMonopole_BRDF(
            const Vector &u0, const Vector &n0, const Vector &uL, const Vector &R,
            Float s) const;
    void implDirectionBoundaryAwareMonopole_BRDF(
            Vector &u0, const Vector &n0, const Vector &uL, const Vector &R,
            Float s, Sampler *sampler, Float *pdf) const;

    /// Returns the pdf
    Float sampleDirectionBoundaryAwareMonopole(
            Vector &u0, const Vector &n0, const Vector &uL, const Vector &R,
            Float s, bool useEffectiveBRDF, Sampler *sampler) const;
    Float pdfDirectionBoundaryAwareMonopole(
            const Vector &u0, const Vector &n0, const Vector &uL, const Vector &R,
            Float s, bool useEffectiveBRDF) const;

    /// Returns the pdf
    Float sampleDirectionBoundaryAwareMonopole_orig(
            Vector &u0, const Vector &n0, const Vector &uL, const Vector &R,
            Float s, Sampler *sampler) const;
    Float pdfDirectionBoundaryAwareMonopole_orig(
            const Vector &u0, const Vector &n0, const Vector &uL, const Vector &R,
            Float s) const;

    /// Returns the pdf
    Float sampleDirectionBoundaryAwareMonopole_bis(
            Vector &u0, const Vector &n0, const Vector &uL, const Vector &R,
            Float s, Sampler *sampler) const;
    Float pdfDirectionBoundaryAwareMonopole_bis(
            const Vector &u0, const Vector &n0, const Vector &uL, const Vector &R,
            Float s) const;
    void implDirectionBoundaryAwareMonopole_bis(
            Vector &u0, const Vector &n0, const Vector &uL, const Vector &R,
            Float s, Sampler *sampler, Float *pdf) const;

    const Float mu; /// Gaussian angle phase function standard deviation
    const Float sigma_s; /// Scattering coefficient of medium
    const Float sigma_a; /// Absorption coefficient of medium

    /**
     * Bit of a hack for index-MISmatched dipole configurations. This makes
     * the dipole refract its directions and change the virtual source
     * displacement (as determined by the Zvmode). This are 'implicit'
     * boundary conditions, as opposed to an explicit 'index matched'
     * (m_eta = 1) coupling to a proper BSDF as boundary.  */
    const Float m_eta;
};

MTS_NAMESPACE_END


#include "fwdscat_impl.h"


#endif /* __MITSUBA_MEDIUM_FWDSCAT_H_ */
