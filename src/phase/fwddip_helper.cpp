#include <mitsuba/render/phase.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/quad.h>
#include <gsl/gsl_sf_lambert.h>

MTS_NAMESPACE_BEGIN

/*!\plugin{fwddip_helper}{Helper function for the Forward Scattering Dipole}
 * \order{2}
 * \parameters{
 *     \parameter{g}{\Float}{
 *       This parameter is analogous to the $g$ parameter of the
 *       Henyey-Greenstein phase function. It gives the mean cosine of the
 *       scattering angle, but this Gaussian phase function demands that
 *       $g$ is sufficiently close to 1 (it only makes sense for strongly
 *       forward scattering).
 *     }
 * }
 * This plugin implements a simple gaussian phase function on the
 * scattering angle. It is assumed that the phase function is strongly
 * forward scattering ($g$ approaching unity).
 */
class FwdDipHelper : public PhaseFunction {
public:
    FwdDipHelper(const Properties &props)
        : PhaseFunction(props) {
    }

    FwdDipHelper(Stream *stream, InstanceManager *manager)
        : PhaseFunction(stream, manager) {
        configure();
    }

    virtual ~FwdDipHelper() { }

    void serialize(Stream *stream, InstanceManager *manager) const {
        PhaseFunction::serialize(stream, manager);
    }

    void configure() {
        PhaseFunction::configure();
        m_type = EAngleDependence;
    }

    inline Float sample(PhaseFunctionSamplingRecord &pRec,
            Sampler *sampler) const {
        Float u = sampler->next1D();
        //Float cosTheta = 1 - 2./9.*exp(gsl_sf_lambert_Wm1((u-1)/M_E) + 1);
        Float cosTheta = 1 - 2./3.*exp(gsl_sf_lambert_Wm1((u-1)/M_E) + 1);
        //Float cosTheta = (1 - 2*u + 4*sqrt(u)) / 3.f; // ps weight: 1/sqrt(ps)

        if (cosTheta < -1.001 || cosTheta > 1.001)
            Log(EWarn, "Numerical instability in sampling cosTheta: %f", cosTheta);

        cosTheta = math::clamp(cosTheta, (Float) -1, (Float) 1); // For safety
        Float sinTheta = sqrt(1 - math::square(cosTheta));

        Float sinPhi, cosPhi;
        math::sincos(TWO_PI*sampler->next1D(), &sinPhi, &cosPhi);

        pRec.wo = Frame(-pRec.wi).toWorld(Vector(
            sinTheta * cosPhi,
            sinTheta * sinPhi,
            cosTheta
        ));

        return 1.0f;
    }

    Float sample(PhaseFunctionSamplingRecord &pRec,
            Float &pdf, Sampler *sampler) const {
        sample(pRec, sampler);
        pdf = eval(pRec);
        return 1.0f;
    }

    Float eval(const PhaseFunctionSamplingRecord &pRec) const {
        Float cosTheta = math::clamp(-dot(pRec.wi, pRec.wo),
                (Float) -1, (Float) 1);

        //if (cosTheta <= 7./9.)
        if (cosTheta <= 1./3.)
            return 0.0f;
        if (cosTheta == 1)
            return 0.0f;

        //Float pdf = log(16*sqrt(2)/19683) - 9./2.*log(1 - cosTheta);
        Float pdf = log(2.*sqrt(6.)/9.) - 3./2.*log(1 - cosTheta);
        //Float pdf = 3./2.*(2 - sqrt(6 - 6*cosTheta)) / sqrt(6 - 6*cosTheta); // ps weight: 1/sqrt(ps)
        pdf /= TWO_PI;
        Assert(std::isfinite(pdf));
        Assert(pdf >= 0);
        return pdf;
    }

    Float getMeanCosine() const {
        Log(EError, "TODO");
        return -1;
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "FwdDipHelper";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_S(FwdDipHelper, false, PhaseFunction)
MTS_EXPORT_PLUGIN(FwdDipHelper,
        "Helper phase function for FwdDip");
MTS_NAMESPACE_END
