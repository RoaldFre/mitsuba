#include <mitsuba/render/phase.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/quad.h>

MTS_NAMESPACE_BEGIN

/*!\plugin{forward_gaussian}{Strongly forward scattering Gaussian phase function}
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
class ForwardGaussianPhaseFunction : public PhaseFunction {
public:
    ForwardGaussianPhaseFunction(const Properties &props)
        : PhaseFunction(props) {
        m_g = props.getFloat("g", 0.9f);
        //if (m_g >= 1 || m_g <= 0)
        if (m_g >= 1 || m_g < 0) // relax the g=0 edge case for testing purpose (in principle we could also de backscattering for g in (-1,1))
            Log(EError, "The asymmetry parameter must lie in the interval "
                    "(0, 1) and ideally be close to 1!");
    }

    ForwardGaussianPhaseFunction(Stream *stream, InstanceManager *manager)
        : PhaseFunction(stream, manager) {
        m_g = stream->readFloat();
        configure();
    }

    virtual ~ForwardGaussianPhaseFunction() { }

    void serialize(Stream *stream, InstanceManager *manager) const {
        PhaseFunction::serialize(stream, manager);

        stream->writeFloat(m_g);
    }

    void configure() {
        PhaseFunction::configure();
        m_type = EAngleDependence;
        // TODO: support negative g as well
        if (m_g > 0.96) {
            // Expansion mu -> 0
            m_mu = 0.75 - 0.25*sqrt(24*m_g - 15);
        } else if (m_g < 0.023) {
            // Expansion mu -> infty
            const Float pi2 = M_PI*M_PI;
            const Float pi4 = pi2*pi2;
            m_mu = (pi2 + sqrt(pi4 - 16*pi2*m_g))/(32 * m_g);
        } else {
            // Rational Chebychev approximation
            const Float g = m_g;
            const Float g2 = g*g;
            const Float g3 = g2*g;
            m_mu = (1.67795885157-1.58282020037*g-0.951964559365e-1*g2)
            /(0.611648897428e-4+2.71722238539*g-1.42102908810*g2+.477385217310*g3);
        }
        m_samplingFactor = math::fastexp(-2/m_mu);
        Float pdfNormalization = INV_TWOPI / (m_mu * (2*sinh(1/m_mu)));
        m_pdfLogNormalization = log(pdfNormalization);
    }

    inline Float sample(PhaseFunctionSamplingRecord &pRec,
            Sampler *sampler) const {
        /* cosTheta = z is distributed according to an exponential
         * distribution exp(-(1-z)/mu) (truncated for z between -1 and 1) */

        Float cosTheta;
        Float u = sampler->next1D();
        if (1/m_mu < Epsilon) {
            /* Expansion in small 1/mu, up to second order. */
            Float u2 = u*u;
            Float u3 = u2*u;
            /* The expanded cosTheta is still guaranteed to stay within -1..1 */
            cosTheta = 2.*u - 1
                    + (-2.*u2 + 2.*u) / m_mu
                    + (8./3.*u3 - 4.*u2 + 4./3.*u) / (m_mu*m_mu);
        } else if (1/m_mu > LOG_REDUCED_PRECISION / 2) {
            cosTheta = 1 + log(u) * m_mu;
            if (cosTheta < -1) {
                /* *INSANELY* unlikely (pdf below would probably cut off to
                 * zero anyway, but universe would die of heat death first) */
                Log(EWarn, "Woah! Universe should have encountered heat death, "
                        "or code is bugged -- cosTheta: %f", cosTheta);
                cosTheta = -1;
            }
        } else {
            Assert(std::isfinite(m_samplingFactor));
            cosTheta = log(m_samplingFactor * (1 - u) + u)*m_mu + 1;
        }

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
        Float cosTheta = math::clamp(-dot(pRec.wi, pRec.wo), (Float) -1, (Float) 1);

        Float pdf;
        if (1/m_mu < Epsilon) {
            /* Expansion in small 1/mu, up to second order */
            /* The expanded pdf is still guaranteed to be >= 0 */
            pdf = INV_FOURPI * (1 + cosTheta / m_mu
                    + (0.5*cosTheta*cosTheta - 1./6.) / (m_mu*m_mu));
        } else if (1/m_mu > LOG_REDUCED_PRECISION / 2) {
            pdf = INV_TWOPI * exp((cosTheta - 1) / m_mu) / m_mu;
        } else {
            Assert(std::isfinite(m_pdfLogNormalization));
            pdf = exp(cosTheta / m_mu + m_pdfLogNormalization);
        }

        Assert(std::isfinite(pdf));
        return pdf;
    }

    Float getMeanCosine() const {
        return m_g;
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "ForwardGaussianPhaseFunction[g=" << m_g << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    Float m_g; /// mean squared cosine
    Float m_mu; /// angular variance
    Float m_pdfLogNormalization; /// Cache for normalization constant of the pdf
    Float m_samplingFactor; /// Cache for factor that gets used in the sampling routine
};

MTS_IMPLEMENT_CLASS_S(ForwardGaussianPhaseFunction, false, PhaseFunction)
MTS_EXPORT_PLUGIN(ForwardGaussianPhaseFunction,
        "Strongly forward scattering Gaussian phase function");
MTS_NAMESPACE_END
