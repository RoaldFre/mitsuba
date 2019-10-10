#include <mitsuba/render/scene.h>
#include <mitsuba/render/dss.h>
#include <mitsuba/core/plugin.h>
#include "../medium/materials.h"

MTS_NAMESPACE_BEGIN

/* Because our importance sampling is rather crude, we can choose to use 
 * more samples... */
#define MTS_DSS_DEON_DUAL_BEAM_NUM_SAMPLES 1

class DualBeamdEon : public DirectSamplingSubsurface {
public:
    DualBeamdEon(const Properties &props)
        : DirectSamplingSubsurface(props) {
        lookupMaterial(props, m_sigmaS, m_sigmaA, m_g, &m_eta);
        m_modifiedDipoleTangentPlane = props.getBoolean("modifiedDipoleTangentPlane", true);

        if (m_eta != 1)
            Log(EWarn, "ATTENTION! The dual beam model was only fitted for "
                    "index matched media! You have requested implicit "
                    "boundaries (eta = %f), which will only use Fresnel "
                    "refraction and transmission scaling, but will *not* "
                    "take into account internal Fresnel reflection (not even "
                    "implicitly)! Use explicit boundaries to obtain results "
                    "with internal reflection!", m_eta);

        m_sourcesResID = -1;
    }

    DualBeamdEon(Stream *stream, InstanceManager *manager)
     : DirectSamplingSubsurface(stream, manager) {
        m_sigmaS = Spectrum(stream);
        m_sigmaA = Spectrum(stream);
        m_g = Spectrum(stream);
        m_sourcesIndex = stream->readInt();
        m_sourcesResID = -1;
        m_modifiedDipoleTangentPlane = stream->readBool();
        configure();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        DirectSamplingSubsurface::serialize(stream, manager);
        m_sigmaS.serialize(stream);
        m_sigmaA.serialize(stream);
        m_g.serialize(stream);
        stream->writeInt(m_sourcesIndex);
        stream->writeBool(m_modifiedDipoleTangentPlane);
    }

    virtual Spectrum sampleBssrdfDirection(const Scene *scene,
            const Intersection &its_out, const Vector &d_out,
            Intersection &its_in,        Vector       &d_in,
            const void *extraParams, const Spectrum &throughput,
            Sampler *sampler) const {
        return DirectSamplingSubsurface::sampleBssrdfDirection(
                scene, its_out, d_out, its_in, d_in, extraParams,
                throughput, sampler);
    }

    virtual Spectrum pdfBssrdfDirection(const Scene *scene,
            const Intersection &its_out, const Vector &d_out,
            const Intersection &its_in,  const Vector &d_in,
            const void *extraParams, const Spectrum &throughput) const {
        return DirectSamplingSubsurface::pdfBssrdfDirection(
                scene, its_out, d_out, its_in, d_in, extraParams,
                throughput);
    }

    size_t extraParamsSize() const {
        return sizeof(ExtraParams);
    }

    Spectrum sampleExtraParams(const Scene *scene,
            const Intersection &its_out, const Vector &d_out,
            const Intersection &its_in,  const Vector *d_in,
            const Spectrum &throughput, void *extraParams,
            Sampler *sampler) const {
        /* TODO: currently simple importance sampling of the exponentials,
         * oblivious to the 1/r or 1/r^2 divergence */
        ExtraParams *params = static_cast<ExtraParams*>(extraParams);
        for (int s = 0; s < MTS_DSS_DEON_DUAL_BEAM_NUM_SAMPLES; s++) {
            Spectrum &pdf = params->samples[s].pdf;
            Float *us = params->samples[s].u;
            Float *vs = params->samples[s].v;
            for (int i = 0; i < SPECTRUM_SAMPLES; i++) {
                if (m_noSpectralDependence && i > 0) {
                    pdf[i] = pdf[0];
                    continue;
                }

                if (throughput[i] == 0.0f) {
                    us[i] = -1;
                    vs[i] = -1;
                    pdf[i] = 0;
                    continue;
                }

                Float u, v, uPdf, vPdf;
                Assert(m_uvSampler.get());
                bool success = m_uvSampler->sampleMIS(i, u, sampler, &uPdf)
                            && m_uvSampler->sampleMIS(i, v, sampler, &vPdf);
                if (!success) {
                    us[i] = -1;
                    vs[i] = -1;
                    pdf[i] = 0;
                } else {
                    us[i] = u;
                    vs[i] = v;
                    pdf[i] = uPdf * vPdf;
                }
            }
        }
        return Spectrum(1.0f); // individual pdfs are stored in extraParams
    }

    Spectrum pdfExtraParams(const Scene *scene,
            const Intersection &its_out, const Vector &d_out,
            const Intersection &its_in,  const Vector *d_in,
            const Spectrum &throughput, const void *extraParams) const {
        return Spectrum(1.0f);
    }

    virtual Spectrum bssrdf(const Scene *scene, const Point &p_in, const Vector &d_in, const Normal &n_in,
            const Point &p_out, const Vector &d_out, const Normal &n_out,
            const void *extraParams) const {
        const ExtraParams *params = static_cast<const ExtraParams*>(extraParams);
        Spectrum result(0.0f);
        for (int s = 0; s < MTS_DSS_DEON_DUAL_BEAM_NUM_SAMPLES; s++) {
            Spectrum pdf = params->samples[s].pdf;
            const Float *u = params->samples[s].u;
            const Float *v = params->samples[s].v;
            if (!pdf.isZero())
                result += bssrdfIntegrand(p_in, d_in, n_in,
                        p_out, d_out, n_out, u, v) * pdf.invertButKeepZero();
        }
        return result / MTS_DSS_DEON_DUAL_BEAM_NUM_SAMPLES;
    }

    /// u and v are distances in 'world' units, not in "mu_t (or sigmaTPrime) = 1"
    Spectrum bssrdfIntegrand(const Point &p_in, const Vector &d_in_ext, const Normal &n_in,
            const Point &p_out, const Vector &d_out_ext, const Normal &n_out,
            const Float u[SPECTRUM_SAMPLES], const Float v[SPECTRUM_SAMPLES]) const {
        Spectrum bssrdfIntegrand;

        Vector d_in_int(d_in_ext);
        Vector d_out_int(d_out_ext);
        Float fresnelTransmittance = handleImplicitBounds(d_in_int, n_in, d_out_int, n_out);
        if (fresnelTransmittance == 0.0f)
            return Spectrum(0.0f);

        for (int i = 0; i < SPECTRUM_SAMPLES; i++) {
            if (m_noSpectralDependence && i > 0) {
                bssrdfIntegrand[i] = bssrdfIntegrand[0];
                continue;
            }

            if (u[i] < 0) {
                Assert(u[i] == -1 && v[i] == -1);
                bssrdfIntegrand[i] = 0;
                continue;
            }

            Point x = p_out - d_out_int*u[i]; // detector point
            Point s = p_in  + d_in_int*v[i];  // source point

            Vector n;
            if (m_modifiedDipoleTangentPlane) {
                /* Use the modified tangent plane of the directional dipole
                 * model of Frisvad et al */
                Vector R = p_out - p_in;
                n = cross(normalize(R), normalize(cross(n_in, R)));
                Assert(dot(n, n_in) > -Epsilon);
            } else {
                n = n_in;
            }

            /* Note: Error in paper? The factors of alpha should be
             * sigmaSPrime in (1) and (2): the assumption sigmaSPrime = 1
             * has not been made there, yet. Eq (1) is also the
             * (differential) L_o contribution due to S_d instead of S_d
             * itself, because the phi definition in (2) already takes an
             * incoming radiance. */
            bssrdfIntegrand[i] = exp(-m_sigmaTPrime[i]*(u[i]+v[i]))
                    * math::square(m_sigmaSPrime[i])*INV_FOURPI
                    * pointToPointGreenFunction(x, s, n, p_out, i);
        }
        return bssrdfIntegrand * fresnelTransmittance;
    }

    /** phi_M(x,s) in paper (x: query point below surface, s: source 
     * location below surface, x_o: outgoing query point on the surface) */
    Float pointToPointGreenFunction(Point x, Point s, Vector n, Point x_o, int channel) const {
        Vector xs = s-x; // x to source s
        Vector xs_mirr = xs + 2*dot(x_o-s,n)*n; // x to mirror source wrt halfspace-plane (no z-offset)
        Float r_real = xs.length();
        Float r_v_un = (xs_mirr + m_z_un[channel]*n).length();
        Float r_v_D  = (xs_mirr + m_z_D[channel]*n).length();
        return grosjeanUncollided(r_real, channel) + grosjeanDiffuse(r_real, channel)
                - m_a_un[channel] * grosjeanUncollided(r_v_un, channel)
                - m_a_D[channel]  * grosjeanDiffuse(r_v_D,  channel);
    }

    /// G_un(r) in paper (with full units)
    Float grosjeanUncollided(Float r, int channel) const {
        return exp(-m_sigmaTPrime[channel] * r) * INV_FOURPI / (r*r);
    }

    /// G_D(r) in paper (with full units)
    Float grosjeanDiffuse(Float r, int channel) const {
        return m_CD[channel] * exp(-m_muEff[channel]*r) / r;
    }

    void configure() {
        m_sigmaSPrime = m_sigmaS * (Spectrum(1.0f) - m_g);
        m_sigmaTPrime = m_sigmaSPrime + m_sigmaA;
        m_alpha = m_sigmaSPrime / m_sigmaTPrime;
        m_CD = INV_FOURPI * (3*m_sigmaSPrime*m_sigmaTPrime) / (2*m_sigmaA + m_sigmaSPrime);
        Spectrum D = (2*m_sigmaA + m_sigmaSPrime) / (3*m_sigmaTPrime*m_sigmaTPrime);
        m_muEff = (m_sigmaA / D).sqrt();

        if (m_alpha.min() < 0.5)
            Log(EError, "Minimal alpha %e was too low (<0.5) for fitted approximations", m_alpha.min());

        // For sigmaTPrime = 1
        Spectrum sqrtAlpha = m_alpha.sqrt();
        m_z_D  = 0.335867*m_alpha*m_alpha - 0.62166*m_alpha + 0.944945/sqrtAlpha;
        m_z_un = 0.154352*m_alpha - Spectrum(0.142497f);
        for (int i = 0; i < SPECTRUM_SAMPLES; i++)
            m_z_un[i] = std::max(m_z_un[i], -0.03);
        m_a_D  = 0.359563*m_alpha*m_alpha - 0.692592*m_alpha + Spectrum(1.34954f);
        m_a_un = Spectrum(-7.7f) + (((9.8*m_alpha - Spectrum(22.8f))*m_alpha) + Spectrum(20.f))*m_alpha + 1.1/m_alpha;

        // Rescale to actual sigmaTPrime
        m_z_D  /= m_sigmaTPrime;
        m_z_un /= m_sigmaTPrime;

        if (m_sigmaS.max() == m_sigmaS.min()
         && m_sigmaA.max() == m_sigmaA.min()
         && m_g.max() == m_g.min()) {
            m_noSpectralDependence = true;
            Log(EInfo, "Scattering parameters are not spectrally dependant, using simplified method.");
        } else {
            m_noSpectralDependence = false;
        }

        Spectrum sigmaTr = (m_sigmaA * m_sigmaTPrime * 3.0f).sqrt();
        if (sigmaTr.isZero())
            Log(EError, "Subsurface integrator needs nonzero product of sigma_s and sigma_a!");

        m_uvSampler = new ExpSampler1D(m_sigmaTPrime);

        ref<RadialSampler2D> exactDipoleSampler(
                new RadialSampler2D(new RadialExactDipoleSampler2D(m_sigmaA, m_sigmaS, m_g, m_eta)));
        ref<WeightIntersectionSampler> itsSampler(
                new WeightIntersectionSampler(distanceWeightWrapper(
                    makeExactDiffusionDipoleDistanceWeight(m_sigmaA, m_sigmaS, m_g, m_eta)),
                m_itsDistanceCutoff));
        registerSampler(1, new ProjSurfaceSampler(
                DSSProjFrame::ENormalNormal,  // normal
                exactDipoleSampler.get(), itsSampler.get()));
        registerSampler(1, new ProjSurfaceSampler(
                DSSProjFrame::ENormalForward, //forward
                exactDipoleSampler.get(), itsSampler.get()));
        registerSampler(1, new ProjSurfaceSampler(
                DSSProjFrame::ENormalSide,    // side
                exactDipoleSampler.get(), itsSampler.get()));
        normalizeSamplers();
    }

    bool preprocess(const Scene *scene, RenderQueue *queue, const RenderJob *job,
            int sceneResID, int cameraResID, int _samplerResID) {
        return DirectSamplingSubsurface::preprocess(scene, queue, job, sceneResID, cameraResID, _samplerResID);
    }

    void cancel() { }

    std::string toString() const {
        std::ostringstream oss;
        oss << "DualBeamdEon[" << endl;
        oss << "  sigmaS = " << m_sigmaS.toString() << endl;
        oss << "  sigmaA = " << m_sigmaA.toString() << endl;
        oss << "  g = " << m_g.toString() << endl;
        oss << "]" << endl;
        return oss.str();
    }

    MTS_DECLARE_CLASS()

private:
    struct ExtraParamsEntry {
        Float u[SPECTRUM_SAMPLES], v[SPECTRUM_SAMPLES]; // dimension of length
        Spectrum pdf;
    };
    struct ExtraParams {
        ExtraParamsEntry samples[MTS_DSS_DEON_DUAL_BEAM_NUM_SAMPLES];
    };

    Spectrum m_sigmaS, m_sigmaA, m_g;
    Spectrum m_sigmaSPrime, m_sigmaTPrime;
    Spectrum m_alpha;
    Spectrum m_CD, m_muEff; // dimensionful
    Spectrum m_z_un, m_z_D; // dimensionful
    Spectrum m_a_un, m_a_D;
    bool m_noSpectralDependence;
    bool m_modifiedDipoleTangentPlane;
    ref<const Sampler1D> m_uvSampler;
};

MTS_IMPLEMENT_CLASS_S(DualBeamdEon, false, DirectSamplingSubsurface)
MTS_EXPORT_PLUGIN(DualBeamdEon, "The dual-beam 3D searchlight BSSRDF of Eugene d'Eon");
MTS_NAMESPACE_END
