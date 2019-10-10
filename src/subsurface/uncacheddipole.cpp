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

#include <mitsuba/render/scene.h>
#include <mitsuba/render/dss.h>
#include <mitsuba/core/plugin.h>
#include "../medium/materials.h"

MTS_NAMESPACE_BEGIN


/*!\plugin{uncacheddipole}{Dipole-based subsurface scattering model without irradiance cache}
 * \parameters{
 *     \parameter{material}{\String}{
 *         Name of a material preset, see
 *         \tblref{medium-coefficients}. \default{\texttt{skin1}}
 *     }
 *     \parameter{sigmaA, sigmaS}{\Spectrum}{
 *         Absorption and scattering
 *         coefficients of the medium in inverse scene units.
 *         These parameters are mutually exclusive with \code{sigmaT} and \code{albedo}
 *         \default{configured based on \code{material}}
 *     }
 *     \parameter{sigmaT, albedo}{\Spectrum}{
 *         Extinction coefficient in inverse scene units
 *         and a (unitless) single-scattering albedo.
 *         These parameters are mutually exclusive with \code{sigmaA} and \code{sigmaS}
 *         \default{configured based on \code{material}}
 *     }
 *     \parameter{scale}{\Float}{
 *         Optional scale factor that will be applied to the \code{sigma*} parameters.
 *         It is provided for convenience when accomodating data based on different units,
 *         or to simply tweak the density of the medium. \default{1}}
 *     \parameter{intIOR}{\Float\Or\String}{Interior index of refraction specified
 *      numerically or using a known material name. \default{based on \code{material}}}
 *     \parameter{extIOR}{\Float\Or\String}{Exterior index of refraction specified
 *      numerically or using a known material name. \default{based on \code{material}}}
 *     \parameter{numSamples}{\Integer}{
 *         Number of samples to use when estimating the irradiance
 *         at a point on the surface. \default{1}
 *     }
 *     \parameter{useClosestProjectedPoint}{\Boolean}{
 *         If set to \c true: only look for gather points that are the
 *         closest projection to the surface from a sampled point on the
 *         tangential plane at the query point. If set to false: find all
 *         intersections and allow all of them to contribute to the
 *         subsurface transport. This violates the halfinfinite medium
 *         assumption somewhat more, but it is consistent with a dipole
 *         solution where the irradiance of points on the surface is
 *         computed in a preprocessing step and all this points are allowed
 *         to contribute.
 *
 *         Note: even when setting this to \c false, there is an edge case
 *         that causes an inability to render transport from light entering
 *         at planes perpendicular to the tangent frame of the query point.
 *         This is not so much a problem for natural/organic scenes, but it
 *         can be problematic for man-made/architectural objects with
 *         planes at right angles. \default{false}
 *     }
 * }
 *
 * This plugin implements the initial classical dipole due to Jensen
 * et.\ al. The difference with the regular dipole plugin is that the
 * uncacheddipole does not use an irradiance cache but samples the
 * irradiance during the path tracing itself. For more information, see the
 * documentation of the dipole plugin.
 */

class IsotropicDipoleUncached : public DirectSamplingSubsurface {
public:
    IsotropicDipoleUncached(const Properties &props)
        : DirectSamplingSubsurface(props) {

        lookupMaterial(props, m_sigmaS, m_sigmaA, m_g, &m_eta);

        m_sourcesResID = -1;
    }

    IsotropicDipoleUncached(Stream *stream, InstanceManager *manager)
     : DirectSamplingSubsurface(stream, manager) {
        m_sigmaS = Spectrum(stream);
        m_sigmaA = Spectrum(stream);
        m_g = Spectrum(stream);
        m_sourcesIndex = stream->readInt();
        m_sourcesResID = -1;
        configure();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        DirectSamplingSubsurface::serialize(stream, manager);
        m_sigmaS.serialize(stream);
        m_sigmaA.serialize(stream);
        m_g.serialize(stream);
        stream->writeInt(m_sourcesIndex);
    }

    virtual Spectrum bssrdf(const Scene *scene, const Point &p_in, const Vector &d_in, const Normal &n_in,
            const Point &p_out, const Vector &d_out, const Normal &n_out,
            const void *extraParams) const {
        Spectrum rSqr = Spectrum((p_in - p_out).lengthSquared());

        /* Distance to the real source */
        Spectrum dr = (rSqr + m_zr*m_zr).sqrt();

        /* Distance to the image point source */
        Spectrum dv = (rSqr + m_zv*m_zv).sqrt();

        Spectrum C1 = m_zr * (m_sigmaTr + Spectrum(1.0f) / dr);
        Spectrum C2 = m_zv * (m_sigmaTr + Spectrum(1.0f) / dv);

        /* Do not include the reduced albedo - will be canceled out later */
        Spectrum dMo = Spectrum(INV_FOURPI) *
             (C1 * ((-m_sigmaTr * dr).exp()) / (dr * dr)
            + C2 * ((-m_sigmaTr * dv).exp()) / (dv * dv));

        /* Mask off the channels which have no contribution (infinite mean
         * free path, will end up as nan) */
        for (int i = 0; i < SPECTRUM_SAMPLES; i++) {
            if (!std::isfinite(m_zr[i]))
                dMo[i] = 0.0f;
        }

        /* For eta != 1: modulate with Fresnel transmission.
         * NOTE: The explicit boundary should be a index-matched or null
         * boundary anyway for this to make sense */
        if (m_eta != 1.0f) {
            dMo *= 1.0f - fresnelDielectricExt(dot(n_out, d_out), m_eta);
            dMo *= 1.0f - fresnelDielectricExt(dot(n_in,  -d_in), m_eta);
        }

        return dMo * INV_PI;
    }

    void configure() {
        m_sigmaSPrime = m_sigmaS * (Spectrum(1.0f) - m_g);
        m_sigmaTPrime = m_sigmaSPrime + m_sigmaA;

        Spectrum mfp = Spectrum(1.0f) / m_sigmaTPrime;

        /* Average diffuse reflectance due to mismatched indices of refraction */
        // in 'to out' (but reflected back inwards to medium) -> 1/eta = extIOR/intIOR
        m_Fdr = fresnelDiffuseReflectance(1 / m_eta);

        /* Dipole boundary condition distance term */
        Float A = (1 + m_Fdr) / (1 - m_Fdr);

        /* Effective transport extinction coefficient */
        m_sigmaTr = (m_sigmaA * m_sigmaTPrime * 3.0f).sqrt();
        if (m_sigmaTr.isZero())
            Log(EError, "Subsurface integrator needs nonzero product of sigma_s and sigma_a!");

        /* Distance of the two dipole point sources to the surface */
        m_zr = mfp;
        m_zv = mfp * (1.0f + 4.0f/3.0f * A);

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
        return 0;
    }

    Spectrum sampleExtraParams(const Scene *scene,
            const Intersection &its_out, const Vector &d_out,
            const Intersection &its_in,  const Vector *d_in,
            const Spectrum &throughput, void *extraParams,
            Sampler *sampler) const {
        return Spectrum(1.0f);
    }

    Spectrum pdfExtraParams(const Scene *scene,
            const Intersection &its_out, const Vector &d_out,
            const Intersection &its_in,  const Vector *d_in,
            const Spectrum &throughput, const void *extraParams) const {
        return Spectrum(1.0f);
    }

    void cancel() { }

    std::string toString() const {
        std::ostringstream oss;
        oss << "IsotropicDipoleUncached[" << endl;
        oss << "  sigmaS = " << m_sigmaS.toString() << endl;
        oss << "  sigmaA = " << m_sigmaA.toString() << endl;
        oss << "  g = " << m_g.toString() << endl;
        oss << "]" << endl;
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    Float m_Fdr;
    Spectrum m_sigmaS, m_sigmaA, m_g;
    Spectrum m_sigmaTr, m_zr, m_zv;
    Spectrum m_sigmaSPrime, m_sigmaTPrime;
};

MTS_IMPLEMENT_CLASS_S(IsotropicDipoleUncached, false, DirectSamplingSubsurface)
MTS_EXPORT_PLUGIN(IsotropicDipoleUncached, "Uncached isotropic dipole model");
MTS_NAMESPACE_END
