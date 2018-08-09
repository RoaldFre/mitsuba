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
#include <mitsuba/core/statistics.h>

MTS_NAMESPACE_BEGIN

static StatsCounter avgPathLength("Volumetric path tracer", "Average path length", EAverage);

/*!\plugin[volpathsimple]{volpath\_simple}{Simple volumetric path tracer}
 * \order{3}
 * \parameters{
 *     \parameter{maxDepth}{\Integer}{Specifies the longest path depth
 *         in the generated output image (where \code{-1} corresponds to $\infty$).
 *         A value of \code{1} will only render directly visible light sources.
 *         \code{2} will lead to single-bounce (direct-only) illumination,
 *         and so on. \default{\code{-1}}
 *     }
 *     \parameter{rrDepth}{\Integer}{Specifies the minimum path depth, after
 *        which the implementation will start to use the ``russian roulette''
 *        path termination criterion. \default{\code{5}}
 *     }
 *     \parameter{rrForcedDepth}{\Integer}{Specifies the minimum path depth, after
 *        which the implementation will force the ``russian roulette'' path
 *        termination probabilities to be less than unity. A value of \code{-1}
 *        corresponds to $\infty$.\default{\code{-1}}
 *     }
 *     \parameter{rrTargetThroughput}{\Float}{The ``russian roulette'' path
 *        termination criterion will try to keep the path weights at or
 *        above this value. When the interesting parts of the scene end up
 *        being much less bright than the light sources, setting this to a
 *        lower value can be beneficial.
 *        \default{\code{1.0}}
 *     }
 *     \parameter{strictNormals}{\Boolean}{Be strict about potential
 *        inconsistencies involving shading normals? See
 *        page~\pageref{sec:strictnormals} for details.
 *        \default{no, i.e. \code{false}}
 *     }
 *     \parameter{hideEmitters}{\Boolean}{Hide directly visible emitters?
 *        See page~\pageref{sec:hideemitters} for details.
 *        \default{no, i.e. \code{false}}
 *     }
 * }
 *
 * This plugin provides a basic volumetric path tracer that can be used to
 * compute approximate solutions of the radiative transfer equation. This
 * particular integrator is named ``simple'' because it does not make use of
 * multiple importance sampling. This results in a potentially
 * faster execution time. On the other hand, it also means that this
 * plugin will likely not perform well when given a scene that contains
 * highly glossy materials. In this case, please use \pluginref{volpath}
 * or one of the bidirectional techniques.
 *
 * This integrator has special support for \emph{index-matched} transmission
 * events (i.e. surface scattering events that do not change the direction
 * of light). As a consequence, participating media enclosed by a stencil shape (see
 * \secref{shapes} for details) are rendered considerably more efficiently when this
 * shape has \emph{no}\footnote{this is what signals to Mitsuba that the boundary is
 * index-matched and does not interact with light in any way. Alternatively,
 * the \pluginref{mask} and \pluginref{thindielectric} BSDF can be used to specify
 * index-matched boundaries that involve some amount of interaction.} BSDF assigned
 * to it (as compared to, say, a \pluginref{dielectric} or \pluginref{roughdielectric} BSDF).
 *
 * \remarks{
 *    \item This integrator performs poorly when rendering
 *      participating media that have a different index of refraction compared
 *      to the surrounding medium.
 *    \item This integrator has difficulties rendering
 *      scenes that contain relatively glossy materials (\pluginref{volpath} is preferable in this case).
 *    \item This integrator has poor convergence properties when rendering
 *    caustics and similar effects. In this case, \pluginref{bdpt} or
 *    one of the photon mappers may be preferable.
 * }
 */
class SimpleVolumetricPathTracer : public MonteCarloIntegrator {
protected:
    bool m_onlySingleScatter;
    bool m_noSingleScatter;
    bool m_explicitSubsurfBoundary;
public:
    SimpleVolumetricPathTracer(const Properties &props) : MonteCarloIntegrator(props) {
        m_onlySingleScatter = props.getBoolean("onlySingleScatter", false);
        m_noSingleScatter = props.getBoolean("noSingleScatter", false);
        m_explicitSubsurfBoundary = props.getBoolean("explicitSubsurfBoundary", true);
        if (m_onlySingleScatter && m_noSingleScatter) {
            Log(EError, "Conflicting options: onlySingleScatter and noSingleScatter are both active!");
        }
    }

    /// Unserialize from a binary data stream
    SimpleVolumetricPathTracer(Stream *stream, InstanceManager *manager)
     : MonteCarloIntegrator(stream, manager) {
        m_onlySingleScatter = stream->readBool();
        m_noSingleScatter = stream->readBool();
        m_explicitSubsurfBoundary = stream->readBool();
     }

    void serialize(Stream *stream, InstanceManager *manager) const {
        MonteCarloIntegrator::serialize(stream, manager);
        stream->writeBool(m_onlySingleScatter);
        stream->writeBool(m_noSingleScatter);
        stream->writeBool(m_explicitSubsurfBoundary);
    }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        MediumSamplingRecord mRec;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        bool nullChain = true;
        bool scattered = rRec.depth != 1;
        int mediumInteractionChain = 0;
        Float eta = 1.0f;

        /* Perform the first ray intersection (or ignore if the
           intersection has already been provided). */
        rRec.rayIntersect(ray);
        Spectrum throughput(1.0f);

        if (m_maxDepth == 1)
            rRec.type &= RadianceQueryRecord::EEmittedRadiance;

        /**
         * Note: the logic regarding maximum path depth may appear a bit
         * strange. This is necessary to get this integrator's output to
         * exactly match the output of other integrators under all settings
         * of this parameter.
         */
        while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
            /* ==================================================================== */
            /*                 Radiative Transfer Equation sampling                 */
            /* ==================================================================== */
            if (rRec.medium && rRec.medium->sampleDistance(Ray(ray, 0, its.t), mRec,
                    rRec.sampler, &throughput)) {
                if (m_onlySingleScatter && mediumInteractionChain >= 1)
                    break;
                mediumInteractionChain++;
                /* Sample the integral
                   \int_x^y tau(x, x') [ \sigma_s \int_{S^2} \rho(\omega,\omega') L(x,\omega') d\omega' ] dx'
                */
                const PhaseFunction *phase = rRec.medium->getPhaseFunction();

                throughput *= mRec.sigmaS * mRec.transmittance / mRec.pdfSuccess;

                /* ==================================================================== */
                /*                     Direct illumination sampling                     */
                /* ==================================================================== */

                /* Estimate the single scattering component if this is requested */
                if (rRec.type & RadianceQueryRecord::EDirectMediumRadiance
                        && !(m_onlySingleScatter && mediumInteractionChain != 1)
                        && !(m_noSingleScatter && mediumInteractionChain == 1)) {
                    DirectSamplingRecord dRec(mRec.p, mRec.time);
                    int maxInteractions = m_maxDepth - rRec.depth - 1;

                    Spectrum value = scene->sampleAttenuatedEmitterDirect(
                            dRec, rRec.medium, maxInteractions,
                            rRec.nextSample2D(), rRec.sampler);

                    if (!value.isZero())
                        Li += throughput * value * phase->eval(
                                PhaseFunctionSamplingRecord(mRec, -ray.d, dRec.d));
                }

                /* Stop if multiple scattering was not requested, or if the path gets too long */
                if ((rRec.depth + 1 >= m_maxDepth && m_maxDepth > 0) ||
                    !(rRec.type & RadianceQueryRecord::EIndirectMediumRadiance))
                    break;

                /* ==================================================================== */
                /*             Phase function sampling / Multiple scattering            */
                /* ==================================================================== */

                PhaseFunctionSamplingRecord pRec(mRec, -ray.d);
                Float phaseVal = phase->sample(pRec, rRec.sampler);
                if (phaseVal == 0)
                    break;
                throughput *= phaseVal;

                /* Trace a ray in this direction */
                ray = Ray(mRec.p, pRec.wo, ray.time);
                ray.mint = 0;
                scene->rayIntersect(ray, its);
                nullChain = false;
                scattered = true;
            } else {
                if (m_onlySingleScatter && mediumInteractionChain != 0 && mediumInteractionChain != 1)
                    break;
                if (m_noSingleScatter && mediumInteractionChain == 1)
                    break;
                mediumInteractionChain = 0;

                /* Sample
                    tau(x, y) * (Surface integral). This happens with probability mRec.pdfFailure
                    Account for this and multiply by the proper per-color-channel transmittance.
                */

                if (rRec.medium)
                    throughput *= mRec.transmittance / mRec.pdfFailure;

                if (!its.isValid()) {
                    /* If no intersection could be found, possibly return
                       attenuated radiance from a background luminaire */
                    if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
                        && (!m_hideEmitters || scattered)) {
                        Spectrum value = throughput * scene->evalEnvironment(ray);
                        if (rRec.medium)
                            value *= rRec.medium->evalTransmittance(ray);
                        Li += value;
                    }
                    break;
                }

                /* Possibly include emitted radiance if requested */
                if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
                    && (!m_hideEmitters || scattered))
                    Li += throughput * its.Le(-ray.d);

                /* Include radiance from a subsurface integrator if requested */
                if (its.hasSubsurface() && (!m_explicitSubsurfBoundary || !its.hasLiSubsurface())
                        && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
                    Li += throughput * its.LoSub(scene, rRec.sampler, -ray.d, throughput, rRec.depth);

                /* Prevent light leaks due to the use of shading normals */
                Float wiDotGeoN = -dot(its.geoFrame.n, ray.d),
                      wiDotShN  = Frame::cosTheta(its.wi);
                if (m_strictNormals && wiDotGeoN * wiDotShN < 0)
                    break;

                /* ==================================================================== */
                /*                     Direct illumination sampling                     */
                /* ==================================================================== */

                const BSDF *bsdf = its.getBSDF(ray);

                /* Estimate the direct illumination if this is requested */
                if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance &&
                        (bsdf->getType() & BSDF::ESmooth)) {
                    DirectSamplingRecord dRec(its);
                    int maxInteractions = m_maxDepth - rRec.depth - 1;

                    Spectrum value = scene->sampleAttenuatedEmitterDirect(
                            dRec, its, rRec.medium, maxInteractions,
                            rRec.nextSample2D(), rRec.sampler);

                    if (!value.isZero()) {
                        /* Allocate a record for querying the BSDF */
                        BSDFSamplingRecord bRec(its, its.toLocal(dRec.d));
                        bRec.sampler = rRec.sampler;

                        Float woDotGeoN = dot(its.geoFrame.n, dRec.d);
                        /* Prevent light leaks due to the use of shading normals */
                        if (!m_strictNormals ||
                            woDotGeoN * Frame::cosTheta(bRec.wo) > 0)
                            Li += throughput * value * bsdf->eval(bRec);
                    }
                }

                /* ==================================================================== */
                /*                   BSDF sampling / Multiple scattering                */
                /* ==================================================================== */

                /* Sample BSDF * cos(theta) */
                BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
                Spectrum bsdfVal = bsdf->sample(bRec, rRec.nextSample2D());
                if (bsdfVal.isZero())
                    break;

                /* Recursively gather indirect illumination? */
                int recursiveType = 0;
                if ((rRec.depth + 1 < m_maxDepth || m_maxDepth < 0) &&
                    (rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
                    recursiveType |= RadianceQueryRecord::ERadianceNoEmission;

                /* Recursively gather direct illumination? This is a bit more
                   complicated by the fact that this integrator can create connection
                   through index-matched medium transitions (ENull scattering events) */
                if ((rRec.depth < m_maxDepth || m_maxDepth < 0) &&
                    (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) &&
                    (bRec.sampledType & BSDF::EDelta) &&
                    (!(bRec.sampledType & BSDF::ENull) || nullChain)) {
                    recursiveType |= RadianceQueryRecord::EEmittedRadiance;
                    nullChain = true;
                } else {
                    nullChain &= bRec.sampledType == BSDF::ENull;
                }

                /* Potentially stop the recursion if there is nothing more to do */
                if (recursiveType == 0)
                    break;
                rRec.type = recursiveType;

                /* Prevent light leaks due to the use of shading normals */
                const Vector wo = its.toWorld(bRec.wo);
                Float woDotGeoN = dot(its.geoFrame.n, wo);
                if (woDotGeoN * Frame::cosTheta(bRec.wo) <= 0 && m_strictNormals)
                    break;

                /* Keep track of the throughput, medium, and relative
                   refractive index along the path */
                throughput *= bsdfVal;
                eta *= bRec.eta;
                if (its.isMediumTransition())
                    rRec.medium = its.getTargetMedium(wo);

                /* If we cross the surface and have a subsurface integrator 
                 * that can sample Li behind it, then return it if 
                 * requested or stop if it is not requested.
                 * Don't check for cosTheta<0 because the subsurface 
                 * integrator may allow 'incoming' outgoing directions, 
                 * (e.g. DirectSamplingSubsurface with 
                 * allowIncomingOutgoingDirections to check boundary 
                 * conditions) */
                if (its.hasSubsurface() && its.hasLiSubsurface() && m_explicitSubsurfBoundary) {
                    if (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance)
                        Li += throughput * its.LiSub(scene, rRec.sampler, wo,
                                throughput, rRec.splits, rRec.depth);

                    /* If 'outgoing' direction is away from the medium (wo 
                     * pointing into the medium) -> then we are looking at 
                     * the medium from the outside, in which case we want 
                     * the medium to be opaque apart from its explicit Li, 
                     * so we stop the tracing. */
                    if (Frame::cosTheta(bRec.wo) < 0)
                        break;
                }

                /* In the next iteration, trace a ray in this direction */
                ray = Ray(its.p, wo, ray.time);
                scene->rayIntersect(ray, its);
                scattered |= bRec.sampledType != BSDF::ENull;
            }

            rRec.depth++;
            Float q = m_rr.roulette(rRec.depth, throughput, eta, rRec.sampler);
            if (q == 0.0f)
                break;
            throughput /= q;
        }
        avgPathLength.incrementBase();
        avgPathLength += rRec.depth;
        return Li;
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "SimpleVolumetricPathTracer[" << endl
            << "  maxDepth = " << m_maxDepth << "," << endl
            << "  rr = " << m_rr.toString() << "," << endl
            << "  strictNormals = " << m_strictNormals << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_S(SimpleVolumetricPathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(SimpleVolumetricPathTracer, "Simple volumetric path tracer");
MTS_NAMESPACE_END
