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
#include <mitsuba/core/warp.h>
#include "../medium/materials.h"
#include "fwdscat.h"
#include "dipoleUtil.h"

MTS_NAMESPACE_BEGIN

/**
 * MIS balanced combination of phase functions.
 */
class PhaseCombo {
public:
    PhaseCombo() { }
    PhaseCombo(size_t size) :
            m_dist(size), m_phases(size) { }
    PhaseCombo(const std::vector<std::pair<Float, PhaseFunction*>> fs) :
            m_dist(fs.size()), m_phases(fs.size()) {
        for (auto f : fs)
            append(f.first, f.second);
        normalize();
    }

    void append(Float weight, PhaseFunction *f) {
        m_dist.append(weight);
        m_phases.push_back(f);
    }

    void normalize() {
        m_dist.normalize();
    }

    void sample(PhaseFunctionSamplingRecord &pRec, Sampler *sampler) const {
        Float _prob;
        int i = m_dist.sample(sampler->next1D(), _prob);
        SAssert(_prob > 0); /* we assume that sampling is always succesful */
        m_phases[i]->sample(pRec, sampler);
    }

    Float pdf(const PhaseFunctionSamplingRecord &pRec) const {
        Float pdf = 0;
        for (size_t i = 0; i < m_phases.size(); i++) {
            Float phasePdf = m_phases[i]->pdf(pRec);
            SAssert(std::isfinite(phasePdf));
            SAssert(std::isfinite(m_dist[i]));
            pdf += m_dist[i] * m_phases[i]->pdf(pRec);
        }
        SAssert(std::isfinite(pdf));
        SAssert(pdf >= 0);
        return pdf;
    }

protected:
    DiscreteDistribution m_dist;
    ref_vector<PhaseFunction> m_phases;
};

/* Phase function fitted to integral of multiple scatter phase function
 * integrated over 0 to 1 scatterings */
// Fitted to integral of small-ps weight, for ps=0..1
Float comboData[][2] = {
    // Fit with unmodified weight '9' in exp
    /*              weight                            mu                 */
    { .500007021624456374828679132905, 0.0876300947606325545901181156815},
    { .499992978375543625171320867095, 0.0234801123390553423893714938526}
    // Fit with modified weight '9/2' in exp
    //{ 0.511657417969959158991192292700, 0.173587737684146627907042267638},
    //{ 0.488342582030040841008807707300, 0.0456028760012490017906164939990}
};

class MTS_EXPORT_RENDER FwdDipSmallLengthRadialSampler2D : public Sampler1D {
public:
    FwdDipSmallLengthRadialSampler2D(Spectrum sigma_s, Spectrum g) :
        m_p(0.5*sigma_s*(Spectrum(1.0f) - g)) { }

    virtual bool sample(int channel, Float &r,
            Sampler *sampler, Float *thePdf = NULL) const {
        Float p = m_p[channel];
        Float u = sampler->next1D();
        r = sqrt(M_PI/6) * (1 - sqrt((1-u))) / p;
        if (thePdf)
            *thePdf = pdf(channel, r);
        return true;
    }
    virtual Float pdf(int channel, Float r) const {
        // pdf in the plane! -> includes 1/(2*pi*r) factor
        Float p = m_p[channel];
        Float r_p1 = r*p;
        Float max_r_p1 = sqrt(M_PI/6);
        if (r_p1 > max_r_p1)
            return 0;
        return p*p * sqrt(6.)*(sqrt(M_PI) - sqrt(6.)*r_p1) / (M_PI*M_PI*r_p1);
    }

    MTS_DECLARE_CLASS();
protected:
    const Spectrum m_p;
};

/* Higher than 1: confines pdf more so that its peak gets higher but we can
 * cut off too soon -- lower than 1; broader pdf, but we might be
 * undersampling the extreme peak then */
#define P_SAFETY_FACTOR 1

/**
 * \brief Sampler in the 2D plane perpendicular to the outgoing direction,
 * i.e. with the projection along the outgoing direction.
 */
class MTS_EXPORT_RENDER FwdDipSmallLengthSamplerPerpToDir : public TangentSampler2D {
public:
    FwdDipSmallLengthSamplerPerpToDir(Spectrum sigma_s, Spectrum g,
            Float pSafetyFactor = P_SAFETY_FACTOR) :
                m_p(pSafetyFactor * 0.5*sigma_s*(Spectrum(1.0f) - g)) {
        Assert(sigma_s.isFinite());
        Assert(g.isFinite());
        Assert(m_p.isFinite());
    }

    /**
     * \brief Sample a point x in the 2D plane.
     *
     * \param x The point to sample in the (tangent) plane, with respect to
     *          the intersection point. The first coordinate is along the
     *          'forward' direction, i.e. along the tangential component of
     *          the queried radiance direction.
     *          Due to symmetry, the sign of the second coordinate should be
     *          irrelevant and during sampling the sign will be randomized.
     * \param cosTheta The cosine between the query direction and the
     *                 plane normal. Taken to be positive.
     */
    virtual bool sample(int channel, Vector2 &x, Float cosTheta,
            const Vector2 &xLo, const Vector2 &xHi,
            Sampler *sampler, Float *thePdf = NULL) const {
        Assert(channel>=0);
        Float p = m_p[channel];
        if (p == 0)
            return false;
        Float phi = sampler->next1D() * TWO_PI;
        Float sinPhi,cosPhi;
        math::sincos(phi, &sinPhi, &cosPhi);
        Float r = 1 - sqrt(1 - sampler->next1D());
        x[0] = sinPhi * r / p;
        x[1] = cosPhi * r / p;
        if (thePdf)
            *thePdf = pdf(channel, x, cosTheta, xLo, xHi);
        return true;
    }

    virtual Float pdf(int channel, Vector2 x, Float cosTheta,
            const Vector2 &xLo, const Vector2 &xHi) const {
        Assert(channel>=0);
        Float p = m_p[channel];
        Float r = x.length() * p;
        if (p == 0 || r > 1)
            return 0;
        return 2*(1-r) * p*p / (TWO_PI*r); // triangle
    }

    MTS_DECLARE_CLASS();

protected:
    virtual ~FwdDipSmallLengthSamplerPerpToDir() { }
    const Spectrum m_p;
};

/*
 * Sampler in a plane that contains the outgoing direction
 */
class MTS_EXPORT_RENDER FwdDipSmallLengthSamplerAlongDir : public TangentSampler2D {
public:
    FwdDipSmallLengthSamplerAlongDir(Spectrum sigma_s, Spectrum g,
            Float pSafetyFactor = P_SAFETY_FACTOR) :
                m_p(pSafetyFactor * 0.5*sigma_s*(Spectrum(1.0f) - g)) {
        Assert(sigma_s.isFinite());
        Assert(g.isFinite());
        Assert(m_p.isFinite());
    }

    /* How much wider we make the sample area, to make sure we have covered
     * the peak properly. */
    const Float dMaxSafetyScale = 2;
    /* Maximum length along the outgoing, queried radiance direction, in
     * units of 1/p. */
    const Float Rmax = 1;

    /**
     * \brief Sample a point x in the 2D plane.
     *
     * \param x The point to sample in the (tangent) plane, with respect to
     *          the intersection point. The first coordinate is along the
     *          'forward' direction, i.e. along the tangential component of
     *          the queried radiance direction.
     *          Due to symmetry, the sign of the second coordinate should be
     *          irrelevant and during sampling the sign will be randomized.
     * \param cosTheta The cosine between the query direction and the
     *                 plane normal. Taken to be positive.
     */
    virtual bool sample(int channel, Vector2 &x, Float cosTheta,
            const Vector2 &xLo, const Vector2 &xHi,
            Sampler *sampler, Float *thePdf = NULL) const {
        Assert(channel>=0);
        Float p = m_p[channel];
        if (p == 0)
            return false;

        // We work in p=1
        Float R = Rmax * sampler->next1D();
        //Float R = Rmax * sqrt(sampler->next1D());
        //Float R = Rmax * pow(sampler->next1D(),2./3.);

        Float stddev = dMaxSafetyScale * sqrt(R*R*R/6);
        Float d = stddev * warp::squareToStdNormal(sampler->next2D()).x;

        /* We should displace ourselves backwards along the outgoing
         * direction to sample the query point! */
        x[0] = -R/p; // 'backwards'
        x[1] = d/p;

        if (thePdf)
            *thePdf = pdf(channel, x, cosTheta, xLo, xHi);
        return true;
    }

    virtual Float pdf(int channel, Vector2 x, Float cosTheta,
            const Vector2 &xLo, const Vector2 &xHi) const {
        Assert(channel>=0);
        Float p = m_p[channel];
        Float R = -x[0]*p; // displacement is backwards along the query point!
        Float d = x[1]*p;
        if (p == 0 || R < 0 || R > Rmax)
            return 0;

        Float stddev = dMaxSafetyScale * sqrt(R*R*R/6);
        Float dPdf = warp::lineToStdNormalPdf(d / stddev) / stddev;
        Float Rpdf = 1./Rmax;
        //Float Rpdf = 2*R/(Rmax*Rmax);
        //Float Rpdf = 1.5*sqrt(R / (Rmax*Rmax*Rmax));
        return Rpdf * dPdf * p*p;
    }

    MTS_DECLARE_CLASS();

protected:
    virtual ~FwdDipSmallLengthSamplerAlongDir() { }
    const Spectrum m_p;
};

inline IntersectionWeightFunc fwdDipSmallLengthWeightFunc(
        Spectrum sigma_s, Spectrum g) {
    Spectrum pSpec(0.5*sigma_s*(Spectrum(1.0f) - g));

    return [=] (const Intersection &its_in, const Intersection &its_out,
            const Vector &d_out, int i) {
        SAssert(i>=0);
        Vector rDir = its_out.p - its_in.p;
        if (rDir.length() == 0)
            return (Float) 1.0f;
        Float cosTheta = dot(d_out, normalize(rDir));
        Float p = pSpec[i];
        SAssert(p >= 0);
        if (p == 0)
            return (Float) 0.0f;
        Float R = rDir.length() * p;
        //return (Float) exp(3.f*(cosTheta - 1.f)/R)/R;
        return (Float) exp(3.f*(cosTheta - 1.f)/R)/(R*R);
    };
}


class MTS_EXPORT_RENDER PhaseFunctionSurfaceSampler : public SurfaceSampler {
public:
    PhaseFunctionSurfaceSampler(Float mu, const Spectrum &sigmaTr,
            Float retreatFactor, const IntersectionSampler *itsSampler)
            : m_sigmaTr(sigmaTr), m_retreatFactor(retreatFactor), m_itsSampler(itsSampler) {
        /* Initialize phase function */
        size_t N = sizeof(comboData)/sizeof(*comboData);

        // MIS: Forward log divergent peak
        Properties phaseProps("fwddip_helper");
        PhaseFunction *phase = static_cast<PhaseFunction *>(
                PluginManager::getInstance()->createObject(
                    MTS_CLASS(PhaseFunction), phaseProps));
        m_phase.append(1.0, phase);

        // MIS: Gaussian tail combo
        Log(EInfo, "Initializing phase function with %d components", N);
        for (size_t i = 0; i < N; i++) {
            Properties phaseProps("forward_gaussian");
            Float thisMu = comboData[i][1];
            phaseProps.setFloat("g", 1 - thisMu);
            PhaseFunction *phase = static_cast<PhaseFunction *>(
                    PluginManager::getInstance()->createObject(
                        MTS_CLASS(PhaseFunction), phaseProps));
            phase->configure();
            m_phase.append(comboData[i][0], phase);
        }

        m_phase.normalize();
    }

    virtual Float sample(const Intersection &its,
            const Vector &d_out, const Scene *scene,
            const std::vector<Shape *> &shapes,
            Intersection &newIts, int channel,
            Sampler *sampler) const {
        // TODO take channel into account for the phase function sampling...
        PhaseFunctionSamplingRecord pRec(MediumSamplingRecord(), d_out);
        m_phase.sample(pRec, sampler);
        Float phasePdf = m_phase.pdf(pRec);
        std::vector<Intersection> intersections;
        Point startPoint = getStartPoint(its, d_out);
        Float intersectionProb = m_itsSampler->sample(newIts, scene,
                startPoint, pRec.wo, its.time,
                shapes, its, d_out, channel, sampler, false);
        if (intersectionProb == 0)
            return 0.0f;

        Float r = distance(startPoint, newIts.p);
        AssertWarn(r <= m_itsSampler->getItsDistanceCutoff() * 1.01);
        Float surfaceCosine = dot(pRec.wo, newIts.geoFrame.n);
        Float absSurfCosine = math::abs(surfaceCosine);
        if (absSurfCosine <= MTS_DSS_COSINE_CUTOFF) {
            /* Sampling this surface was badly conditioned -- abandon ship here
             * and let the other sampling strategies save us from bias.
             * TODO: NOTE: we are still slightly biased in the way that
             * our pdf is now not properly normalized anymore! But these
             * occurences should only account for a tiny fraction of
             * 'integrated weight' precicely because the cosine is so
             * small. */
            return 0.0f;
        }
        Float solidAngleToAreaWeight= absSurfCosine/(r*r);
        Float pdf = solidAngleToAreaWeight * phasePdf * intersectionProb;
        if (!std::isfinite(pdf)) {
            Log(EWarn, "Invalid pdf in samplePointFromPhaseFunction: %f, "
                    "saToWeight %e, phase %e, intersect %e",
                    pdf, solidAngleToAreaWeight, phasePdf,
                    intersectionProb);
            return 0;
        }
#ifdef MTS_FWDDIP_DEBUG
        Float pdfCheck = this->pdf(its, d_out, scene, shapes, newIts, channel);
        if (fabs((pdf - pdfCheck)/pdf) > 1e-4) {
            Log(EWarn, "Inconsistent pdfs: %e vs %e, rel %e",
                    pdf,pdfCheck, (pdf-pdfCheck)/pdf);
        }
#endif
        return pdf;
    }

    virtual Float pdf(const Intersection &its,
            const Vector &d_out, const Scene *scene,
            const std::vector<Shape *> &shapes,
            const Intersection &newIts, int channel) const {
        Point startPoint = getStartPoint(its, d_out);
        Vector sampledDirection = normalize(newIts.p - startPoint);
        PhaseFunctionSamplingRecord pRec(MediumSamplingRecord(), d_out,
                sampledDirection);
        Float surfaceCosine = dot(pRec.wo, newIts.geoFrame.n);
        Float absSurfCosine = math::abs(surfaceCosine);
        if (absSurfCosine <= MTS_DSS_COSINE_CUTOFF) {
            return 0.0f;
        }
        Float phasePdf = m_phase.pdf(pRec);
        if (phasePdf <= RCPOVERFLOW)
            return 0.0f;

        Float intersectionProb = m_itsSampler->pdf(newIts, scene,
                startPoint, sampledDirection, its.time,
                shapes, its, d_out, channel, false);
        if (intersectionProb == 0)
            return 0.0f;

        Float r = distance(startPoint, newIts.p);
        AssertWarn(r <= m_itsSampler->getItsDistanceCutoff() * 1.01);
        Float solidAngleToAreaWeight = absSurfCosine/(r*r);
        Float pdf = solidAngleToAreaWeight * phasePdf * intersectionProb;
        if (!std::isfinite(pdf)) {
            return 0;
        }
        return pdf;
    }

    MTS_DECLARE_CLASS();
protected:
    Point getStartPoint(const Intersection &its, const Vector &d_out) const {
        /* So we can actually intersect ourself locally in case of planar
         * geometry: retreat a bit below the surface in the (opposite)
         * direction of the outgoing query -- d_out points outwards */
        return its.p - m_retreatFactor/m_sigmaTr.max() * d_out;
    }

    /// Virtual destructor
    virtual ~PhaseFunctionSurfaceSampler() { }

    PhaseCombo m_phase;

    const Spectrum m_sigmaTr;

    /// In fractions of the (channel-wise minimal) reduced scattering length
    const Float m_retreatFactor;

    const ref<const IntersectionSampler> m_itsSampler;
};


/*!\plugin{fwddip}{Forward Scattering Dipole subsurface scattering model}
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
 *     \parameter{g}{\Float}{
 *         Anisotropy of the phase function. The Forward Scattering Dipole
 *         model works best for strongly forward scattering media, i.e.
 *         \code{g} close to 1.
 *     }
 *     \parameter{scale}{\Float}{
 *         Optional scale factor that will be applied to the \code{sigma*} parameters.
 *         It is provided for convenience when accomodating data based on different units,
 *         or to simply tweak the density of the medium. \default{1}
 *     }
 *     \parameter{intIOR}{\Float\Or\String}{Interior index of refraction specified
 *         numerically or using a known material name.
 *         For index-mismatched media, it is advised to keep this Forward
 *         Scattering Dipole subsurface scattering model index-matched
 *         internally (i.e. \code{intIOR} = \code{extIOR}) and to couple this
 *         to a proper refractive BSDF that will explicitly handle the
 *         boundary conditions.
 *         \default{based on \code{material}}
 *     }
 *     \parameter{extIOR}{\Float\Or\String}{Exterior index of refraction specified
 *         numerically or using a known material name.
 *         For index-mismatched media, it is advised to keep this Forward
 *         Scattering Dipole subsurface scattering model index-matched
 *         internally (i.e. \code{intIOR} = \code{extIOR}) and to couple this
 *         to a proper refractive BSDF that will explicitly handle the
 *         boundary conditions.
 *         \default{based on \code{material}}
 *     }
 *     \parameter{numSIR}{\Integer}{
 *         Number of tentative Sample Importance Resampling to generate
 *         when sampling incoming query locations. This can temper outliers
 *         somewhat, in return for an increase in computation time.
 *         \default{1}
 *     }
 *     \parameter{directSampling}{\Boolean}{
 *         Use direct sampling strategies?
 *         \default{\code{true}}
 *     }
 *     \parameter{directSamplingMIS}{\Boolean}{
 *         Use Multiple Importance Sampling to combine the direct sampling
 *         strategy with indirect sampling? (Only has an effect when
 *         \code{directSampling} is \code{true})
 *         \default{\code{true}}
 *     }
 *     \parameter{singleChannel}{\Boolean}{
 *         Force the BSSRDF to only select a single colour channel?
 *         \default{\code{false}}
 *     }
 *     \parameter{cutoffNumAbsorptionLengths}{\Float}{
 *         When sampling an incoming query point on the surface of the
 *         associated geometry, don't consider points that are further away
 *         from the outgoing point than this number of absorption lengths.
 *         For `coloured' absorption, this corresponds to the longest
 *         absorption length over all channels.
 *         \default{10}
 *     }
 *     \parameter{reciprocal}{\Boolean}{
 *         Force reciprocity of the model?
 *         \default{\code{false}}
 *     }
 *     \parameter{zvMode}{\String}{
 *         The method used for determining the $z_\mathrm{v}$ displacement
 *         of the virtual source.
 *         \begin{enumerate}[(i)]
 *             \item \code{diff}:    Like the classical Jensen et al. dipole.
 *             \item \code{better}:  Like the better dipole of d'Eon.
 *             \item \code{Frisvad}: Like the directional dipole of
 *                                   Frisvad et al.
 *         \end{enumerate}
 *         \default{\texttt{diff}}
 *     }
 *     \parameter{tangentMode}{\String}{
 *         The method used for determining the normal of the tangent plane for the dipole configuration.
 *         \begin{enumerate}[(i)]
 *             \item \code{incoming}: Use the normal of the incoming point.
 *             \item \code{outgoing}: Use the normal of the outgoing point.
 *             \item \code{Frisvad}:  Use the modified tangent plane of
 *                                    the directinal dipole of Frisvad et al.
 *             \item \code{FrisvadMean}: Like the modified tangent plane of
 *                                    the directinal dipole of Frisvad et al.,
 *                                    but based on the 'average' normal at
 *                                    the incoming and outgoing points
 *                                    instead of on the incoming normal.
 *         \end{enumerate}
 *         \default{\texttt{Frisvad}}
 *     }
 *     \parameter{rejectInternalIncoming}{\Boolean}{
 *         Reject incoming directions that appear to come from
 *         \emph{inside} the medium instead of outside. This can
 *         happen due to the approximated tangent plane of the
 *         dipole model. Setting this to \code{true} helps with
 *         some overestimation, but it may cause thin edges to
 *         appear overly dark.
 *         \default{\code{true}}
 *     }
 *     \parameter{useEffectiveBRDF}{\Boolean}{
 *         Instead of using a full BSSRDF, use the associated effective
 *         BRDF approximation. This can be useful for objects that are
 *         densely scattering or far away from the camera, as the effective
 *         BRDF is much more easy to compute and its estimator has a
 *         greatly diminished variance.
 *         \default{\code{false}}
 *     }
 *     \parameter{onlyReal}{\Boolean}{
 *         Only show the contributions of the real source of the dipole.
 *         This is mainly for testing/debugging purposes.
 *         \default{\code{false}}
 *     }
 *     \parameter{onlyVirt}{\Boolean}{
 *         Only show the contributions of the (positive) virtual source of
 *         the dipole. This is mainly for testing/debugging purposes.
 *         \default{\code{false}}
 *     }
 *     \parameter{allowIncomingOutgoingDirections}{\Boolean}{
 *         Do we allow outgoing directions of the dipole model to actually
 *         point back towards the inside? This option is mostly only useful
 *         for debugging. For instance, it can be used to check the
 *         validity of the boundary condition assumptions, which state that
 *         (for an index-matched boundary) \emph{no} radiance should be
 *         emitted back to the inside of the boundary.
 *         \default{\code{false}}
 *     }
 * }
 *
 * This plugin implements the forward scattering dipole model from
 * Frederickx and Dutr\'e (`A Forward Scattering Dipole Model from a
 * Functional Integral Approximation', SIGGRAPH2017).
 *
 * This subsurface scattering model currently only works with the
 * \c volpath integrator. It is recommended to use path splitting (combined
 * with Russian roulette) and ideally with a statistically robust rendering
 * method such as provided by the \c RobustAdaptiveMC integrator, which can
 * wrap around \c volpath. Doing so should drastically increase the
 * convergence speed by lessening the effect of fireflies.
 *
 * For scenes where the geometry associated with this subsurface model is
 * less complex than the geometry of the rest of the scene, it might also
 * be worthwile to increase \c numSIR to obtain better estimates of the
 * subsurface contribution per ray.
 */
class MTS_EXPORT_RENDER FwdDip : public DirectSamplingSubsurface {
public:
    FwdDip(const Properties &props)
        : DirectSamplingSubsurface(props) {
        m_rejectInternalIncoming = props.getBoolean(
                "rejectInternalIncoming", false);
        m_reciprocal = props.getBoolean("reciprocal", false);

        std::string zvModeStr = props.getString("zvMode", "diff");
        if (zvModeStr == "diff") {
            m_zvMode = FwdScat::EClassicDiffusion;
        } else if (zvModeStr == "better") {
            m_zvMode = FwdScat::EBetterDipoleZv;
        } else if (zvModeStr == "Frisvad") {
            m_zvMode = FwdScat::EFrisvadEtAlZv;
        } else {
            Log(EError, "Unknown zvMode: %s", zvModeStr.c_str());
        }

        std::string tangentModeStr = props.getString("tangentMode", "Frisvad");
        if (tangentModeStr == "incoming") {
            m_tangentMode = FwdScat::EUnmodifiedIncoming;
        } else if (tangentModeStr == "outgoing") {
            m_tangentMode = FwdScat::EUnmodifiedOutgoing;
        } else if (tangentModeStr == "Frisvad") {
            m_tangentMode = FwdScat::EFrisvadEtAl;
        } else if (tangentModeStr == "FrisvadMean") {
            m_tangentMode = FwdScat::EFrisvadEtAlWithMeanNormal;
        } else {
            Log(EError, "Unknown tangentMode: %s", tangentModeStr.c_str());
        }

        bool onlyReal = props.getBoolean("onlyReal", false);
        bool onlyVirt = props.getBoolean("onlyVirt", false);
        if (onlyReal && onlyVirt)
            Log(EError, "Requested both *only* real and *only* virtual "
                    "contributions!");
        if (onlyReal) {
            m_dipoleMode = FwdScat::EReal;
            Log(EInfo, "Requested only contributions from real source");
        } else if (onlyVirt) {
            m_dipoleMode = FwdScat::EVirt;
            Log(EInfo, "Requested only contributions from (positive) "
                    "virtual source");
        } else {
            m_dipoleMode = FwdScat::ERealAndVirt;
        }

        m_winklerCorrection = props.getBoolean("winklerCorrection", false);

        m_useEffectiveBRDF = props.getBoolean("useEffectiveBRDF", false);

        lookupMaterial(props, m_sigmaS, m_sigmaA, m_g, &m_eta);
        Log(EInfo, "Loaded FwdDip with:\n"
                "sigma_s = %s\nsigma_a = %s\ng = %s\np= %s\neta = %f",
                m_sigmaS.toString().c_str(),
                m_sigmaA.toString().c_str(),
                m_g.toString().c_str(),
                (0.5*m_sigmaS * (Spectrum(1.0f) - m_g)).toString().c_str(),
                m_eta);

        if (m_eta != 1) {
            Log(EWarn, "You have chosen to make this Forward Scattering "
                    "Dipole use non-index matched, 'implicit' boundaries "
                    "internally (eta = %f, presumably coupled to a NULL or "
                    "index-matched BSDF). Although this is possible, it is "
                    "much advised to keep eta = 1 here and then couple this "
                    "'index matched' subsurface scattering model to a "
                    "refractive BSDF for more correct, 'explicit' boundary "
                    "conditions!", m_eta);
        }

        configure();
    }

    FwdDip(Stream *stream, InstanceManager *manager)
     : DirectSamplingSubsurface(stream, manager) {
        m_sigmaS = Spectrum(stream);
        m_sigmaA = Spectrum(stream);
        m_g = Spectrum(stream);
        m_rejectInternalIncoming = stream->readBool();
        m_reciprocal = stream->readBool();
        m_tangentMode = static_cast<FwdScat::TangentPlaneMode>(stream->readInt());
        m_zvMode = static_cast<FwdScat::ZvMode>(stream->readInt());
        m_dipoleMode = static_cast<FwdScat::DipoleMode>(stream->readInt());
        m_useEffectiveBRDF = stream->readBool();
        m_winklerCorrection = stream->readBool();
        configure();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        DirectSamplingSubsurface::serialize(stream, manager);
        m_sigmaS.serialize(stream);
        m_sigmaA.serialize(stream);
        m_g.serialize(stream);
        stream->writeBool(m_rejectInternalIncoming);
        stream->writeBool(m_reciprocal);
        stream->writeInt(m_tangentMode);
        stream->writeInt(m_zvMode);
        stream->writeInt(m_dipoleMode);
        stream->writeBool(m_useEffectiveBRDF);
        stream->writeBool(m_winklerCorrection);
    }


    /**
     * Sample lengths for all (non-zero throughput) spectral channels,
     * returns the sampling weights. Unsuccessful sampling sets the
     * length(s) to -1. */
    inline Spectrum sampleLengths(
            const Point &p_in,  const Vector &n_in,  const Vector *d_in,
            const Point &p_out, const Vector &n_out, const Vector &d_out,
            Float *lengths, const Spectrum &throughput,
            Sampler *sampler) const {
        Spectrum weights;
        Vector R = p_out - p_in;
        if (m_useEffectiveBRDF)
            Assert(R.isZero());
        if (m_fwdScat.size() == 1) {
            weights = Spectrum(m_fwdScat[0]->sampleLengthDipole(
                        d_out, n_out, R, d_in, n_in, m_tangentMode,
                        lengths[0], sampler));
            if (weights[0] == 0.0f)
                lengths[0] = -1;
        } else {
            for (int i = 0; i < SPECTRUM_SAMPLES; i++) {
                if (throughput[i] == 0) {
                    weights[i] = 0;
                    lengths[i] = -1;
                } else {
                    weights[i] = m_fwdScat[i]->sampleLengthDipole(
                                d_out, n_out, R, d_in, n_in, m_tangentMode,
                                lengths[i], sampler);
                    if (weights[i] == 0.0f)
                        lengths[i] = -1;
                }
            }
        }
        Spectrum pdf = weights.invertButKeepZero();
        return pdf;
    }

    inline Spectrum pdfLengths(
            const Point &p_in,  const Vector &n_in,  const Vector *d_in,
            const Point &p_out, const Vector &n_out, const Vector &d_out,
            const Float *lengths, const Spectrum &throughput) const {
        Spectrum pdf;
        Vector R = p_out - p_in;
        if (m_fwdScat.size() == 1) {
            if (lengths[0] == -1)
                return Spectrum(0.0f);
            pdf = Spectrum(m_fwdScat[0]->pdfLengthDipole(
                        d_out, n_out, R, d_in, n_in, m_tangentMode, lengths[0]));
        } else {
            for (int i = 0; i < SPECTRUM_SAMPLES; i++) {
                if (throughput[i] == 0 || lengths[i] == -1) {
                    pdf[i] = 0;
                } else {
                    pdf[i] = m_fwdScat[i]->pdfLengthDipole(
                                d_out, n_out, R, d_in, n_in, m_tangentMode, lengths[i]);
                }
            }
        }
        return pdf;
    }

    size_t extraParamsSize() const {
        return sizeof(ExtraParams);
    }

    Spectrum sampleExtraParams(const Scene *scene,
            const Intersection &its_out, const Vector &d_out,
            const Intersection &its_in,  const Vector *d_in,
            const Spectrum &throughput,
            void *extraParams, Sampler *sampler) const {
        Vector n_in = its_in.shFrame.n;
        Vector n_out = its_out.shFrame.n;
        Point p_in = its_in.p;
        Point p_out = its_out.p;

        Assert(dot(d_out, n_out) >= -Epsilon);
        Assert(!d_in || dot(*d_in, n_in) <= Epsilon);
        Assert(!m_useEffectiveBRDF || n_in == n_out);
        Assert(!m_useEffectiveBRDF || p_in == p_out);

        Spectrum extraParamsPdf = sampleLengths(p_in, n_in, d_in,
                p_out, n_out, d_out, getLengths(extraParams), throughput,
                sampler);
        return extraParamsPdf;
    }

    virtual Spectrum pdfExtraParams(const Scene *scene,
            const Intersection &its_out, const Vector &d_out,
            const Intersection &its_in,  const Vector *d_in,
            const Spectrum &throughput, const void *extraParams) const {
        Vector n_in = its_in.shFrame.n;
        Vector n_out = its_out.shFrame.n;
        Point p_in = its_in.p;
        Point p_out = its_out.p;

        Assert(dot(d_out, n_out) >= -Epsilon);
        Assert(!d_in || dot(*d_in, n_in) <= Epsilon);
        Assert(!m_useEffectiveBRDF || n_in == n_out);
        Assert(!m_useEffectiveBRDF || p_in == p_out);

        Spectrum extraParamsPdf = pdfLengths(p_in, n_in, d_in,
                p_out, n_out, d_out, getLengths(extraParams), throughput);
        return extraParamsPdf;
    }

    /* Mix in a bit of hemisphere sampling for safety. Not very beneficial
     * in combination with direct sampling, tough, so keep this small
     * (but not zero for safety, because there can still be an bright
     * *indirect* contribution, which happens to fall in a region where our
     * importance sampling undersamples!) */
    const Float direction_hemiSampleWeight = 0.05;

    /**
     * MIS weighting of importance sampling the transport and sampling the
     * (cosine) hemisphere */
    inline Float sampleDirection(const Vector &d_out, const Vector &n_out,
            const Vector &n_in, const Vector &R, const Float *lengths,
            Vector &d_in, const Spectrum &throughput, Sampler *sampler) const {
        Float pdfHemi, pdfImp;
        if (sampler->next1D() < direction_hemiSampleWeight) {
            pdfHemi = sampleDirectionHemisphere(d_in, n_in, sampler);
            if (pdfHemi == 0)
                return 0;
            pdfImp = pdfDirectionImportance(
                    d_out, n_out, n_in, R, lengths, d_in, throughput);
        } else {
            pdfImp = sampleDirectionImportance(
                    d_out, n_out, n_in, R, lengths, d_in, throughput, sampler);
            if (pdfImp == 0)
                return 0;
            pdfHemi = pdfDirectionHemisphere(d_in, n_in);
        }
        return direction_hemiSampleWeight * pdfHemi
                + (1 - direction_hemiSampleWeight) * pdfImp;;
    }

    inline Float pdfDirection(const Vector &d_out, const Vector &n_out,
            const Vector &n_in, const Vector &R, const Float *lengths,
            const Vector &d_in, const Spectrum &throughput) const {
        Float pdf = 0;
        pdf += direction_hemiSampleWeight
                * pdfDirectionHemisphere(d_in, n_in);
        pdf += (1 - direction_hemiSampleWeight)
                * pdfDirectionImportance(
                        d_out, n_out, n_in, R, lengths, d_in, throughput);
        return pdf;
    }

    /// Returns pdf on the (non-cosine-weighted) hemisphere
    inline Float sampleDirectionHemisphere(
            Vector &d_in, const Vector &n_in, Sampler *sampler) const {
        Frame frame(n_in); // TODO: could re-use shFrame...
        Vector d_in_local = warp::squareToCosineHemisphere(sampler->next2D());
        Float cosTheta = d_in_local.z;
        d_in_local.z *= -1; // Pointing inwards
        d_in = frame.toWorld(d_in_local);
        AssertWarn(dot(d_in, n_in) <= 0);
        return cosTheta * INV_PI;
    }

    /// Returns pdf on the (non-cosine-weighted) hemisphere
    inline Float pdfDirectionHemisphere(
            const Vector &d_in, const Vector &n_in) const {
        if (dot(d_in, n_in) >= 0)
            return 0;
        return -dot(d_in,n_in) * INV_PI;
    }

    inline Float sampleDirection(int i,
            Vector &d_in, const Vector &n_in,
            const Vector &d_out, const Vector &n_out,
            const Vector &R, Float s, Sampler *sampler) const {
        Assert(s>=0);
        Assert(dot(d_out, n_out) >= -Epsilon);
        Assert(!m_useEffectiveBRDF || n_in == n_out);
        Assert(!m_useEffectiveBRDF || R.isZero());

        Float thePdf = m_fwdScat[i]->sampleDirectionDipole(
                        d_in, n_in, d_out, n_out, R, s, m_tangentMode,
                        m_useEffectiveBRDF, sampler);
        if (thePdf == 0) {
            /* Note: Or use hemisphere sampler? (nah: if dipole sampling
             * fails, that means bssrdf evaluation will fail as well [for
             * *any* direction] -- because e.g. the modified tangent plane
             * can't be computed) */
            return 0;
        }
#ifdef MTS_FWDDIP_DEBUG
        Float pdfCheck = pdfDirection(i, d_in, n_in, d_out, n_out, R, s);
        if (math::abs(thePdf - pdfCheck)/thePdf > 1e-3)
            Log(EWarn, "Inconsistent direction pdf: %e vs %e, rel %f",
                    thePdf, pdfCheck, (thePdf-pdfCheck)/thePdf);
#endif
        Assert(thePdf > 0);
        return thePdf;
    }

    inline Float pdfDirection(int i,
            const Vector &d_in, const Vector &n_in,
            const Vector &d_out, const Vector &n_out,
            const Vector &R, Float s) const {
        Assert(s>=0);
        Assert(dot(d_out, n_out) >= -Epsilon);
        Assert(!m_useEffectiveBRDF || n_in == n_out);
        Assert(!m_useEffectiveBRDF || R.isZero());

        return m_fwdScat[i]->pdfDirectionDipole(
                            d_in, n_in, d_out, n_out, R, s, m_tangentMode,
                            m_useEffectiveBRDF);
    }

    /* Can only sample one direction, because otherwise we would be
     * branching into degenerate single-spectral-channel transport.
     *
     * We sample a non-zero throughput channel uniformly and the effective
     * pdf becomes averaged pdf over all (non-zero-throughput and
     * non-(-1)-length) channels (essentially like the MIS balance
     * heuristic). Returns the pdf */
    inline Float sampleDirectionImportance(
            const Vector &d_out, const Vector &n_out,
            const Vector &n_in, const Vector &R, const Float *lengths,
            Vector &d_in, const Spectrum &throughput, Sampler *sampler) const {
        if (m_fwdScat.size() == 1) {
            if (lengths[0] == -1) {
                return 0.0f;
            }
            return sampleDirection(0,
                    d_in, n_in, d_out, n_out, R, lengths[0], sampler);
        } else {
            /* Only consider nonzero throughput channels that have a
             * validly sampled length */
            Spectrum effectiveThroughput(throughput);
            for (int j = 0; j < SPECTRUM_SAMPLES; j++) {
                if (lengths[j] == -1)
                    effectiveThroughput[j] = 0;
            }
            int i = effectiveThroughput.sampleNonZeroChannelUniform(sampler);
            Assert(lengths[i] >= 0);
            Float pdf = sampleDirection(i,
                    d_in, n_in, d_out, n_out, R, lengths[i], sampler);
            if (pdf == 0)
                return 0;
            int N = 1; // number of nonzero throughput and non-(-1) lengths
            for (int j = 0; j < SPECTRUM_SAMPLES; j++) {
                if (i == j || effectiveThroughput[j] == 0)
                    continue;
                pdf += pdfDirection(j,
                    d_in, n_in, d_out, n_out, R, lengths[j]);
                N++;
            }
            pdf /= N;
            return pdf;
        }
    }

    inline Float pdfDirectionImportance(
            const Vector &d_out, const Vector &n_out,
            const Vector &n_in, const Vector &R, const Float *lengths,
            const Vector &d_in, const Spectrum &throughput) const {
        if (m_fwdScat.size() == 1) {
            if (lengths[0] == -1)
                return 0.0f;
            return pdfDirection(0,
                    d_in, n_in, d_out, n_out, R, lengths[0]);
        } else {
            Float pdf = 0;
            int N = 0; // number of nonzero-throughput & valid-length channels
            for (int j = 0; j < SPECTRUM_SAMPLES; j++) {
                if (throughput[j] == 0 || lengths[j] == -1)
                    continue;
                pdf += pdfDirection(j,
                    d_in, n_in, d_out, n_out, R, lengths[j]);
                N++;
            }
            if (N > 0)
                pdf /= N;
            return pdf;
        }
    }

    inline virtual Spectrum bssrdf(const Scene *scene,
            const Point &p_in,  const Vector &d_in,  const Normal &n_in,
            const Point &p_out, const Vector &d_out, const Normal &n_out,
            const void *extraParams) const {
        Assert(MTS_DSS_ALLOW_INTERNAL_INCOMING_DIR || dot(d_in, n_in) <= 0);
        Assert(m_allowIncomingOutgoingDirections || dot(d_out, n_out) >= 0);
        Spectrum result;
        for (int i = 0; i < SPECTRUM_SAMPLES; i++) {
            // Shortcut for when the given spectra are effectively 1D:
            if (m_fwdScat.size() == 1 && i > 0) {
                result[i] = result[0];
                continue;
            }

            const FwdScat *fwdScat = m_fwdScat[i].get();

            const Float *lengths = getLengths(extraParams);
            Float s = lengths[i];
            if (s == -1) {
                result[i] = 0;
                continue;
            }

            result[i] = fwdScat->evalDipole(
                    n_in, d_in, n_out, d_out, p_out - p_in, s,
                    m_rejectInternalIncoming, m_reciprocal,
                    m_tangentMode, m_zvMode, m_useEffectiveBRDF,
                    m_dipoleMode);
        }
        return result;
    }



    Spectrum sampleBssrdfDirection(const Scene *scene,
            const Intersection &its_out, const Vector &d_out,
            Intersection &its_in,        Vector       &d_in,
            const void *extraParams, const Spectrum &throughput,
            Sampler *sampler) const {
        Point  p_out = its_out.p;
        Point  p_in  = its_in.p;
        Vector n_out = its_out.shFrame.n;
        Vector n_in  = its_in.shFrame.n;

        /* Sample an incoming direction (on our side of the medium)
         * (This is the only difference with respect to the default
         * implementation in DirectSamplingSubsurface) -- TODO introduce a
         * sampleDirection function? (note: it could be that instead of
         * first sampling a point on the surface, then sampling additional
         * parameters and then sampling an outgoing direction, we want to
         * sample the direction first, then additional params, then point
         * on surface -- or additional params at very beginning or end, in
         * general ...) */
        Float directionPdf = sampleDirection(d_out, n_out, n_in, p_out - p_in,
                getLengths(extraParams), d_in, throughput, sampler);
        its_in.wi = its_in.toLocal(d_in);;
        /* d_in should point inwards! */
        if (directionPdf == 0 ||
                (!MTS_DSS_ALLOW_INTERNAL_INCOMING_DIR
                    && dot(d_in, n_in) >= 0)) {
            return Spectrum(0.0f);
        }

#ifdef MTS_FWDDIP_DEBUG
        Float pdf2 = pdfBssrdfDirection(scene, its_out, d_out, its_in,
                d_in, extraParams, throughput).average();
        if (fabs((directionPdf - pdf2)/(directionPdf + pdf2)) > 1e-3)
            SLog(EWarn, "Inconsistent pdfs: %e vs %e, rel %e",
                    directionPdf, pdf2, directionPdf/pdf2);
#endif
        return Spectrum(directionPdf);
    }

    Spectrum pdfBssrdfDirection(const Scene *scene,
            const Intersection &its_out, const Vector &d_out,
            const Intersection &its_in,  const Vector &d_in,
            const void *extraParams, const Spectrum &throughput) const {
        Vector n_out = its_out.shFrame.n;
        Vector n_in  = its_in.shFrame.n;
        if (!MTS_DSS_ALLOW_INTERNAL_INCOMING_DIR && dot(d_in, n_in) >= 0)
            return Spectrum(0.0f);
        Float directionPdf = pdfDirection(d_out, n_out, n_in,
                its_out.p - its_in.p, getLengths(extraParams), d_in,
                throughput);
        return Spectrum(directionPdf);
    }

    void configure() {
        if (m_fwdScat.size() != 0)
            return; /* We have already been configured! */

        if (m_sigmaS.max() == m_sigmaS.min()
         && m_sigmaA.max() == m_sigmaA.min()
         && m_g.max() == m_g.min()) {
            // Effective 1D problem as far as spectral channels are concerned
            m_fwdScat.resize(1);
            m_fwdScat[0] = new FwdScat(
                    m_g.min(), m_sigmaS.min(), m_sigmaA.min(), m_eta, 
                    m_winklerCorrection);
        } else {
            m_fwdScat.resize(SPECTRUM_SAMPLES);
            for (int i = 0; i < SPECTRUM_SAMPLES; i++)
                m_fwdScat[i] = new FwdScat(
                        m_g[i], m_sigmaS[i], m_sigmaA[i], m_eta,
                        m_winklerCorrection);
        }

        Spectrum sigmaSPrime = m_sigmaS * (Spectrum(1.0f) - m_g);
        Spectrum sigmaTPrime = sigmaSPrime + m_sigmaA;
        /* Effective transport extinction coefficient */
        Spectrum sigmaTr = (3 * m_sigmaA * sigmaTPrime).sqrt();

        Float mu = 1 - m_g.average();

        Spectrum p_spectrum = (0.5*sigmaSPrime);


        // No need for fancy planar samplers if we are using the effective BRDF!
        if (m_useEffectiveBRDF) {
            registerSampler(1.0, new BRDFDeltaSurfaceSampler());
        } else {
            /* MIS weights for surface sampler */
            const Float jensenWeight = 1;      /* classical dipole for large lengths */
            const Float smallLengthWeight = 1; /* dedicated sampler for small lengths */

            ref<InstanceManager> manager = new InstanceManager();

            /* Intersection sampler */
            ref<IntersectionSampler> itsSamplerExactJensenDipole =
                    new WeightIntersectionSampler(distanceWeightWrapper(
                            makeExactDiffusionDipoleDistanceWeight(
                            m_sigmaA, m_sigmaS, m_g, m_eta)),
                    m_itsDistanceCutoff);
            ref<IntersectionSampler> itsSamplerUniform =
                    new WeightIntersectionSampler(distanceWeightWrapper(
                            makeConstantDistanceWeight()),
                    m_itsDistanceCutoff);
            ref<IntersectionSampler> itsSamplerFwdDipSmallLen =
                    new WeightIntersectionSampler(
                            fwdDipSmallLengthWeightFunc(m_sigmaS, m_g),
                    m_itsDistanceCutoff);

            std::vector<std::pair<Float, const IntersectionSampler*> > is;
            is.push_back(std::make_pair(0.1, itsSamplerUniform.get()));
            is.push_back(std::make_pair(1.0, itsSamplerExactJensenDipole.get()));
            is.push_back(std::make_pair(1.0, itsSamplerFwdDipSmallLen.get()));
            ref<IntersectionSampler> itsSampler = new MISIntersectionSampler(is);


            /* Plane projection sampler */
            ref<TangentSampler2D> exactJensenDipoleSampler = new RadialSampler2D(
                    new RadialExactDipoleSampler2D(m_sigmaA, m_sigmaS, m_g, m_eta));



            // XXX XXX TODO drop?
            ref<TangentSampler2D> smallLengthSampler = new RadialSampler2D(
                    new FwdDipSmallLengthRadialSampler2D(m_sigmaS, m_g));




            std::vector<std::pair<Float, const TangentSampler2D*> > perp;
            perp.push_back(std::make_pair(smallLengthWeight,
                    new FwdDipSmallLengthSamplerPerpToDir(m_sigmaS, m_g, 1)));
            perp.push_back(std::make_pair(jensenWeight,
                    exactJensenDipoleSampler));

            std::vector<std::pair<Float, const TangentSampler2D*> > along;
            along.push_back(std::make_pair(smallLengthWeight,
                    new FwdDipSmallLengthSamplerAlongDir(m_sigmaS, m_g, 1)));
            along.push_back(std::make_pair(jensenWeight,
                    exactJensenDipoleSampler));

            ref<TangentSampler2D> lengthSampler_perp =
                    new MISTangentSampler2D(perp);
            ref<TangentSampler2D> lengthSampler_along =
                    new MISTangentSampler2D(along);


            registerSampler(1.0, new ProjSurfaceSampler(
                    DSSProjFrame::EDirectionDirection,
                    lengthSampler_perp, itsSampler.get()));
            registerSampler(1.0, new ProjSurfaceSampler(
                    DSSProjFrame::EDirectionOut,
                    lengthSampler_along, itsSampler.get()));
            registerSampler(1.0, new ProjSurfaceSampler(
                    DSSProjFrame::EDirectionSide,
                    lengthSampler_along, itsSampler.get()));

            registerSampler(1.0, new PhaseFunctionSurfaceSampler(mu, sigmaTr, 0, itsSampler));
        }

        normalizeSamplers();
    }

    bool preprocess(const Scene *scene, RenderQueue *queue,
            const RenderJob *job, int sceneResID, int cameraResID,
            int _samplerResID) {
        return DirectSamplingSubsurface::preprocess(scene, queue, job,
                sceneResID, cameraResID, _samplerResID);
    }

    void cancel() { }

    std::string toString() const {
        std::ostringstream oss;
        oss << "FwdDip[" << endl;
        oss << "  sigmaS = " << m_sigmaS.toString() << endl;
        oss << "  sigmaA = " << m_sigmaA.toString() << endl;
        oss << "  g = " << m_g.toString() << endl;
        oss << "]" << endl;
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    Spectrum m_sigmaS, m_sigmaA, m_g;
    bool m_rejectInternalIncoming;
    bool m_reciprocal;
    FwdScat::TangentPlaneMode m_tangentMode;
    FwdScat::ZvMode m_zvMode;
    FwdScat::DipoleMode m_dipoleMode;
    bool m_useEffectiveBRDF;
    bool m_winklerCorrection;
    // Initialized by configure():
    ref_vector<FwdScat> m_fwdScat;

    struct ExtraParams {
        Float lengths[SPECTRUM_SAMPLES];
    };

    static const Float* getLengths(const void *extraParams) {
        const ExtraParams &params(
                *static_cast<const ExtraParams*>(extraParams));
        return params.lengths;
    }
    static Float* getLengths(void *extraParams) {
        ExtraParams &params(*static_cast<ExtraParams*>(extraParams));
        return params.lengths;
    }
};

MTS_IMPLEMENT_CLASS_S(FwdDip, false, DirectSamplingSubsurface);
MTS_IMPLEMENT_CLASS(FwdDipSmallLengthSamplerPerpToDir, false, TangentSampler2D);
MTS_IMPLEMENT_CLASS(FwdDipSmallLengthSamplerAlongDir, false, TangentSampler2D);
MTS_IMPLEMENT_CLASS(FwdDipSmallLengthRadialSampler2D, false, Sampler1D);
MTS_IMPLEMENT_CLASS(PhaseFunctionSurfaceSampler, false, SurfaceSampler);
MTS_EXPORT_PLUGIN(FwdDip, "Forward scattering dipole model based on a "
        "functional integral approximation");
MTS_NAMESPACE_END
