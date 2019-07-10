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
#include <mitsuba/core/lock.h>
#include <mitsuba/core/sched.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/statistics.h>
#include <boost/math/tools/roots.hpp>
#include "../medium/materials.h"

MTS_NAMESPACE_BEGIN

/* Debugging flag to dump information on which sampling steps are the
 * largest sources of variance. */
#define MTS_DSS_CHECK_VARIANCE_SOURCES false
/* If the flag above is set, then only dump variance information when the
 * sample weight exceeds this threshold. */
#define MTS_DSS_CHECK_VARIANCE_SOURCES_THRESHOLD 1000
/* Support external collimated light sources? This is useful for synthetic
 * 'half-infinite medium, searchlight-type' test scenes, for instance.
 * Set to false when using adaptive integrator, as that is currently broken
 * otherwise... (TODO) */
#define MTS_DSS_USE_RADIANCE_SOURCES true


static StatsCounter avgNumSplits("Direct Sampling Subsurface",
        "Average number of internal-reflection path splits", EAverage);

static StatsCounter avgIntReflChainLen("Direct Sampling Subsurface",
        "Average length of an internal-reflection chain", EAverage);

static ref<Mutex> sourcesMutex = new Mutex();
static int sourcesIndex = 0;

static inline bool vectorEquals(const Vector &a, const Vector &b) {
    Vector sum = a+b;
    Vector diff = a-b;
    // TODO: proper length scale for isZero() cases
    return (diff.length() / sum.length() < Epsilon
            || (sum.isZero() && diff.length() < 1e-10)
            || (a.isZero() && b.length() < 1e-10)
            || (b.isZero() && a.length() < 1e-10));
}

/** Refract the given direction and return the Fresnel transmittance (0 for
 * internal reflection). */
static inline Float refract(Float eta, const Vector &d, const Vector &n,
        Vector &d_refracted, const Frame *nFrame = NULL) {
    Float cosThetaI = dot(d, n);
    Float cosThetaT;
    Float transmittance = 1.0f - fresnelDielectricExt(cosThetaI, cosThetaT, eta);
    if (transmittance == 0.0f)
        return 0.0f;

    Frame frame;
    if (nFrame) {
        frame = *nFrame;
    } else {
        frame = Frame(n);
    }

    Float scale = -1.0f / eta;
    Vector d_loc = frame.toLocal(d);
    d_refracted = frame.toWorld(Vector(scale*d_loc.x, scale*d_loc.y, cosThetaT));
    SAssert(fabs(d_refracted.length() - 1) < Epsilon);
    return transmittance;
}

/// Refracts the directions and returns the Fresnel transmittance
Float DirectSamplingSubsurface::handleImplicitBounds(
        Vector &d_in,  const Normal &n_in,
        Vector &d_out, const Normal &n_out) const {
    if (m_eta == 1.0f)
        return 1.0f;

    Float _cosThetaT, F_in, F_out;
    Vector d_in_refr = refract(-d_in, n_in, m_eta, _cosThetaT, F_in);
    if (d_in_refr.isZero()) {
        if (m_eta > 1)
            Log(EWarn, "Could not refract, which is weird because we have a "
                    "higher ior! (eta=%f)", m_eta);
        return 0.0f;
    }

    Vector d_out_refr = -refract(d_out, n_out, m_eta, _cosThetaT, F_out);
    if (d_out_refr.isZero()) {
        if (m_eta > 1)
            Log(EWarn, "Could not refract, which is weird because we have a "
                    "higher ior! (eta=%f)", m_eta);
        return 0.0f;
    }

    Float fresnelTransmittance = (1-F_in)*(1-F_out);
    d_in = d_in_refr;
    d_out = d_out_refr;

    return fresnelTransmittance;
}

DirectSamplingSubsurface::DirectSamplingSubsurface(const Properties &props) :
            Subsurface(props) {
    /* Num proposals for sample importance resampling. */
    m_numSIR = props.getSize("numSIR", 1);

    /* Perform direct sampling of the light sources? */
    m_directSampling = props.getBoolean("directSampling", true);

    /* Perform MIS weighting of regular sampilg and direct sampling of the
     * light sources? */
    m_directSamplingMIS = props.getBoolean("directSamplingMIS", true);

    /* Perform single spectral channel evaluations of the bssrdf? */
    m_singleChannel = props.getBoolean("singleChannel", false);

    /* Allow the outgoing direction for Li to actually be an incoming one.
     * This is useful for checking the validity of assumed boundary
     * conditions for dipole models, for instance. */
    m_allowIncomingOutgoingDirections = props.getBoolean(
            "allowIncomingOutgoingDirections", false);

    /* Maximum number of subsequent internal reflections. Negative value 
     * for unbounded. */
    m_maxInternalReflections = props.getInteger(
            "maxInternalReflections", -1);

    /* Don't consider incoming surface points that are more absorption
     * lengths away from the outgoing query point than this factor. */
    Float cutoffNumAbsorptionLengths = props.getFloat(
            "cutoffNumAbsorptionLengths", 10);

    if (m_numSIR > 1 && !(m_directSampling && m_directSamplingMIS)) {
        Log(EWarn, "ATTENTION: numSIR is > 1 (it's %d), which means that "
                "direct sampling with MIS weighting gets FORCED, even though "
                "those options were not requested! (requested: direct "
                "sampling %d, MIS %d)",
                m_numSIR, m_directSampling, m_directSamplingMIS);
        m_directSampling = true;
        m_directSamplingMIS = true;
    }

    Log(EInfo, "DirectSamplingSubsurface settings: directSampling %d, "
            "MIS %d, singleChannel %d, allowIncomingOutgoingDirections %d, "
            "maxIntRefl %d",
            m_directSampling, m_directSamplingMIS,
            m_singleChannel, m_allowIncomingOutgoingDirections,
            m_maxInternalReflections);
    {
        LockGuard lock(sourcesMutex);
        m_sourcesIndex = sourcesIndex++;
    }
    m_sourcesResID = -1;
    m_nonCollimatedLightSourcesPresent = true; // Safety

    /* Assume that the medium is parametrized with the typical sigma_{a,s}
     * parameters and use the absorption to get a conservative default
     * intersection distance cutoff. This can be overridden by the derived
     * classes if need be. */
    Spectrum sigmaA, sigmaS, g;
    lookupMaterial(props, sigmaS, sigmaA, g, &m_eta); // also sets m_eta
    // TODO: if m_singleChannel: use the exact sigma_a for the current channel!
    m_itsDistanceCutoff = cutoffNumAbsorptionLengths / sigmaA.min();
}

DirectSamplingSubsurface::DirectSamplingSubsurface(Stream *stream,
        InstanceManager *manager) :
            Subsurface(stream, manager) {
    m_numSIR = stream->readSize();
    m_directSampling = stream->readBool();
    m_directSamplingMIS = stream->readBool();
    m_singleChannel = stream->readBool();
    m_allowIncomingOutgoingDirections = stream->readBool();
    m_eta = stream->readFloat();
    m_sourcesIndex = stream->readInt();
    m_sourcesResID = -1;
    m_itsDistanceCutoff = stream->readFloat();
    /* Note: serialize gets called before preprocess, so we can't pass
     * m_nonCollimatedLightSourcesPresent information here. So for safety: */
    m_nonCollimatedLightSourcesPresent = true;
}

void DirectSamplingSubsurface::serialize(Stream *stream,
        InstanceManager *manager) const {
    Subsurface::serialize(stream, manager);
    stream->writeSize(m_numSIR);
    stream->writeBool(m_directSampling);
    stream->writeBool(m_directSamplingMIS);
    stream->writeBool(m_singleChannel);
    stream->writeBool(m_allowIncomingOutgoingDirections);
    stream->writeFloat(m_eta);
    stream->writeInt(m_sourcesIndex);
    stream->writeFloat(m_itsDistanceCutoff);
    /* Note: serialize gets called before preprocess, so we can't pass
     * m_nonCollimatedLightSourcesPresent information here. */
}

DirectSamplingSubsurface::~DirectSamplingSubsurface() {
    if (m_sourcesResID != -1)
        Scheduler::getInstance()->unregisterResource(m_sourcesResID);
}

void DirectSamplingSubsurface::bindUsedResources(ParallelProcess *proc) const {
    if (m_sourcesResID != -1)
        proc->bindResource(formatString("sources%i", m_sourcesIndex),
                m_sourcesResID);
}

bool DirectSamplingSubsurface::preprocess(
        const Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int cameraResID, int samplerResID) {
    if (!scene->getIntegrator()->getClass()
            ->derivesFrom(MTS_CLASS(MonteCarloIntegrator)))
        Log(EError, "Direct sampling subsurface models require "
            "a MonteCarlo-based surface integrator!");

#if MTS_DSS_USE_RADIANCE_SOURCES
    if (m_sources.get())
        return true;

    /*
     * For degenerate light sources:
     *   - point lights, spot lights and directional lights can (in
     *     principle) be 'directly connected' to any sampled position on
     *     the surface, cfr standard direct sampling methods.
     *   - collimated light sources cannot be connected and must be
     *     'sampled' starting at the light. Trace their (attenuated)
     *     contribution to our surface and store the hitpoints with their
     *     incoming radiance here in m_sources.
     */
    m_sources = new RadianceSources();
    m_nonCollimatedLightSourcesPresent = false;
    for (const ref<Emitter> emitter : scene->getEmitters()) {
        /* Ignore non-collimated light sources */
        if (!(emitter->getType() & Emitter::EDeltaDirection
           && emitter->getType() & Emitter::EDeltaPosition)) {
            m_nonCollimatedLightSourcesPresent = true;
            continue;
        }

        Assert(!emitter->needsPositionSample());
        Assert(!emitter->needsDirectSample());
        PositionSamplingRecord pRec;
        DirectionSamplingRecord dRec;
        std::vector<Intersection> intersections;
        Spectrum Li(1.0f);
        Li *= emitter->samplePosition(pRec, Point2(0.5f));
        Li *= emitter->sampleDirection(dRec, pRec, Point2(0.5f));
        Assert(!Li.isZero());
        scene->collectIntersections(pRec.p, dRec.d, pRec.time, -1,
                intersections, &m_shapes);

        if (intersections.size() == 0)
            continue;

        // first intersection
        Intersection &its = *std::min_element(
                intersections.begin(), intersections.end());
        Vector n = its.shFrame.n;
        if (dot(n, dRec.d) > 0)
            Log(EError, "Found a (collimated) light source *inside* of a "
                    "direct sampling subsurface medium (cos=%f), this is not "
                    "supported (yet) [found %d intersections, first @ t=%f]",
                    dot(n, dRec.d), intersections.size(), its.t);
             /* We can't assign an incoming normal vector in this case,
              * because the light does not enter at the medium boundary and
              * a typical dipole setup is not applicable here */


        /* Handle refraction on a smooth dielectric (TODO: in order to
         * account for rough BSDFs, we need to store the light before
         * interaction with the boundary and store the BSDF and
         * intersection record as well, but that's a bit more tricky to do
         * because we have to serialize those as well, then) */
        const BSDF *bsdf = its.getBSDF();
        if (bsdf->getType() & BSDF::ESmooth)
            Log(EError, "Found a collimated light source hitting a medium "
                    "with a non-delta BSDF. I currently can only store "
                    "collimated sources hitting delta BSDFs (e.g. null or "
                    "smooth dielectrics)");

        /* Sample the adjoint bsdf to get an incoming direction in our
         * medium */
        BSDFSamplingRecord bRec(its, NULL, EImportance);
        bRec.typeMask = BSDF::ETransmission; // We need to get inside
        Spectrum bsdfVal = bsdf->sample(bRec, Point2(0.5f));

        Assert(!(bsdfVal.isZero()));
        Assert(Frame::cosTheta(bRec.wo) < 0); // inwards pointing direction

        m_sources->push_back(RadianceSource(
                its.p, its.shFrame.n, its.toWorld(bRec.wo), Li * bsdfVal));
    }
    Log(EInfo, "DirectSamplingSubsurface medium stored %d collimated light "
            "sources", m_sources->get().size());
    if (!m_nonCollimatedLightSourcesPresent)
        Log(EInfo, "DirectSamplingSubsurface medium found no non-collimated "
                "light sources, so we will not try to sample anything!");
    m_sourcesResID = Scheduler::getInstance()->registerResource(
            m_sources.get());
#endif
    return true;
}

void DirectSamplingSubsurface::wakeup(ConfigurableObject *parent,
        std::map<std::string, SerializableObject *> &params) {
#if MTS_DSS_USE_RADIANCE_SOURCES
    std::string sourcesName = formatString("sources%i", m_sourcesIndex);
    if (!m_sources.get()) {
        if (params.find(sourcesName) != params.end()) {
            m_sources = static_cast<RadianceSources *>(params[sourcesName]);
            Assert(m_sources.get());
            m_active = true;
        } else {
            Log(EWarn, "Woke up but could not find RadianceSources "
                    "resource! (num in params: %ld)", params.size());
        }
    }
#endif
}

MTS_EXPORT_RENDER DistanceWeightFunc makeExactDiffusionDipoleDistanceWeight(
        const Spectrum &sigmaA, const Spectrum &sigmaS,
        const Spectrum &g, Float eta) {
    Spectrum sigmaSPrime = sigmaS * (Spectrum(1.0f) - g);
    Spectrum sigmaTPrime = sigmaSPrime + sigmaA;

    Spectrum mfp = Spectrum(1.0f) / sigmaTPrime;

    /* Average diffuse reflectance due to mismatched indices of refraction
     * in 'to out' (but reflected back inwards to medium)
     *   -> 1/eta = extIOR/intIOR */
    Float Fdr = fresnelDiffuseReflectance(1 / eta);

    /* Dipole boundary condition distance term */
    Float A = (1 + Fdr) / (1 - Fdr);

    /* Effective transport extinction coefficient */
    Spectrum sigmaTr = (sigmaA * sigmaTPrime * 3.0f).sqrt();
    if (sigmaTr.isZero())
        SLog(EError, "makeExactDiffusionDipoleDistanceWeight needs nonzero "
                "product of sigma_s and sigma_a!");

    /* Distance of the two dipole point sources to the surface */
    Spectrum spectral_zr = mfp;
    Spectrum spectral_zv = mfp * (1.0f + 4.0f/3.0f * A);

    return [=] (Float r, int channel) {
        if (sigmaTPrime[channel] == 0) // infinite mfp
            return (Float) 0.0f;

        Float sigma = sigmaTr[channel];
        Float zr = spectral_zr[channel];
        Float zv = spectral_zv[channel];

        Float sr = sqrt(zr*zr + r*r);
        Float sv = sqrt(zv*zv + r*r);

        return zr*(1 + sigma*sr) * exp(-sigma*sr)/(sr*sr*sr)
             + zv*(1 + sigma*sv) * exp(-sigma*sv)/(sv*sv*sv);
    };
}

RadialExactDipoleSampler2D::RadialExactDipoleSampler2D(
        const Spectrum &sigmaA, const Spectrum &sigmaS,
        const Spectrum &g, Float eta) {
    Assert(sigmaA.isFinite());
    Assert(sigmaS.isFinite());
    Assert(g.isFinite());
    Assert(std::isfinite(eta));

    Spectrum sigmaSPrime = sigmaS * (Spectrum(1.0f) - g);
    Spectrum sigmaTPrime = sigmaSPrime + sigmaA;

    Spectrum mfp = sigmaTPrime.invertButKeepZero();

    /* Average diffuse reflectance due to mismatched indices of refraction
     * in 'to out' (but reflected back inwards to medium)
     *   -> 1/eta = extIOR/intIOR */
    Float Fdr = fresnelDiffuseReflectance(1 / eta);

    /* Dipole boundary condition distance term */
    Float A = (1 + Fdr) / (1 - Fdr);

    /* Effective transport extinction coefficient */
    m_sigmaTr = (sigmaA * sigmaTPrime * 3.0f).sqrt();
    if (m_sigmaTr.isZero())
        SLog(EError, "makeExactDiffusionDipoleDistanceWeight needs nonzero "
                "product of sigma_s and sigma_a!");

    /* Distance of the two dipole point sources to the surface */
    m_zr = mfp;
    m_zv = mfp * (1.0f + 4.0f/3.0f * A);

    m_pdfNorm = ( (-m_sigmaTr*m_zr).exp()
                + (-m_sigmaTr*m_zv).exp() ).invertButKeepZero();
    m_T = (-m_sigmaTr*m_zr).exp() * m_pdfNorm;

    Assert(m_pdfNorm.isFinite());
    Assert(m_T.isFinite());
}

bool RadialExactDipoleSampler2D::sample(int channel, Float &r,
        Sampler *sampler, Float *thePdf) const {
    Float sigma = m_sigmaTr[channel];
    if (sigma == 0)
        return false;
    Float zr = m_zr[channel];
    Float zv = m_zv[channel];
    Float T = m_T[channel];

    Float xi = sampler->next1D();
    Float z;
    if (xi <= T) { // typo in paper
        z = zr;
        xi = xi/T;
    } else {
        z = zv;
        xi = (xi - T)/(1 - T);
    }
    Float u = boost::math::tools::newton_raphson_iterate([=](Float u) {
            return std::make_tuple(
//                  sigma*z*(u-1) + log(u) + log(1-xi), /* f(u) */
                    sigma*z*(u-1) + log(u*(1-xi)),      /* f(u) */
                    sigma*z + 1./u); },                 /* f'(u) */
//          (Float) 1, (Float) 0, std::numeric_limits<Float>::infinity(),
            (Float) 1, (Float) 10*RCPOVERFLOW, (Float) 1.0/(10*RCPOVERFLOW),
#ifdef DOUBLE_PRECISION
            40
#else
            20
#endif
            );
    r = z*math::safe_sqrt(u*u - 1);
    if (thePdf) {
        *thePdf = pdf(channel, r);
    }
    return true;
}

Float RadialExactDipoleSampler2D::pdf(int channel, Float r) const {
    Float sigma = m_sigmaTr[channel];
    if (sigma == 0)
        return false;
    Float zr = m_zr[channel];
    Float zv = m_zv[channel];

    Float sr = sqrt(zr*zr + r*r);
    Float sv = sqrt(zv*zv + r*r);

    /* 2*pi*r factor already built-in to take the pdf to the area measure
     * as required */
    return INV_TWOPI * m_pdfNorm[channel]
            * (zr*(1 + sigma*sr) * exp(-sigma*sr)/(sr*sr*sr)
              +zv*(1 + sigma*sv) * exp(-sigma*sv)/(sv*sv*sv));
}

/**
 * \brief Decrease the significant digits in an intersection record.
 *
 * To get somewhat more reproducible intersections when hitting 'the same'
 * point from different directions. This is, for instance, the case when
 * calculating the MIS contribution for a variety of sampling/projection
 * techniques. For highly peaked pdfs, small numerical inaccuracies can
 * cause inconsistent results, which is somewhat alleviated by carrying out
 * this rounding operation on the intersection before computing the pdf.
 * It also helps us 'find/identify' our own intersection again when given a
 * list of intersections by removing some of the numerical noise. */
static inline Intersection roundItsForStability(const Intersection &its) {
#if 1 /* If 1: don't do any rounding -- any possible problems that might
         crop up, can be taken care of with a robust integrator (e.g.
         adaptiveRobustMC) */
    return its;
#else
    Intersection newIts(its);
    newIts.p = roundPointForStability(its.p);
    newIts.shFrame.n = roundDirectionForStability(its.shFrame.n);
    newIts.geoFrame.n = roundDirectionForStability(its.geoFrame.n);
    /* NOTE: the other axes are now no longer orthogonal, but does not seem
     * to be a problem. */
    return newIts;
#endif
}

static inline std::vector<Intersection> roundItssForStability(
        const std::vector<Intersection> &itss) {
    std::vector<Intersection> newItss(itss.size());
    for (size_t i = 0; i < itss.size(); i++)
        newItss[i] = roundItsForStability(itss[i]);
    return newItss;
}

Float WeightIntersectionSampler::sample(
        const std::vector<Intersection> &intersections, Intersection &newIts,
        const Intersection &its_out, const Vector &d_out,
        int channel, Sampler *sampler) const {
    SAssert(intersections.size() != 0);
    Float intersectionProb;
    Vector n_geo(0.0f);
    Float weights[intersections.size()];
    Float cumulWeight = 0;
    /* All decisions happen based on the safeItss (but newIts gets set to
     * the corresponding original one, of course) */
    std::vector<Intersection> safeItss = roundItssForStability(intersections);
    for (size_t i = 0; i < safeItss.size(); i++) {
        weights[i] = channelMean(channel,
                    [=] (int chan) { return m_intersectionWeight(
                        safeItss[i], its_out, d_out, chan); });
        Assert(weights[i] >= 0);
        cumulWeight += weights[i];
    }
    SAssert(cumulWeight >= 0);
    if (cumulWeight <= RCPOVERFLOW) {
        /* Can potentially happen if all weights fall below the smallest
         * properly representable number (e.g. when intersections are far
         * away for exponential weights). We could try to just pick an
         * intersection uniformally and use that in order to avoid bias as
         * much as possible (though the same problem occurs if only some
         * weights are zero, which is not handled here), but the
         * contribution will probably end up being zero anyway, so just
         * bail out. */
        return 0.0f;
    }
    intersectionProb = -1;
    size_t idx = histogramSample(weights, intersections.size(),
            sampler->next1D(), &intersectionProb);
    newIts = intersections[idx];
    n_geo = newIts.geoFrame.n;
    SAssert(std::isfinite(intersectionProb) && intersectionProb > 0);
    SAssert(!n_geo.isZero());
#ifdef MTS_DSS_PDF_CHECK
    Float pdf2 = pdf(intersections, newIts, its_out, d_out, channel);
    if (fabs((intersectionProb - pdf2)/(intersectionProb + pdf2)) > ShadowEpsilon)
        SLog(EWarn, "Inconsistent pdfs: %e vs %e, rel %e",
                intersectionProb, pdf2, intersectionProb/pdf2);
#endif
    return intersectionProb;
}

Float WeightIntersectionSampler::pdf(
        const std::vector<Intersection> &intersections,
        const Intersection &newIts, const Intersection &its_out,
        const Vector &d_out, int channel) const {
    SAssert(intersections.size() != 0);

    std::vector<Intersection> safeItss = roundItssForStability(intersections);
    Intersection newSafeIts = roundItsForStability(newIts);

    /* Find our weight */
    Float ourWeight = channelMean(channel,
                    [=] (int chan) { return m_intersectionWeight(
                        newSafeIts, its_out, d_out, chan); });
    Assert(ourWeight >= 0);
    /* Find cumul weight of all intersections (and check if we find ours!) */
    size_t ourIdx = (size_t) -1;
    Float cumulWeight = 0;
    for (size_t i = 0; i < intersections.size(); i++) {
        Float thisWeight = channelMean(channel,
                    [=] (int chan) { return m_intersectionWeight(
                        safeItss[i], its_out, d_out, chan); });
        Assert(thisWeight >= 0);
        cumulWeight += thisWeight;
        if (vectorEquals(Vector(newSafeIts.p), Vector(safeItss[i].p)))
            ourIdx = i;
    }

    if (cumulWeight <= RCPOVERFLOW)
        return 0.0f;

    /* Can still fail in very rare cases. For doubly-infinite rays, the
     * number of found intersections is sometimes not an even number
     * either. But even for semi-infinite rays, the actually sampled
     * intersection should be in the list when this function is called
     * (otherwise we would have bailed out more early!) */
    if (ourIdx == (size_t) -1) {
#ifdef MTS_DEBUG
        SLog(EWarn, "Could not seem to find our own intersection. "
                "(Num candidates: %d, cumul weight %e, newIts.p %s, "
                "newIts.n %s, shape %s)",
                intersections.size(), cumulWeight,
                newIts.p.toString().c_str(),
                newIts.geoFrame.n.toString().c_str(),
                newIts.shape->getName().c_str());
#endif
        return 0.0f;
    }

    if (!(ourWeight <= cumulWeight * (1 + sqrt(Epsilon)))) {
#ifdef MTS_DEBUG
        Float sumOfOtherWeights = 0;
        for (size_t i = 0; i < intersections.size(); i++) {
            if (i == ourIdx)
                continue;
            Float thisWeight = channelMean(channel,
                        [=] (int chan) { return m_intersectionWeight(
                            safeItss[i], its_out, d_out, chan); });
            sumOfOtherWeights += thisWeight;
        }
        SLog(EWarn, "Inconsistent weights for pdfIntersection: "
                "ours %e, cumul %e, rel %e, num its %d, sumOfOthers %e, relSumOthers %e",
                ourWeight, cumulWeight, ourWeight/cumulWeight,
                intersections.size(), sumOfOtherWeights, sumOfOtherWeights/cumulWeight);
#endif
        return 1.0f;
    }

    return ourWeight / cumulWeight;
}




Float UniformSurfaceSampler::sample(const Intersection &its,
        const Vector &d_out, const Scene *scene,
        const std::vector<Shape *> &shapes,
        Intersection &newIts, int channel,
        Sampler *sampler) const {
    size_t N = shapes.size();
    Float weights[N];
    Float SA = 0;
    for (size_t i = 0; i < N; i++) {
        weights[i] = shapes[i]->getSurfaceArea();
        SA += weights[i];
    }
    const Shape *shape = shapes[
            histogramSample(weights, N, sampler->next1D())];

    PositionSamplingRecord pRec(its.time);
    shape->samplePosition(pRec, sampler->next2D());
    newIts.p = pRec.p;

    // Set up a fake interesection record
    newIts.geoFrame = Frame(pRec.n);
    newIts.shFrame = newIts.geoFrame;
    newIts.hasUVPartials = false;
    newIts.shape = shape;
    newIts.time = its.time;

    return 1.0f / SA;
}

Float UniformSurfaceSampler::pdf(const Intersection &its,
        const Vector &d_out, const Scene *scene,
        const std::vector<Shape *> &shapes,
        const Intersection &newIts, int channel) const {
    Float SA = 0;
    for (auto shape : shapes) {
        SA += shape->getSurfaceArea();
    }
    return 1.0f / SA;
}


Float BRDFDeltaSurfaceSampler::sample(const Intersection &its,
        const Vector &d_out, const Scene *scene,
        const std::vector<Shape *> &shapes,
        Intersection &newIts, int channel,
        Sampler *sampler) const {
    newIts = its;
    return 1.0;
}

Float BRDFDeltaSurfaceSampler::pdf(const Intersection &its,
        const Vector &d_out, const Scene *scene,
        const std::vector<Shape *> &shapes,
        const Intersection &newIts, int channel) const {
    if (distance(newIts.p, its.p) <= Epsilon*Vector(its.p).length())
        return 1.0f;
    return 0.0f;
}


void ProjSurfaceSampler::getProjFrame(
        Vector &u, Vector &v, Vector &projectionDir,
        const Intersection &its, const Vector &d_out) const {
    /* Geoframe might seem like a more robust version to actually hit
     * geometry, but it would give a slightly worse importance sampling (in
     * principle, at least). Moreover, for shFrame, cosTheta >= 0 is
     * guaranteed, but not for the geoFrame (well, it usually is through a
     * 'light leaks check', though) */
    const Frame &frame = its.shFrame;
    Vector n_out = frame.n;
    m_projFrame.getProjFrame(u, v, projectionDir,
            n_out, d_out);
}

static void getExtremalPlaneValues(const Vector &u, const Vector &v,
        const std::vector<Shape *> &shapes, const Point &p,
        Vector2 &xLo, Vector2 &xHi) {
    AABB aabb;
    for (const Shape *shape : shapes) {
        aabb.expandBy(shape->getAABB());
    }
    xLo = Vector2( std::numeric_limits<Float>::infinity());
    xHi = Vector2(-std::numeric_limits<Float>::infinity());
    for (int i = 0; i < 8; i++) {
        Vector corner = aabb.getCorner(i) - p;
        xLo.x = std::min(xLo.x, dot(u, corner));
        xLo.y = std::min(xLo.y, dot(v, corner));
        xHi.x = std::max(xHi.x, dot(u, corner));
        xHi.y = std::max(xHi.y, dot(v, corner));
    }
    /* Because we only get called when the point p is actually on the
     * surface, we should have only negative xLo values and only positive
     * xHi values. */
    SAssertWarn(xLo.x <= 0 && xLo.y <= 0);
    SAssertWarn(xHi.x >= 0 && xHi.y >= 0);

    SAssertWarn(xLo.isFinite());
    SAssertWarn(xHi.isFinite());
}

Float ProjSurfaceSampler::sample(const Intersection &its,
        const Vector &d_out, const Scene *scene,
        const std::vector<Shape *> &shapes,
        Intersection &newIts, int channel,
        Sampler *sampler) const {
    /* Sample from the uniformly weighted sum of the pdfs associated with
     * the different spectral channels. First pick a single channel
     * uniformly. */
    Float chosenChannel = channel == -1
            ? sampler->next1D() * SPECTRUM_SAMPLES
            : channel;

    Vector n = its.shFrame.n; // NOTE: should be the same in getProjFrame()
    Float cosTheta = dot(n, d_out);

    Vector u, v, projectionDir;
    getProjFrame(u, v, projectionDir, its, d_out);

    Vector2 xLo, xHi;
    getExtremalPlaneValues(u, v, shapes, its.p, xLo, xHi);

    Vector2 x;
    if (!m_planeSampler->sample(chosenChannel, x, cosTheta, xLo, xHi, sampler))
        return 0.0f;
    Float planePdf = channelMean(channel,
                    [=] (int chan) { return m_planeSampler->pdf(
                                       chan, x, cosTheta, xLo, xHi); });

    if (!std::isfinite(planePdf) || planePdf <= 0) {
        /* Note: planePdf shouldn't be 0 because we already could sample 
         * our 'own' channel! */
        Log(EWarn, "Invalid plane pdf: %e", planePdf);
        return 0.0f;
    }

    Point o = its.p + x[0]*u + x[1]*v;

    /* Project our point back to the actual surface mesh. Keep
     * intersecting and record all intersections 'with ourself' (as per
     * shapes) and probabilistically choose which intersection to use
     * based on an importance weight.
     *
     * Note that this fails for right angles like so:
     *
     *                query point
     *                |
     *                v
     * ________________________________  <- projection plane
     *         .---------------.
     *         |               | <- geometry
     *         |               |
     *
     *         ^---------------^--- the side planes here cannot
     *                              be sampled
     *
     * This should not be a problem for 'natural/organic' scenes, but
     * for man-made/architectural scenes, this can be problematic, so it is
     * advised to combine samplers multiple projection directions.
     *
     * Alternatively, we can just always take the closest intersection
     * point. This is somewhat problematic for concave geometry, though
     * ('concave on the length scales that are relevant for subsurface
     * trasport') E.g.:
     *
     *                query point
     *                |
     *                v
     * ________________________________  <- projection plane
     *           .-------------.
     *          /               \
     *         |                 \ <- geometry
     *         |
     *          \____  . . . .
     *               \       . surface here
     *                |      . cannot be sampled
     *    ___________/ . . . .
     *   /    .       .
     *  |     .       .
     *        . . . . .
     *         surfaces
     *       above cannot
     *        be sampled
     */
    Float intersectionProb = m_itsSampler->sample(newIts, scene,
            o, projectionDir, its.time,
            shapes, its, d_out, channel, sampler);
    if (intersectionProb == 0)
        return 0.0f;

    Vector n_geo = newIts.geoFrame.n;
    if (math::abs(dot(n_geo, projectionDir)) <= MTS_DSS_COSINE_CUTOFF) {
        /* Sampling this surface was badly conditioned -- abandon ship here
         * and let a sampler with a different projection direction save us
         * from bias. */
        return 0.0f;
    }

    /* pdf in area measure of the surface of the object */
    Float thePdf = planePdf * intersectionProb
            * math::abs(dot(n_geo, projectionDir));
#ifdef MTS_DSS_PDF_CHECK
    Float pdf2 = pdf(its, d_out, scene, shapes, newIts, channel);
    if (fabs((thePdf - pdf2)/thePdf) > ShadowEpsilon) {
        Log(EWarn, "PDF problem! %e vs %e, rel: %e",
                thePdf, pdf2, thePdf/pdf2);
    }
#endif
    return thePdf;
}

Float ProjSurfaceSampler::pdf(const Intersection &its,
        const Vector &d_out, const Scene *scene,
        const std::vector<Shape *> &shapes,
        const Intersection &newIts, int channel) const {
    Vector u, v, projectionDir;
    getProjFrame(u, v, projectionDir, its, d_out);

    Vector n_geo = newIts.geoFrame.n;
    if (math::abs(dot(n_geo, projectionDir)) <= MTS_DSS_COSINE_CUTOFF) {
        return 0.0f;
    }

    Vector n = its.shFrame.n;
    Float cosTheta = dot(n, d_out);

    Vector2 xLo, xHi;
    getExtremalPlaneValues(u, v, shapes, its.p, xLo, xHi);

    Vector delta = newIts.p - its.p;
    Vector2 x(dot(delta,u), dot(delta,v));

    /* Pdf of the point in the tangent plane in its area measure, taking
     * into account that we could have sampled from any channel */
    Float planePdf = channelMean(channel,
                    [=] (int chan) { return m_planeSampler->pdf(
                                       chan, x, cosTheta, xLo, xHi); });
    if (planePdf == 0)
        return 0.0f;
    if (!std::isfinite(planePdf) || planePdf < 0) {
        SLog(EWarn, "Something fishy happened");
        return 0.0f;
    }
    std::vector<Intersection> intersections;
    /* We should be able to choose the origin of our intersection
     * collection at the surface instead of where we would have started
     * from during the sampling step (point 'o'): */
    Float intersectionProb = m_itsSampler->pdf(newIts, scene,
            newIts.p, projectionDir, its.time,
            shapes, its, d_out, channel);
    if (intersectionProb == 0)
        return 0.0f;

    /* pdf in area measure of the surface of the object */
    return planePdf * intersectionProb * math::abs(dot(n_geo, projectionDir));
}



size_t DirectSamplingSubsurface::extraParamsSize() const {
    return 0;
}


Spectrum DirectSamplingSubsurface::sampleBssrdfDirection(const Scene *scene,
        const Intersection &its_out, const Vector &d_out,
        Intersection       &its_in,  Vector       &d_in,
        const void *extraParams, const Spectrum &throughput,
        Sampler *sampler) const {
    /* Sample an incoming direction (on our side of the medium) on the
     * cosine-weighted hemisphere */
    Vector hemiSamp = warp::squareToCosineHemisphere(sampler->next2D());
    Float pdf = warp::squareToCosineHemispherePdf(hemiSamp);
    hemiSamp.z = -hemiSamp.z; // pointing inwards
    its_in.wi = hemiSamp;
    d_in = its_in.toWorld(hemiSamp); // pointing inwards

#ifdef MTS_DSS_PDF_CHECK
    Float pdf2 = pdfBssrdfDirection(scene, its_out, d_out, its_in,
            d_in, extraParams, throughput).average();
    if (fabs((pdf - pdf2)/(pdf + pdf2)) > ShadowEpsilon)
        SLog(EWarn, "Inconsistent pdfs: %e vs %e, rel %e",
                pdf, pdf2, pdf/pdf2);
#endif

#if !MTS_DSS_ALLOW_INTERNAL_INCOMING_DIR
    if (dot(d_in, its_in.shFrame.n) > 0)
        return Spectrum(0.0); // protect against roundoff errors
#endif

    return Spectrum(pdf);
}

Spectrum DirectSamplingSubsurface::pdfBssrdfDirection(const Scene *scene,
        const Intersection &its_out, const Vector &d_out,
        const Intersection &its_in,  const Vector &d_in,
        const void *extraParams, const Spectrum &throughput) const {
    Vector n_in = its_in.shFrame.n;
#if MTS_DSS_ALLOW_INTERNAL_INCOMING_DIR
    return Spectrum(INV_PI * math::abs(dot(d_in, n_in)));
#else
    Assert(-dot(d_in, n_in) >= 0);
    return Spectrum(INV_PI * (-dot(d_in, n_in)));
#endif
}



Spectrum DirectSamplingSubsurface::sampleDirectionsFromBssrdf(
        const Scene *scene,
        const Intersection &its_out, const Vector &d_out,
        Intersection       &its_in,  Vector       &d_in,
        Vector &rec_wi,
        EMeasure &bsdfMeasure,
        Spectrum &pdf_d_in_and_rec_wi,
        const Spectrum &throughput,
        const void *extraParams,
        Sampler *sampler) const {

    /* Let the bssrdf sample the direction */
    Spectrum d_in_pdf = sampleBssrdfDirection(
            scene, its_out, d_out, its_in, d_in, extraParams,
            throughput, sampler);
    if (d_in_pdf.isZero())
        return Spectrum(0.0f);
    Assert(MTS_DSS_ALLOW_INTERNAL_INCOMING_DIR
            || dot(d_in, its_in.shFrame.n) <= 0);

    /* Sample BSDF * cos(theta) */
    // TODO: version with ray argument needed when texturing?
    const BSDF *bsdf = its_in.getBSDF();
    BSDFSamplingRecord bRec(its_in, sampler, ERadiance);
    Float rec_wi_pdf;
    Spectrum bsdfVal = bsdf->sample(bRec, rec_wi_pdf, sampler->next2D());
    if (bsdfVal.isZero())
        return Spectrum(0.0f);
    bsdfVal *= rec_wi_pdf; // Transform back to actual bsdf value
    /* The bsdf already takes care of one cosine factor, get the other
     * cosine in as well */
#if MTS_DSS_ALLOW_INTERNAL_INCOMING_DIR
    bsdfVal *= math::abs(dot(d_in, its_in.shFrame.n));
#else
    bsdfVal *= -dot(d_in, its_in.shFrame.n);
#endif
    /* We need to store the measure to get the correct pdfs down the road
     * (e.g. discrete versus solid angle) */
    bsdfMeasure = bsdf->getMeasure(bRec.sampledType);
    rec_wi = its_in.toWorld(bRec.wo);

    pdf_d_in_and_rec_wi = d_in_pdf * rec_wi_pdf;
#ifdef MTS_DSS_PDF_CHECK
    Spectrum pdf2 = pdfDirectionsFromBssrdf(scene, its_out, d_out, its_in,
            d_in, rec_wi, bsdfMeasure, throughput, extraParams);
    Float relErr = fabs(
              (pdf_d_in_and_rec_wi.average() - pdf2.average())
            / (pdf_d_in_and_rec_wi.average() + pdf2.average()));
    if (relErr > ShadowEpsilon)
        SLog(EWarn, "Inconsistent pdfs: %e vs %e, rel %e",
                pdf_d_in_and_rec_wi.average(), pdf2.average(),
                relErr);
#endif
    return bsdfVal;
}

Spectrum DirectSamplingSubsurface::pdfDirectionsFromBssrdf(
        const Scene *scene,
        const Intersection &its_out, const Vector &d_out,
        const Intersection &its_in,  const Vector &d_in,
        const Vector &rec_wi,
        EMeasure bsdfMeasure,
        const Spectrum &throughput,
        const void *extraParams) const {
    Assert(bsdfMeasure != EInvalidMeasure);
    Spectrum d_in_pdf = pdfBssrdfDirection(
            scene, its_out, d_out, its_in, d_in, extraParams,
            throughput);
    if (d_in_pdf.isZero())
        return Spectrum(0.0f);

    BSDFSamplingRecord bRec(its_in, its_in.toLocal(d_in),
            its_in.toLocal(rec_wi), ERadiance);
    const BSDF *bsdf = its_in.getBSDF();
    Float rec_wi_pdf = bsdf->pdf(bRec, bsdfMeasure);

    return d_in_pdf * rec_wi_pdf;
}

Spectrum DirectSamplingSubsurface::sampleDirectionsDirect(
        const Scene *scene,
        const Intersection &its_out, const Vector &d_out,
        Intersection       &its_in,  Vector       &d_in,
        Vector &rec_wi,
        EMeasure &bsdfMeasure,
        EMeasure &lightSamplingMeasure,
        Spectrum &pdf_d_in_and_rec_wi,
        Spectrum &LiDirect,
        Sampler *sampler) const {
    Vector n_in = its_in.shFrame.n;

    /* Direct sampling of a light source */
    DirectSamplingRecord dRec(its_in);
    Spectrum LiDirectWeighted = scene->sampleEmitterDirect(
            dRec, sampler->next2D());
    Float lightCosineFactor = dot(dRec.d, n_in);
    if (LiDirectWeighted.isZero() || lightCosineFactor <= 0) {
        /* light is outside of our medium */
        return Spectrum(0.0);
    }
    lightSamplingMeasure = dRec.measure;

    Float rec_wi_pdf = dRec.pdf;
    /* Transform the sample weight back to the actual Li value */
    LiDirect = LiDirectWeighted * rec_wi_pdf;

    /* Sample the adjoint bsdf to get an incoming direction in our
     * medium */
    rec_wi = dRec.d;
    /* Fill its_in as if we would have sampled in the regular ('ERadiance')
     * way, use a reversed intersection record so we can set up the
     * BSDFSamplingRecord (get its 'wi' to be our 'rec_wi', which is the
     * 'wi' of the reversed intersection record). Yes, nasty. */
    Intersection reverse_its_in(its_in);
    reverse_its_in.wi = its_in.toLocal(rec_wi);
    BSDFSamplingRecord bRec(reverse_its_in, sampler, EImportance);
    bRec.typeMask = BSDF::ETransmission; // We only care about transmitted rays
    const BSDF *bsdf = its_in.getBSDF();
    Float d_in_pdf;

    Spectrum bsdfVal = bsdf->sample(bRec, d_in_pdf, sampler->next2D());
    if (bsdfVal.isZero())
        return Spectrum(0.0);
    Assert(Frame::cosTheta(bRec.wo) < 0); // inwards pointing direction
    bsdfVal *= d_in_pdf; // Transform back to actual bsdf value

    /* The included cosine factor in the bsdfVal is on our 'd_in', get the
     * other cosine factor in as well */
    Assert(dot(rec_wi, n_in) >= 0);
    bsdfVal *= dot(rec_wi, n_in);

    bsdfMeasure = bsdf->getMeasure(bRec.sampledType);
    its_in.wi = bRec.wo; // Make it as if we sampled in the radiance direction
    d_in = its_in.toWorld(bRec.wo);
    pdf_d_in_and_rec_wi = Spectrum(d_in_pdf * rec_wi_pdf);

#ifdef MTS_DSS_PDF_CHECK
    Spectrum pdf2 = pdfDirectionsDirect(scene, its_out, d_out, its_in,
            d_in, rec_wi, bsdfMeasure);
    Float relErr = fabs(
              (pdf_d_in_and_rec_wi.average() - pdf2.average())
            / (pdf_d_in_and_rec_wi.average() + pdf2.average());
    if (relErr > ShadowEpsilon)
        SLog(EWarn, "Inconsistent pdfs: %e vs %e, rel %e",
                pdf_d_in_and_rec_wi.average(), pdf2.average(),
                relErr);
#endif

    return bsdfVal;
}

Spectrum DirectSamplingSubsurface::pdfDirectionsDirect(
        const Scene *scene,
        const Intersection &its_out, const Vector &d_out,
        const Intersection &its_in,  const Vector &d_in,
        const Vector &rec_wi,
        EMeasure bsdfMeasure) const {
    Assert(bsdfMeasure != EInvalidMeasure);

    const BSDF *bsdf = its_in.getBSDF();
    Vector n_in = its_in.shFrame.n;

    DirectSamplingRecord dRec(its_in);
    dRec.d = rec_wi;
    Float lightCosineFactor = dot(dRec.d, n_in);
    if (lightCosineFactor <= 0 /* light is outside of our medium */)
        return Spectrum(0.0);

    /* We need to find the actual intersection first by shooting a ray
     * (based on rayIntersectAndLookForEmitter() in volpath.cpp) */
    Intersection emitterIts;
    Ray ray(its_in.p, rec_wi, its_in.time);
    bool surface = scene->rayIntersect(ray, emitterIts);
    if (surface) {
        /* Intersected something - check if it was a luminaire */
        if (emitterIts.isEmitter()) {
            dRec.setQuery(ray, emitterIts);
        } else {
            return Spectrum(0.0f);
        }
    } else {
        /* Intersected nothing -- perhaps there is an environment map? */
        const Emitter *env = scene->getEnvironmentEmitter();

        if (!env || !env->fillDirectSamplingRecord(dRec, ray))
            return Spectrum(0.0f);
    }
    Float rec_wi_pdf = scene->pdfEmitterDirect(dRec);

    if (rec_wi_pdf == 0)
        return Spectrum(0.0);

    BSDFSamplingRecord bRec(its_in, its_in.toLocal(rec_wi),
            its_in.toLocal(d_in), EImportance);
    bRec.typeMask = BSDF::ETransmission;
    Float d_in_pdf = bsdf->pdf(bRec, bsdfMeasure);

    return Spectrum(d_in_pdf * rec_wi_pdf);
}

Spectrum DirectSamplingSubsurface::sampleDirect(
        const Scene *scene, Sampler *sampler,
        Intersection &its_in, const Intersection &its_out,
        const Vector &d_out,
        const Spectrum &effectiveThroughput,
        Vector &d_in, Vector &rec_wi, void *extraParams,
        EMeasure &bsdfMeasure, EMeasure &lightSamplingMeasure,
        Spectrum &bsdfVal, Spectrum &LiDirect) const {
    Spectrum pdf_d_in_and_rec_wi;

    /* Sample incoming directions d_in&rec_wi based on its_in */
    bsdfVal = sampleDirectionsDirect(
            scene, its_out, d_out, its_in, d_in, rec_wi, bsdfMeasure,
            lightSamplingMeasure, pdf_d_in_and_rec_wi, LiDirect, sampler);
    if (bsdfVal.isZero())
        return Spectrum(0.0f);

    /* Sample extraParams based on its_in and d_in&rec_wi */
    Spectrum extraParamsPdf = sampleExtraParams(
            scene, its_out, d_out, its_in, &d_in, effectiveThroughput,
            extraParams, sampler);
    if (extraParamsPdf.isZero())
        return Spectrum(0.0f);

    Vector n_in = its_in.shFrame.n;
    Float rec_wi_cosine = dot(rec_wi, n_in);
    Assert(rec_wi_cosine >= 0);
    Float d_in_cosine = dot(d_in, n_in);
    Assert(MTS_DSS_ALLOW_INTERNAL_INCOMING_DIR || d_in_cosine <= 0);

    Spectrum thePdf = pdf_d_in_and_rec_wi * extraParamsPdf;

#ifdef MTS_DSS_PDF_CHECK
    Float pdf2 = pdfDirect(scene, its_in, its_out, d_out,
            effectiveThroughput, d_in, rec_wi, extraParams,
            bsdfMeasure).average();
    Float relErr = fabs( (thePdf.average() - pdf2)
                       / (thePdf.average() + pdf2) );
    if (relErr > ShadowEpsilon)
        SLog(EWarn, "Inconsistent pdfs: %e vs %e, rel %e",
                thePdf.average(), pdf2, relErr);
#endif

    return thePdf;
}

Spectrum DirectSamplingSubsurface::pdfDirect(const Scene *scene,
        const Intersection &its_in, const Intersection &its_out,
        const Vector &d_out,
        const Spectrum &effectiveThroughput,
        const Vector &d_in, const Vector &rec_wi, const void *extraParams,
        EMeasure bsdfMeasure) const {
    Assert(bsdfMeasure != EInvalidMeasure);
    Spectrum pdf_d_in_and_rec_wi = pdfDirectionsDirect(
            scene, its_out, d_out, its_in, d_in, rec_wi, bsdfMeasure);
    if (pdf_d_in_and_rec_wi.isZero())
        return Spectrum(0.0f);

    Spectrum extraParamsPdf = pdfExtraParams(
            scene, its_out, d_out, its_in, &d_in, effectiveThroughput,
            extraParams);

    return pdf_d_in_and_rec_wi * extraParamsPdf;
}


Spectrum DirectSamplingSubsurface::sampleIndirect(
        const Scene *scene, Sampler *sampler,
        Intersection &its_in, const Intersection &its_out,
        const Vector &d_out,
        const Spectrum &effectiveThroughput,
        Vector &d_in, Vector &rec_wi, void *extraParams,
        EMeasure &bsdfMeasure, Spectrum &bsdfVal) const {

    /* Sample extraParams based on its_in */
    Spectrum extraParamsPdf = sampleExtraParams(
            scene, its_out, d_out, its_in, NULL, effectiveThroughput,
            extraParams, sampler);
    if (extraParamsPdf.isZero())
        return Spectrum(0.0f);

    Spectrum pdf_d_in_and_rec_wi;

    /* Sample incoming directions d_in&rec_wi based on its_in and
     * extraParams */
    bsdfVal = sampleDirectionsFromBssrdf(
            scene, its_out, d_out, its_in, d_in, rec_wi, bsdfMeasure,
            pdf_d_in_and_rec_wi,
            effectiveThroughput * extraParamsPdf.zeroMask(),
            extraParams, sampler);
    if (bsdfVal.isZero())
        return Spectrum(0.0f);

    Vector n_in = its_in.shFrame.n;
    Float d_in_cosine = dot(d_in, n_in);
    Assert(MTS_DSS_ALLOW_INTERNAL_INCOMING_DIR || d_in_cosine <= 0);

    Spectrum thePdf = pdf_d_in_and_rec_wi * extraParamsPdf;

#ifdef MTS_DSS_PDF_CHECK
    Float pdf2 = pdfIndirect(scene, its_in, its_out, d_out,
            effectiveThroughput, d_in, rec_wi, extraParams,
            bsdfMeasure).average();
    Float relErr = fabs( (thePdf.average() - pdf2)
                       / (thePdf.average() + pdf2) );
    if (relErr > ShadowEpsilon)
        SLog(EWarn, "Inconsistent pdfs: %e vs %e, rel %e",
                thePdf.average(), pdf2, relErr);
#endif

    return thePdf;
}

Spectrum DirectSamplingSubsurface::pdfIndirect(const Scene *scene,
        const Intersection &its_in, const Intersection &its_out,
        const Vector &d_out,
        const Spectrum &effectiveThroughput,
        const Vector &d_in, const Vector &rec_wi, const void *extraParams,
        EMeasure bsdfMeasure) const {
    Assert(bsdfMeasure != EInvalidMeasure);
    Float d_in_cosine = dot(d_in, its_in.shFrame.n);
    Assert(MTS_DSS_ALLOW_INTERNAL_INCOMING_DIR || d_in_cosine <= 0);

    Spectrum extraParamsPdf = pdfExtraParams(
            scene, its_out, d_out, its_in, NULL, effectiveThroughput,
            extraParams);
    if (extraParamsPdf.isZero())
        return Spectrum(0.0f);

    Spectrum pdf_d_in_and_rec_wi = pdfDirectionsFromBssrdf(
            scene, its_out, d_out, its_in, d_in, rec_wi, bsdfMeasure,
            effectiveThroughput * extraParamsPdf.zeroMask(),
            extraParams);

    return pdf_d_in_and_rec_wi * extraParamsPdf;
}





void DirectSamplingSubsurface::checkSourcesOfVariance(
        const Scene *scene, const Spectrum &throughput,
        const Intersection its_out,
        const Vector d_out,
        const Intersection check_its_in,
        const Vector check_d_in,
        const Vector check_rec_wi,
        const void *check_extraParams,
        const Spectrum check_bssrdfVal,
        const Spectrum check_bsdfVal,
        const EMeasure check_bsdfMeasure,
        Sampler *sampler,
        const bool absify) const {

    /* BEWARE: This is rather nasty code, for debugging purposes... */

    const Point  p_out = its_out.p;
    const Normal n_out = its_out.shFrame.n;

    char extraParamsBuffer[extraParamsSize()];
    void *extraParams = extraParamsBuffer;

    const int numIntSamples = 100000; // 2000;
    const int continueWithZeroFactor = 10;

    // Compute the original PDFS
    Float check_surfacePdf = pdfPointOnSurface(
            its_out, d_out, scene, check_its_in, throughput);
    Spectrum check_extraParamsPdf = pdfExtraParams(
            scene, its_out, d_out, check_its_in, NULL, throughput,
            check_extraParams);
    Spectrum check_pdf_d_in_and_rec_wi = pdfDirectionsFromBssrdf(
            scene, its_out, d_out, check_its_in,
            check_d_in, check_rec_wi, check_bsdfMeasure, throughput,
            check_extraParams);
    if (check_surfacePdf == 0)
        cerr << "check_surfacePdf was zero!" << endl;
    if (check_extraParamsPdf.isZero())
        cerr << "check_extraParamsPdf was zero!" << endl;
    if (check_pdf_d_in_and_rec_wi.isZero())
        cerr << "check_pdf_d_in_and_rec_wi was zero!" << endl;


    /* GET EXACT INTEGRAL */
    VarianceFunctor integral;
    Intersection its_in;
    Vector d_in, rec_wi;
    EMeasure bsdfMeasure;
    int integralSuccess = 0;
    int intSurfaceSuccess = 0;
    for (int i = 0;
            i < numIntSamples
            || (integralSuccess == 0
                && i < continueWithZeroFactor*numIntSamples);
            i++) {
        /* Sample BSSRDF marginalized over incoming direction and
         * extraParams */
        Float surfacePdf = samplePointOnSurface(
                its_out, d_out, scene, its_in, throughput, sampler);
        Point  p_in = its_in.p;
        Vector n_in = its_in.shFrame.n;
        if (surfacePdf == 0) {
            integral.update(0);
            continue;
        }
        intSurfaceSuccess++;

        /* Sample extraParams based on its_in */
        Spectrum extraParamsPdf = sampleExtraParams(
                scene, its_out, d_out, its_in, NULL, throughput,
                extraParams, sampler);
        if (extraParamsPdf.isZero()) {
            integral.update(0);
            continue;
        }

        /* Sample incoming directions d_in&rec_wi based on its_in and
         * extraParams */
        Spectrum pdf_d_in_and_rec_wi;
        Spectrum bsdfVal = sampleDirectionsFromBssrdf(
                scene, its_out, d_out, its_in, d_in, rec_wi, bsdfMeasure,
                pdf_d_in_and_rec_wi, throughput, extraParams, sampler);
        if (bsdfVal.isZero()) {
            integral.update(0);
            continue;
        }
        Assert(bsdfMeasure == check_bsdfMeasure);

        /* Evaluate BSSRDF */
        Spectrum bssrdfVal = bssrdf(scene, p_in, d_in, n_in, p_out,
                d_out, n_out, extraParams);
        if (bssrdfVal.isZero()) {
            integral.update(0);
            continue;
        }

        Float cosTheta = math::abs(dot(d_in, n_in));
        if (cosTheta == 0) {
            integral.update(0);
            continue;
        }
        Spectrum fullPdf = pdf_d_in_and_rec_wi*extraParamsPdf*surfacePdf;
        Float contribution = (bssrdfVal*bsdfVal/fullPdf).averageNonNan();
        if (absify)
            contribution = math::abs(contribution);
        if (!std::isfinite(contribution)) {
            integral.update(0);
            continue;
        }
        integral.update(contribution);
        integralSuccess++;
    }
    Float theIntegral = integral.mean();



    /* CHECK EACH SUCCESSIVE CONDITIONAL FACTOR IN THE COMBINED PDF */


    // INTEGRATE EXTRA PARAMS AND DIRECTION

    its_in = check_its_in;
    Float surfacePdf = 1; // we don't integrate over this here
    Point  p_in = its_in.p;
    Vector n_in = its_in.shFrame.n;

    VarianceFunctor intExtraParamsAndDir;
    int intExtraParamsAndDirSuccess = 0;
    int intExtraParamsSuccess = 0;
    for (int i = 0;
            i < numIntSamples
            || (intExtraParamsAndDirSuccess == 0
                && i < continueWithZeroFactor*numIntSamples);
            i++) {
        /* Sample extraParams based on its_in */
        Spectrum extraParamsPdf = sampleExtraParams(
                scene, its_out, d_out, its_in, NULL, throughput,
                extraParams, sampler);
        if (extraParamsPdf.isZero()) {
            intExtraParamsAndDir.update(0);
            continue;
        }
        intExtraParamsSuccess++;

        /* Sample incoming directions d_in&rec_wi based on its_in and
         * extraParams */
        Spectrum pdf_d_in_and_rec_wi;
        Spectrum bsdfVal = sampleDirectionsFromBssrdf(
                scene, its_out, d_out, its_in, d_in, rec_wi, bsdfMeasure,
                pdf_d_in_and_rec_wi, throughput, extraParams, sampler);
        if (bsdfVal.isZero()) {
            intExtraParamsAndDir.update(0);
            continue;
        }
        Assert(bsdfMeasure == check_bsdfMeasure);

        /* Evaluate BSSRDF */
        Spectrum bssrdfVal = bssrdf(scene, p_in, d_in, n_in, p_out,
                d_out, n_out, extraParams);
        if (bssrdfVal.isZero()) {
            intExtraParamsAndDir.update(0);
            continue;
        }

        Float cosTheta = math::abs(dot(d_in, n_in));
        if (cosTheta == 0) {
            intExtraParamsAndDir.update(0);
            continue;
        }
        Spectrum fullPdf = pdf_d_in_and_rec_wi*extraParamsPdf*surfacePdf;
        Float contribution = (bssrdfVal*bsdfVal/fullPdf).averageNonNan();
        if (absify)
            contribution = math::abs(contribution);
        if (!std::isfinite(contribution)) {
            intExtraParamsAndDir.update(0);
            continue;
        }
        intExtraParamsAndDir.update(contribution);
        intExtraParamsAndDirSuccess++;
    }

    Float extraParamsAndDirIntegral = intExtraParamsAndDir.mean();
    if (intExtraParamsAndDir.errorOfMean()
            / std::max(theIntegral, math::abs(intExtraParamsAndDir.mean()))
            > 0.3) {
        cerr << "!big error in intExtraParamsAndDir! weight: "
                << intExtraParamsAndDir.mean() << " +- " << intExtraParamsAndDir.errorOfMean()
                << " ["<<intExtraParamsAndDir.min()<<".."<<intExtraParamsAndDir.max()
                <<", succ"<<(float)intExtraParamsAndDirSuccess / numIntSamples<<"]"
                << " relInt: " << intExtraParamsAndDir.errorOfMean()/theIntegral
                << " relErr: " << intExtraParamsAndDir.errorOfMean()/math::abs(intExtraParamsAndDir.mean())
                << endl;
    }


    // INTEGRATE DIRECTION

    const void *constExtraParams = check_extraParams;
    Spectrum extraParamsPdf(1.0); // we don't integrate over this here

    VarianceFunctor intDirection;
    int intDirectionSuccess = 0;
    int intDirectionSampleOnlySuccess = 0;
    for (int i = 0;
            i < numIntSamples
            || (intDirectionSuccess == 0
                && i < continueWithZeroFactor*numIntSamples);
            i++) {
        /* Sample incoming directions d_in&rec_wi based on its_in and
         * extraParams */
        Spectrum pdf_d_in_and_rec_wi;
        Spectrum bsdfVal = sampleDirectionsFromBssrdf(
                scene, its_out, d_out, its_in, d_in, rec_wi, bsdfMeasure,
                pdf_d_in_and_rec_wi, throughput, constExtraParams, sampler);
        if (bsdfVal.isZero()) {
            intDirection.update(0);
            continue;
        }
        Assert(bsdfMeasure == check_bsdfMeasure);
        intDirectionSampleOnlySuccess++;

        /* Evaluate BSSRDF */
        Spectrum bssrdfVal = bssrdf(scene, p_in, d_in, n_in, p_out,
                d_out, n_out, constExtraParams);
        if (bssrdfVal.isZero()) {
            intDirection.update(0);
            continue;
        }

        Float cosTheta = math::abs(dot(d_in, n_in));
        if (cosTheta == 0) {
            intDirection.update(0);
            continue;
        }
        Spectrum fullPdf = pdf_d_in_and_rec_wi*extraParamsPdf*surfacePdf;
        Float contribution = (bssrdfVal*bsdfVal/fullPdf).averageNonNan();
        if (absify)
            contribution = math::abs(contribution);
        if (!std::isfinite(contribution)) {
            intDirection.update(0);
            continue;
        }
        intDirection.update(contribution);
        intDirectionSuccess++;
    }

    Float directionIntegral = intDirection.mean();
    if (intDirection.errorOfMean()
            / std::max(theIntegral, math::abs(intDirection.mean()))
            > 0.3) {
        cerr << "!big error in intDirection! weight: "
                << intDirection.mean() << " +- " << intDirection.errorOfMean()
                << " ["<<intDirection.min()<<".."<<intDirection.max()
                <<", succ"<<(float)intDirectionSuccess / numIntSamples<<"]"
                << " relInt: " << intDirection.errorOfMean()/theIntegral
                << " relErr: " << intDirection.errorOfMean()/math::abs(intDirection.mean())
                << endl;
    }


    Float idealSurfacePdf = extraParamsAndDirIntegral / theIntegral;
    Float idealExtraParamsPdf = directionIntegral / extraParamsAndDirIntegral;
    Float idealDirectionPdf = (check_bssrdfVal * check_bsdfVal).average() / directionIntegral;


#if 0
    // EEEEEEEEEEEWWWW hardwired for fwddip :P
    struct ExtraParams {
        Float lengths[SPECTRUM_SAMPLES];
    };
    ExtraParams *lengths = static_cast<ExtraParams*>((void*)check_extraParams);
#endif
    Float lRl = distance(its_out.p, check_its_in.p);

    cerr<< std::fixed << std::setprecision(3);
    cerr<<"abs:"<<absify<<" ";
    cerr<<"rel:"
        /* Fractional deviation in the pdf wrt ideal pdf. These should be
         * close to unity. Small values lead to fireflies! */
        <<" sur: "<<check_surfacePdf / idealSurfacePdf
        <<", ex: " << check_extraParamsPdf.average() / idealExtraParamsPdf
        <<", dir: " << check_pdf_d_in_and_rec_wi.average() / idealDirectionPdf
        /* Relative errors of the ideal pdf estimators themselves.
         * Order:
         *    full integral = surface & extraparams & directions
         *    extraparams & directions
         *    direction
         * The values in parentheses give a relative indication of how much
         * sampling attempts actually succeeded. */
        <<"   rel.errs:"
        <<" " << integral.errorOfMean()/theIntegral
            <<"("<<(float)intSurfaceSuccess/numIntSamples
            <<","<<(float)integralSuccess/numIntSamples<<")"
        <<" " << intExtraParamsAndDir.errorOfMean() / extraParamsAndDirIntegral
            <<"("<<(float)intExtraParamsSuccess/numIntSamples
            <<","<<(float)intExtraParamsAndDirSuccess/numIntSamples<<")"
        <<" " << intDirection.errorOfMean() / directionIntegral
            <<"("<<(float)intDirectionSampleOnlySuccess/numIntSamples
            <<","<<(float)intDirectionSuccess/numIntSamples<<")"
        /* The actual integral */
        <<"   int: " << theIntegral
        /* Some additional details */
        <<"  lRl: " << lRl
#if 0
        <<"  s/R: " << lengths->lengths[0] / lRl
#endif
        <<" dni: " << dot(check_d_in, n_in)
        <<" dno: " << dot(d_out, n_in)
        <<" Rni: " << dot(normalize(p_out - p_in), n_in)
        <<" Rdi: " << dot(normalize(p_out - p_in), check_d_in)
        <<" Rdo: " << dot(normalize(p_out - p_in), d_out)
        << endl;
}


/* Query Li(its_out,d) at intersection its_out. Sample new query point on
 * the surface of our shape with associated intersection its_in from which
 * we query a sample of dE_i = Li(its_in,omega)*cos(theta)*domega
 * ourselves, which we propagate with our bsdf to return a sample of the
 * requested Li(its_out,d) at its_out.
 *
 * Convention: d should point inwards, conforming to a traditional 'wi'
 * vector. */
Spectrum DirectSamplingSubsurface::Li_internal(const Scene *scene, Sampler *sampler,
        const Intersection &its_out, const Vector &d,
        const Spectrum &throughput, int &splits, int depth, int numInternalRefl) const {

    const Vector d_out = -d;
    const Normal n_out = its_out.shFrame.n;

    /* If eta != 1, then this is our cue that the user wants to have the
     * subsurface model take into account the extra influx due to internal
     * reflections itself already. So don't count this twice. */
    bool allowInternalReflection = (m_eta == 1);

    if (!m_allowIncomingOutgoingDirections && dot(n_out, d_out) <= 0)
        return Spectrum(0.0f);

    Spectrum channelWeight = handleSingleChannel(throughput, sampler);
    /* Already take into account that we could have transitioned to a
     * single channel mode -- we use this for sampling decisions
     * (pointOnSurface & bssdf) */
    Spectrum channelWeightedThroughput = throughput * channelWeight;

    const MonteCarloIntegrator *integrator =
            dynamic_cast<const MonteCarloIntegrator*>(scene->getIntegrator());
    if (integrator == NULL) {
        Log(EError, "Need a Monte Carlo integrator, but got %s",
                scene->getIntegrator()->toString().c_str());
    }
    Assert(integrator != NULL);


    /* Sample an incoming query point through an 'indirect' sampling of the
     * bssrdf, possibly combined with a direct sampling of the light
     * sources. */
    IndirectSamplingRecord indirectSample;
    char extraParams[extraParamsSize()];
    Spectrum LiDirect;
    bool haveIndirect;
    if (m_numSIR > 1) {
        haveIndirect = indirectSample_SIR(scene, sampler, its_out, d_out,
                channelWeight, channelWeightedThroughput, LiDirect,
                indirectSample, extraParams);
    } else {
        haveIndirect = indirectSample_noSIR(scene, sampler, its_out, d_out,
                channelWeight, channelWeightedThroughput, LiDirect,
                indirectSample, extraParams);
    }
    Spectrum result = LiDirect;
    if (!haveIndirect)
        goto DSS_Li_radianceSourceSampling;


    /* Recursively gather any type of radiance for the indirect sample */
    {
        const Point  &p_in          = indirectSample.its_in.p;
        const Normal &n_in          = indirectSample.its_in.shFrame.n;
        const Vector &rec_wi        = indirectSample.rec_wi;
        const Intersection &its_in  = indirectSample.its_in;

        if (dot(rec_wi, n_in) < 0) { /* Internal reflection */
            if (!allowInternalReflection)
                goto DSS_Li_radianceSourceSampling;

            numInternalRefl++;

            if (m_maxInternalReflections >= 0
                    && numInternalRefl > m_maxInternalReflections)
                goto DSS_Li_radianceSourceSampling;

            /* If we have internal reflection: don't go to the integrator
             * again, but just handle this through local recursion right
             * here until we break out of the medium. */
             // internal reflection, so there is no direct Li here
            Spectrum thisWeight = indirectSample.weightForIndirectContrib;
            Spectrum newThroughput = throughput * thisWeight;
            if (newThroughput.isZero())
                goto DSS_Li_radianceSourceSampling;

            // Russian Roulette and path splitting
            int numSplitsHere = 0;
            Float q = integrator->getRR().roulette(
                    depth, newThroughput, 1.0, sampler);
            if (q == 0)
                goto DSS_Li_radianceSourceSampling;
            if (q == 1) { // There was no RR -> possibly try path splitting
                numSplitsHere = integrator->getRR().split(
                        splits, newThroughput, 1.0, sampler);
            }
            avgNumSplits.incrementBase();
            avgNumSplits += numSplitsHere;

#if MTS_DSS_CHECK_VARIANCE_SOURCES
            const Vector &d_in          = indirectSample.d_in;
            const EMeasure &bsdfMeasure = indirectSample.bsdfMeasure;
            if (thisWeight.maxAbsolute()
                    > MTS_DSS_CHECK_VARIANCE_SOURCES_THRESHOLD) {
                cout << "indirect internal reflection"
                        << thisWeight.toString() << endl;
                checkSourcesOfVariance(scene, channelWeightedThroughput,
                        its_out, d_out, its_in, d_in, rec_wi, extraParams,
                        bssrdfVal, bsdfVal, bsdfMeasure, sampler, true);
                checkSourcesOfVariance(scene, channelWeightedThroughput,
                        its_out, d_out, its_in, d_in, rec_wi, extraParams,
                        bssrdfVal, bsdfVal, bsdfMeasure, sampler, false);
            }
#endif

            int n = numSplitsHere + 1;
            thisWeight /= (q * n);
            newThroughput /= (q * n);
            for (int s = 0; s < n; s++) {
                Spectrum Li = DirectSamplingSubsurface::Li_internal(
                        scene, sampler, its_in, rec_wi,
                        newThroughput, splits, depth + 1, numInternalRefl);
                result += Li * thisWeight;
            }
        } else {
            /* Ray exits our medium, possible after an internal reflection chain */
            if (numInternalRefl > 0) {
                avgIntReflChainLen.incrementBase();
                avgIntReflChainLen += numInternalRefl;
            }

            /* Ray exits our medium, query the integrator for Li */
            RadianceQueryRecord rRecBase(scene, sampler);
            RadianceQueryRecord rRec;
            rRecBase.depth = depth;
            rRecBase.splits = splits;
            rRecBase.medium = its_in.getTargetMedium(1.0f);
            RayDifferential ray(p_in, rec_wi, its_out.time);

            /* Get separate direct contribution if required. */
            RadianceQueryRecord::ERadianceQuery integratorQuery; // 'the rest'
            if (m_directSampling) {
                Spectrum wgt = indirectSample.weightForDirectContrib;
                Assert(m_directSamplingMIS || wgt.isZero()); // !MIS => wgt==0
                if (!wgt.isZero()) {
                    rRec.recursiveQuery(rRecBase,
                            RadianceQueryRecord::EEmittedRadiance
                                | RadianceQueryRecord::EIntersection,
                            throughput * wgt);
                    Spectrum directLi = integrator->Li(ray, rRec);
                    rRecBase.splits = rRec.splits; // should be no-op...
                    result += directLi * wgt;
                }

                /* Request only indirect illumination below */
                integratorQuery = RadianceQueryRecord::ERadianceNoEmission;
            } else {
                /* No separate direct contribution, so gather any type of
                 * radiance recursively. */
                integratorQuery = RadianceQueryRecord::ERadiance;
            }


            /* We could choose to do RR here already, or pass it to the
             * integrator unscathed as a form of protection for a possibly
             * huge weight coming up in the very next step. Either way, the
             * integrator will do splitting for us if required. Currently
             * we skip RR. */
            Spectrum thisWeight = indirectSample.weightForIndirectContrib;
#if MTS_DSS_CHECK_VARIANCE_SOURCES
            if (thisWeight.maxAbsolute()
                    > MTS_DSS_CHECK_VARIANCE_SOURCES_THRESHOLD) {
                cout << "indirect query" << thisWeight.toString() << endl;
                checkSourcesOfVariance(scene, channelWeightedThroughput,
                        its_out, d_out, its_in, d_in, rec_wi, extraParams,
                        bssrdfVal, bsdfVal, bsdfMeasure, sampler, true);
                checkSourcesOfVariance(scene, channelWeightedThroughput,
                        its_out, d_out, its_in, d_in, rec_wi, extraParams,
                        bssrdfVal, bsdfVal, bsdfMeasure, sampler, false);
            }
#endif

            // Get the actual indirect contribution from the integrator
            rRec.recursiveQuery(rRecBase, integratorQuery, thisWeight);
            Spectrum Li = integrator->Li(ray, rRec);
            result += Li * thisWeight;
            splits = rRec.splits;
        }
    }

DSS_Li_radianceSourceSampling:
#if MTS_DSS_USE_RADIANCE_SOURCES /* do radiance source sampling? */
    const Point p_out = its_out.p;
    /* Add light sources that cannot be sampled */
    Assert(m_sources.get());
    for (const RadianceSource rs : m_sources->get()) {
        // Make fake intersection record
        Intersection its_in;
        its_in.shFrame.n = rs.n;
        its_in.geoFrame.n = rs.n;
        its_in.p = rs.p;
        Spectrum extraParamsPdf = sampleExtraParams(
                scene, its_out, d_out, its_in, &rs.d, channelWeight,
                extraParams, sampler);
        if (extraParamsPdf.isZero())
            continue;
        result += rs.Li * channelWeight * extraParamsPdf.invertButKeepZero()
                * bssrdf(scene, rs.p, rs.d, rs.n,
                        p_out, d_out, n_out, extraParams);
    }
#endif

    if (m_singleChannel)
        Assert(result.numNonZeroChannels() <= 1);

    return result;
}



/**
 * Sample Importance Resampling (SIR) of direct&indirect MIS weighted
 * sampling (with full expected value estimator for the direct
 * contribution) */
bool DirectSamplingSubsurface::indirectSample_SIR(
        const Scene *scene, Sampler *sampler,
        const Intersection &its_out, const Vector &d_out,
        const Spectrum &channelWeight,
        const Spectrum &channelWeightedThroughput,
        Spectrum &LiDirectContribution,
        IndirectSamplingRecord &indirectSample, void * extraParams) const {
    const Point  &p_out = its_out.p;
    const Normal &n_out = its_out.shFrame.n;
    LiDirectContribution = Spectrum(0.0f);

    /* Don't bother trying to sample if all light sources that are present
     * are delta sources. */
    if (!m_nonCollimatedLightSourcesPresent)
        return false;

    std::vector<IndirectSamplingRecord> samples;
    samples.reserve(m_numSIR);
    size_t parSize = extraParamsSize();
    char *paramSamples = (parSize == 0 ? NULL : new char[m_numSIR * parSize]);
    DiscreteDistribution sampleWeights;
    sampleWeights.reserve(m_numSIR);

    for (size_t i = 0; i < m_numSIR; i++) {
        IndirectSamplingRecord s;
        void *extraPars = paramSamples + i*parSize;
        Intersection its_in;
        Vector d_in, rec_wi;
        EMeasure bsdfMeasure;// = EInvalidMeasure;

        /* Both direct and indirect illumination sampling start by
         * sampling a point on the surface, so we can share the sampled
         * point. */

        /* Sample BSSRDF marginalized over incoming direction and
         * extraParams */
        Float thisSurfacePdf = samplePointOnSurface(
                its_out, d_out, scene, its_in,
                channelWeightedThroughput, sampler);
        const Point &p_in = its_in.p;
        const Normal &n_in = its_in.shFrame.n;
        if (thisSurfacePdf == 0)
            continue;

        do {
            Assert(m_directSampling && m_directSamplingMIS);

            /* DIRECT SAMPLING */

            /* This seems incredibly wasteful (to do direct lighting
             * estimation for each tentative sample), but remember that
             * we only shoot one shadow ray here, and sampling points
             * on the surface typically shoot many projection rays
             * (collecting all intersections, although these rays have
             * limited range).  Also, there's not really a way to MIS
             * sample the direct lighting of the single, final SIR
             * sample; because we cannot easily compute the actual pdf
             * in general that is used for SIR (which would need to be
             * marginalized over all possible tentative SIR sample
             * generations!) */
            Spectrum directLi, directBsdfVal;
            char directExtraParams[extraParamsSize()];
            EMeasure lightSamplingMeasure;
            EMeasure directBsdfMeasure;
            Spectrum directPdf = sampleDirect(scene, sampler, its_in,
                    its_out, d_out, channelWeightedThroughput, d_in, rec_wi,
                    directExtraParams, directBsdfMeasure, lightSamplingMeasure,
                    directBsdfVal, directLi);
            if (directPdf.isZero())
                break;

            /* Prevent light leaks due to the use of shading normals */
            if (dot(its_in.geoFrame.n, rec_wi)*dot(n_in, rec_wi) <= 0
                    || dot(its_in.geoFrame.n, d_in)*dot(n_in, d_in) <= 0)
                break;

            /* Evaluate BSSRDF */
            Spectrum directBssrdfVal = bssrdf(scene, p_in, d_in, n_in, p_out,
                    d_out, n_out, directExtraParams);
            if (directBssrdfVal.isZero())
                break;

            /* MIS weighting of the sampling of d_in, rec_wi and
             * directExtraParams */
            Spectrum directMISpdf;
            if (lightSamplingMeasure == EDiscrete) {
                 // discrete light source, could not have been sampled indirectly
                directMISpdf = directPdf;
            } else {
                Spectrum indirectPdf = pdfIndirect(scene, its_in,
                        its_out, d_out, channelWeightedThroughput, d_in, rec_wi,
                        directExtraParams, directBsdfMeasure);

                directMISpdf = directPdf + indirectPdf;
            }

            Spectrum directWeight =
                    channelWeight * directBsdfVal * directBssrdfVal
                            * directMISpdf.invertButKeepZero() / thisSurfacePdf;

#if MTS_DSS_CHECK_VARIANCE_SOURCES
            Spectrum thisWeight = directWeight;// / m_numSIR;
            if (thisWeight.maxAbsolute()
                    > MTS_DSS_CHECK_VARIANCE_SOURCES_THRESHOLD) {
                cout << "direct samp " << thisWeight.toString() << endl;
                checkSourcesOfVariance(scene, channelWeightedThroughput,
                        its_out, d_out, its_in, d_in, rec_wi,
                        directExtraParams, directBssrdfVal, directBsdfVal,
                        directBsdfMeasure, sampler, true);
                checkSourcesOfVariance(scene, channelWeightedThroughput,
                        its_out, d_out, its_in, d_in, rec_wi,
                        directExtraParams, directBssrdfVal, directBsdfVal,
                        directBsdfMeasure, sampler, false);
            }
#endif
            // expected value estimator for direct lighting
            LiDirectContribution += directLi * directWeight / m_numSIR;
        } while (false); // hack for break;



        /* INDIRECT SAMPLING to generate tentative SIR samples */

        /* Sample directions and extra parameters */
        Spectrum bsdfVal;
        Spectrum indirectPdf = sampleIndirect(scene, sampler, its_in,
                its_out, d_out, channelWeightedThroughput, d_in, rec_wi,
                extraPars, bsdfMeasure, bsdfVal);
        if (indirectPdf.isZero())
            continue;

        /* Prevent light leaks due to the use of shading normals */
        if (dot(its_in.geoFrame.n, rec_wi)*dot(n_in, rec_wi) <= 0
                || dot(its_in.geoFrame.n, d_in)*dot(n_in, d_in) <= 0)
            continue;

        /* Evaluate BSSRDF */
        Spectrum bssrdfVal = bssrdf(scene, p_in, d_in, n_in, p_out, d_out,
                n_out, extraPars);
        if (bssrdfVal.isZero())
            continue;

        /* Weight factor (excluding the directions&extraParams pdf) */
        Spectrum factor = channelWeight * bsdfVal * bssrdfVal
                / thisSurfacePdf;
        if (factor.isZero())
            continue;

        /* pdf for the direct Li contribution of the 'indirectly sampled'
         * part */
        Spectrum pdfForDirectContrib;
        if (dot(rec_wi, n_in) < 0) {
            /* Internal reflection, guaranteed not to find direct
             * contribution (because we don't allow light sources within
             * our medium -- TODO: for now?), so no need to calculate
             * direct pdf as it will be zero. */
            pdfForDirectContrib = indirectPdf;
        } else {
            Assert(m_directSamplingMIS);
            /* Compute the MIS pdf for the direct Li contribution for this
             * indirect sampling in combination with direct sampling (2
             * sample MIS: 1 direct, 1 indirect -- the direct sample gets
             * taken below) */
            Spectrum directPdf = pdfDirect(scene, its_in, its_out,
                    d_out, channelWeightedThroughput, d_in, rec_wi,
                    extraPars, bsdfMeasure); // note: this shoots a shadow ray!
            pdfForDirectContrib = directPdf + indirectPdf;
        }
        // Store the tentative sample
        s.its_in      = its_in;
        s.d_in        = d_in;
        s.rec_wi      = rec_wi;
        s.bsdfMeasure = bsdfMeasure;
        s.weightForDirectContrib   = factor * pdfForDirectContrib.invertButKeepZero();
        s.weightForIndirectContrib = factor * indirectPdf.invertButKeepZero();

        Float theSampleWeight = s.weightForIndirectContrib.maxAbsolute();
        if (!std::isfinite(theSampleWeight) || theSampleWeight < 0) {
            Log(EWarn, "Problematic sample weight: %e", theSampleWeight);
        } else if (theSampleWeight > 0) {
            /* Store the tentative sample */
            sampleWeights.append(theSampleWeight);
            samples.push_back(s);
            Assert(math::abs(s.its_in.shFrame.n.length() - 1) < Epsilon);
        }
    }

    Assert(sampleWeights.size() == samples.size());
    Assert(samples.size() <= m_numSIR);

    if (samples.size() == 0)
        return false;

    sampleWeights.normalize();
    if (sampleWeights.getSum() == 0)
        return false;

    Float sampleProb;
    size_t idx = sampleWeights.sample(sampler->next1D(), sampleProb);
    Assert(idx >= 0 && idx < samples.size());

    indirectSample = samples[idx];
    /* Update weights with the discrete SIR prob */
    indirectSample.weightForDirectContrib   /= (m_numSIR * sampleProb);
    indirectSample.weightForIndirectContrib /= (m_numSIR * sampleProb);
    if (paramSamples) {
        memcpy(extraParams, paramSamples + idx*parSize, parSize);
        delete[] paramSamples;
    }
    return true;
}






/**
 * Sample an incoming query point on the surface, with seperate direct &
 * indirect sampling strategies for the directions & extraParams if
 * requested (optionally MIS weighted), but without Sample Importance
 * Resampling (SIR) */
bool DirectSamplingSubsurface::indirectSample_noSIR(
        const Scene *scene, Sampler *sampler,
        const Intersection &its_out, const Vector &d_out,
        const Spectrum &channelWeight,
        const Spectrum &channelWeightedThroughput,
        Spectrum &LiDirectContribution,
        IndirectSamplingRecord &indirectSample, void * extraParams) const {
    const Point  p_out = its_out.p;
    const Normal n_out = its_out.shFrame.n;
    LiDirectContribution = Spectrum(0.0f);
    Vector d_in, rec_wi;
    EMeasure bsdfMeasure;

    /* Sample BSSRDF marginalized over incoming direction and extraParams */
    Intersection its_in;
    Float surfacePdf = samplePointOnSurface(
            its_out, d_out, scene, its_in, channelWeightedThroughput, sampler);
    const Point  &p_in = its_in.p;
    const Normal &n_in = its_in.shFrame.n;
    if (surfacePdf == 0)
        return false;

    /* DIRECT SAMPLING */
    if (m_directSampling) { do {
        Spectrum LiDirect, bsdfVal;
        EMeasure lightSamplingMeasure;
        Spectrum directPdf = sampleDirect(scene, sampler, its_in,
                its_out, d_out, channelWeightedThroughput, d_in, rec_wi,
                extraParams, bsdfMeasure, lightSamplingMeasure,
                bsdfVal, LiDirect);
        if (directPdf.isZero())
            break;

        /* Prevent light leaks due to the use of shading normals */
        if (dot(its_in.geoFrame.n, rec_wi)*dot(n_in, rec_wi) <= 0
                || dot(its_in.geoFrame.n, d_in)*dot(n_in, d_in) <= 0)
            break;

        /* Evaluate BSSRDF */
        Spectrum bssrdfVal = bssrdf(scene, p_in, d_in, n_in, p_out,
                d_out, n_out, extraParams);
        if (bssrdfVal.isZero())
            break;

        /* Optional MIS weighting of the sampling of d_in, rec_wi and
         * extraParams */
        Spectrum effectivePdf;
        if (m_directSamplingMIS && lightSamplingMeasure != EDiscrete) {
            Spectrum indirectPdf = pdfIndirect(scene, its_in,
                    its_out, d_out, channelWeightedThroughput, d_in, rec_wi,
                    extraParams, bsdfMeasure);

            effectivePdf = directPdf + indirectPdf;
            /* effectivePdf not weighted by 0.5 because we take another MIS
             * sample below */
        } else {
            effectivePdf = directPdf;
        }

        Spectrum directWeight = channelWeight * bsdfVal * bssrdfVal
                        * effectivePdf.invertButKeepZero() / surfacePdf;

#if MTS_DSS_CHECK_VARIANCE_SOURCES
        Spectrum thisWeight = directWeight;
        if (thisWeight.maxAbsolute()
                > MTS_DSS_CHECK_VARIANCE_SOURCES_THRESHOLD) {
            cout << "direct samp " << thisWeight.toString() << endl;
            checkSourcesOfVariance(scene, channelWeightedThroughput,
                    its_out, d_out, its_in, d_in, rec_wi, extraParams,
                    bssrdfVal, bsdfVal, bsdfMeasure, sampler, true);
            checkSourcesOfVariance(scene, channelWeightedThroughput,
                    its_out, d_out, its_in, d_in, rec_wi, extraParams,
                    bssrdfVal, bsdfVal, bsdfMeasure, sampler, false);
        }
#endif

        LiDirectContribution = LiDirect * directWeight;
    } while (false); } /* hack for break */



    /* INDIRECT SAMPLING */
    Spectrum bsdfVal;
    Spectrum indirectPdf = sampleIndirect(scene, sampler, its_in,
            its_out, d_out, channelWeightedThroughput, d_in, rec_wi,
            extraParams, bsdfMeasure, bsdfVal);
    if (indirectPdf.isZero())
        return false;

    /* Prevent light leaks due to the use of shading normals */
    if (dot(its_in.geoFrame.n, rec_wi)*dot(n_in, rec_wi) <= 0
            || dot(its_in.geoFrame.n, d_in)*dot(n_in, d_in) <= 0)
        return false;

    /* Evaluate BSSRDF */
    Spectrum bssrdfVal = bssrdf(scene, p_in, d_in, n_in, p_out, d_out,
            n_out, extraParams);
    if (bssrdfVal.isZero())
        return false;

    /* Weight factor (excluding the directions&extraParams pdf) */
    Spectrum factor = channelWeight * bsdfVal * bssrdfVal / surfacePdf;
    if (factor.isZero())
        return false;

    /* pdf for the direct Li contribution of the 'indirectly sampled' part */
    Spectrum pdfForDirectContrib;
    if (dot(rec_wi, n_in) < 0 || !m_directSampling) {
        /* Internal reflection or no direct sampling.
         * For internal reflection, guaranteed not to find direct
         * contribution (because we don't allow light sources within
         * our medium -- TODO: for now?), so no need to calculate
         * direct pdf as it will be zero. */
        pdfForDirectContrib = Spectrum(0.0f);
    } else {
        /* case: direct sampling and outgoing direction */
        Spectrum directPdf = pdfDirect(scene, its_in, its_out, d_out,
                channelWeightedThroughput, d_in, rec_wi, extraParams,
                bsdfMeasure); // note: this shoots a shadow ray!
        if (m_directSamplingMIS) {
            /* Compute the MIS pdf for the direct Li contribution for this
             * indirect sampling in combination with direct sampling (2
             * sample MIS: 1 direct, 1 indirect -- the direct sample gets
             * taken below) */
            pdfForDirectContrib = directPdf + indirectPdf;
        } else {
            /* All radiance is in the 'indirect' contribution, don't double
             * count the directContrib! */
            pdfForDirectContrib = Spectrum(0.0f);
        }
    }

    IndirectSamplingRecord &s = indirectSample;
    s.its_in      = its_in;
    s.d_in        = d_in;
    s.rec_wi      = rec_wi;
    s.bsdfMeasure = bsdfMeasure;
    s.weightForDirectContrib   = factor * pdfForDirectContrib.invertButKeepZero();
    s.weightForIndirectContrib = factor * indirectPdf.invertButKeepZero();
    return true;
}


Spectrum DirectSamplingSubsurface::Lo(const Scene *scene, Sampler *sampler,
        const Intersection &its_out, const Vector &d_out,
        const Spectrum &throughput, int depth) const {
    Log(EError, "TODO");
    return Spectrum(0.0f);
}


void DSSProjFrame::getProjFrame(Vector &u, Vector &v, Vector &projDir,
        const Intersection &its_out, const Vector &d_out) const {
    return getProjFrame(u, v, projDir, its_out.shFrame.n, d_out);
}

void DSSProjFrame::getProjFrame(Vector &u, Vector &v, Vector &projDir,
        const Vector &n_out, const Vector &d_out) const {
    switch (m_projType) {
        case ENormalNormal:
        case ENormalForward:
        case ENormalSide: {
            Vector norm,fwd,side;
            norm = n_out;
            if (math::abs(dot(d_out,norm)) > 1 - Epsilon) {
                /* Forward direction badly conditioned -- just pick *some*
                 * directions */
                Frame normalFrame(norm);
                fwd  = normalFrame.s;
                side = normalFrame.t;
            } else {
                // forward (along planar d_out component)
                fwd  = normalize(d_out - norm*dot(d_out,norm));
                // side (sign does not matter because of symmetry)
                side = cross(norm,fwd);
            }
            switch (m_projType) {
            case ENormalNormal:
                projDir = norm;
                u = fwd;
                v = side;
                break;
            case ENormalForward:
                projDir = fwd;
                u = norm;
                v = side;
                break;
            case ENormalSide:
                projDir = side;
                u = norm;
                v = fwd;
                break;
            default: SAssert(false);
            }
            break;
        }
        case EDirectionDirection:
        case EDirectionOut:
        case EDirectionSide: {
            Vector dir,out,side;
            dir = d_out;
            if (math::abs(dot(dir,n_out)) > 1 - Epsilon) {
                /* Outward direction badly conditioned -- just pick *some*
                 * directions */
                Frame directionFrame(dir);
                out  = directionFrame.s;
                side = directionFrame.t;
            } else {
                out  = normalize(n_out - dir*dot(dir,n_out));
                side = cross(dir,out); // sign doesn't matter due to symmetry
            }
            switch (m_projType) {
            case EDirectionDirection:
                projDir = dir;
                u = out;
                v = side;
                break;
            case EDirectionOut:
                projDir = out;
                u = dir;
                v = side;
                break;
            case EDirectionSide:
                projDir = side;
                u = dir;
                v = out;
                break;
            default: SAssert(false);
            }
            break;
        }
        default:
            SLog(EError, "Unknown DSSTangentFrameType: %d", m_projType);
    }
    SAssert(fabs(dot(u, v)) < ShadowEpsilon);
    SAssert(fabs(dot(projDir, u)) < ShadowEpsilon);
    SAssert(fabs(dot(projDir, v)) < ShadowEpsilon);
    SAssert(fabs(u.length() - 1) < ShadowEpsilon);
    SAssert(fabs(v.length() - 1) < ShadowEpsilon);
    SAssert(fabs(projDir.length() - 1) < ShadowEpsilon);
}



MTS_IMPLEMENT_CLASS_IS(RadianceSources, false, SerializableObject)
MTS_IMPLEMENT_CLASS(DirectSamplingSubsurface, true, Subsurface)

MTS_IMPLEMENT_CLASS(Sampler1D, true, Object)
MTS_IMPLEMENT_CLASS(ExpSampler1D, false, Sampler1D)
MTS_IMPLEMENT_CLASS(RadialExactDipoleSampler2D, false, Sampler1D)

MTS_IMPLEMENT_CLASS(TangentSampler2D, true, Object)
MTS_IMPLEMENT_CLASS(MISTangentSampler2D, false, TangentSampler2D)
MTS_IMPLEMENT_CLASS(RadialSampler2D, false, TangentSampler2D)

MTS_IMPLEMENT_CLASS(IntersectionSampler, true, Object)
MTS_IMPLEMENT_CLASS(MISIntersectionSampler, false, IntersectionSampler)
MTS_IMPLEMENT_CLASS(WeightIntersectionSampler, false, IntersectionSampler)

MTS_IMPLEMENT_CLASS(SurfaceSampler, true, Object)
MTS_IMPLEMENT_CLASS(UniformSurfaceSampler, false, SurfaceSampler)
MTS_IMPLEMENT_CLASS(BRDFDeltaSurfaceSampler, false, SurfaceSampler)
MTS_IMPLEMENT_CLASS(ProjSurfaceSampler, false, SurfaceSampler)

MTS_NAMESPACE_END

