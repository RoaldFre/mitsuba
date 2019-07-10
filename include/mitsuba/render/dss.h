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

#pragma once
#if !defined(__MITSUBA_RENDER_DSS_H_)
#define __MITSUBA_RENDER_DSS_H_

#include <mitsuba/render/subsurface.h>
#include <mitsuba/core/pmf.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/shape.h>
#include <gsl/gsl_sf_lambert.h>

#include <iomanip>
#include <functional>

/* When sampling a surface through projection, don't accept the sample if
 * the cosine of the angle between the sampled surface normal and the
 * projection direction becomes less than this value, in order to avoid bad
 * conditioning. Bias is avoided by combining two complementary strategies
 * (e.g. sampling by projecting some distribution in a 2D plane along
 * multiple projection directions), although there is still some slight
 * multiplicative bias because the resulting PDF will not be normalized
 * exactly (but this will be a very small deviation precisely because the
 * cosine [and hence the pdf] would be very low there anyway)
 *
 * Warning: only double precision is thoroughly tested! */
#ifdef SINGLE_PRECISION
#  define MTS_DSS_COSINE_CUTOFF 1e-3
#else
#  define MTS_DSS_COSINE_CUTOFF 1e-5
#endif

//#ifdef MTS_DEBUG
//#    define MTS_DSS_PDF_CHECK
//#endif

/* Allow incoming internal directions? Normally false, but can be useful
 * for debugging. */
#define MTS_DSS_ALLOW_INTERNAL_INCOMING_DIR false

MTS_NAMESPACE_BEGIN

/**
 * Wrapper for averaging a value over channels that explicitly handles the
 * single-channel case.
 *
 * Note: not using std::function wrapper but purely generic F for
 * inlining/optimization reasons.
 *
 * \param f A callable object (function [pointer] or lambda expression)
 *          that takes a channel-index as single argument and returns a
 *          float.
 * \param channel If set to -1: average over all channels, otherwise only
 *                return the result for this given channel.
 */
template<typename F>
inline Float channelMean(int channel, const F f) {
    if (channel == -1) {
        Float sum = 0;
        for (int i = 0; i < SPECTRUM_SAMPLES; i++) {
            sum += f(i);
        }
        return sum / SPECTRUM_SAMPLES;
    } else {
        return f(channel);
    }
}

/**
 * Wrapper for averaging a value over channels for which the given
 * throughput is nonzero.
 *
 * Note: not using std::function wrapper but purely generic F for
 * inlining/optimization reasons.
 *
 * \param f A callable object (function [pointer] or lambda expression)
 *          that takes a channel-index as single argument and returns a
 *          float.
 */
template<typename F>
inline Float nonzeroThroughputMean(const Spectrum &throughput, const F f) {
    Float sum = 0;
    int N = 0;
    for (int i = 0; i < SPECTRUM_SAMPLES; i++) {
        if (throughput[i] != 0) {
            sum += f(i);
            N++;
        }
    }
    return sum / N;
}


/**
 * Determines the frame of reference for the projection that is used when
 * sampling query points on the surface of a DSS. */
struct MTS_EXPORT_RENDER DSSProjFrame {
public:
    /**
     * The projection types are based on a tangent frame (e.g. tangent to
     * the outgoing normal or to the outgoing direction), from which one of
     * the three coordinate axes is chosen as a projection direction. */
    enum MTS_EXPORT_RENDER ProjType {
        ENormalNormal,       /// normal frame, project along normal
        ENormalForward,      /// normal frame, project along 'forward' direction (d_out component)
        ENormalSide,         /// normal frame, project along 'side' direction
        EDirectionDirection, /// direction frame, project along direction
        EDirectionOut,       /// direction frame, project along 'out' direction (n_out component)
        EDirectionSide,      /// direction frame, project along 'side' direction
    };

    DSSProjFrame(ProjType type) : m_projType(type) { }

    void getProjFrame(Vector &u, Vector &v, Vector &projDir,
            const Intersection &its, const Vector &d_out) const;

    void getProjFrame(Vector &u, Vector &v, Vector &projDir,
            const Vector &n_out, const Vector &d_out) const;

protected:
    ProjType m_projType;
};


struct MTS_EXPORT_RENDER RadianceSource {
    Point p;
    Normal n;
    Vector d;
    Spectrum Li;
    RadianceSource() { }
    RadianceSource(Point p, Normal n, Vector d, Spectrum Li) :
        p(p), n(n), d(d), Li(Li) { }
    RadianceSource(Stream *stream) :
        p(stream), n(stream), d(stream), Li(stream) { }
    void serialize(Stream *stream) const {
        p.serialize(stream);
        n.serialize(stream);
        d.serialize(stream);
        Li.serialize(stream);
    }
};

class MTS_EXPORT_RENDER RadianceSources : public SerializableObject {
public:
    RadianceSources() { }
    RadianceSources(Stream *stream, InstanceManager *manager) {
        size_t n = stream->readSize();
        m_sources.reserve(n);
        for (size_t i = 0; i < n; i++)
            m_sources.push_back(RadianceSource(stream));
    }
    void serialize(Stream *stream, InstanceManager *manager) const {
        stream->writeSize(m_sources.size());
        for (const RadianceSource rs : m_sources)
            rs.serialize(stream);
    }
    const std::vector<RadianceSource> &get() const {
        return m_sources;
    }
    void clear() {
        m_sources.clear();
    }
    void push_back(const RadianceSource &rs) {
        m_sources.push_back(rs);
    }
    MTS_DECLARE_CLASS();
protected:
    std::vector<RadianceSource> m_sources;
};


// channel can be -1 (= MIS over all channels)
class MTS_EXPORT_RENDER IntersectionSampler : public Object {
public:
    IntersectionSampler(Float itsDistanceCutoff) :
        m_itsDistanceCutoff(itsDistanceCutoff) { }

    virtual Float sample(const std::vector<Intersection> &intersections,
            Intersection &newIts,
            const Intersection &its_out, const Vector &d_out,
            int channel, Sampler *sampler) const = 0;

    virtual Float pdf(const std::vector<Intersection> &intersections,
            const Intersection &newIts,
            const Intersection &its_out, const Vector &d_out,
            int channel) const = 0;

    Float sample(Intersection &newIts,
            const Scene *scene, const Point &origin, const Vector &direction,
            Float time, const std::vector<Shape *> &shapes,
            const Intersection &its_out, const Vector &d_out,
            int channel, Sampler *sampler, bool bidirectional = true) const {
        const std::vector<Intersection> intersections =
                collectIntersections(scene, origin, direction, time,
                        shapes, its_out, bidirectional);
        if (intersections.size() == 0)
            return 0;
        return sample(intersections, newIts, its_out, d_out, channel, sampler);
    }

    Float pdf(const Intersection &newIts,
            const Scene *scene, const Point &origin, const Vector &direction,
            Float time, const std::vector<Shape *> &shapes,
            const Intersection &its_out, const Vector &d_out,
            int channel, bool bidirectional = true) const {
        const std::vector<Intersection> intersections =
                collectIntersections(scene, origin, direction, time,
                        shapes, its_out, bidirectional);
        if (intersections.size() == 0) {
            SLog(EWarn, "Could not find any intersection, not even our own!");
            return 0.0f;
        }
        return pdf(intersections, newIts, its_out, d_out, channel);
    }

    std::vector<Intersection> collectIntersections(const Scene *scene,
            const Point &origin, const Vector &direction, Float time,
            const std::vector<Shape *> &shapes, const Intersection &its_out,
            bool bidirectional = true) const {
        std::vector<Intersection> intersections;

        /* Find min and max t values that correspond to the
         * m_itsDistanceCutoff range around the its_out.p query point
         * (trivially equal to +/-m_itsDistanceCutoff in case the
         * projection origin coincides with its_out.p, but slightly more
         * involved if that is not the case) */
        Vector centerOffset = origin - its_out.p;
        Float offset2 = centerOffset.lengthSquared();
        Float cutoff2 = math::square(m_itsDistanceCutoff);

        Float proj = dot(centerOffset, direction);
        Float tmp2 = proj*proj - offset2 + cutoff2;
        if (tmp2 < 0)
            return intersections; // no intersection

        Float tmp = sqrt(tmp2);
        Float max_t = -proj + tmp;
        Float min_t = (bidirectional ? -proj - tmp : Epsilon);

        Float maxDist = m_itsDistanceCutoff * (1+Epsilon);
        Assert(distance(its_out.p, origin + direction * max_t) <= maxDist);
        Assert(distance(its_out.p, origin + direction * min_t) <= maxDist);

        scene->rayIntersectFully(Ray(origin,direction,min_t,max_t,time),
                intersections, &shapes);
        return intersections;
    }

    Float getItsDistanceCutoff() const {
        return m_itsDistanceCutoff;
    }

protected:
    virtual ~IntersectionSampler() { }
    MTS_DECLARE_CLASS();

    Float m_itsDistanceCutoff;
};

// channel can be -1 (= MIS over all channels)
class MTS_EXPORT_RENDER MISIntersectionSampler : public IntersectionSampler {
public:
    MISIntersectionSampler(const std::vector<std::pair<Float, const IntersectionSampler*> > &samplers) :
            IntersectionSampler(0.0f) {
        if (samplers.size() < 1)
            Log(EError, "Trying to construct MISIntersectionSampler without "
                    "any samplers!");
        m_samplers.reserve(samplers.size());
        m_weights.reserve(samplers.size());
        for (auto p : samplers) {
            if (p.first <= 0)
                continue;
            m_weights.append(p.first);
            m_samplers.push_back(p.second);
            m_itsDistanceCutoff = std::max(m_itsDistanceCutoff,
                    p.second->m_itsDistanceCutoff);
        }
        if (m_samplers.size() < 1)
            Log(EError, "Trying to construct MISIntersectionSampler without "
                    "any strictly positive weight samplers!");
        m_weights.normalize();
    }


    Float sample(const std::vector<Intersection> &intersections,
            Intersection &newIts,
            const Intersection &its_out, const Vector &d_out,
            int channel, Sampler *sampler) const {
        size_t i = m_weights.sample(sampler->next1D());
        Float thePdf = m_weights[i] * m_samplers[i]->sample(
                intersections, newIts, its_out, d_out, channel, sampler);
        Assert(std::isfinite(thePdf));
        if (thePdf == 0)
            return 0;
        for (size_t j = 0; j < m_weights.size(); j++) {
            if (j == i)
                continue;
            Float thisPdf = m_samplers[j]->pdf(
                    intersections, newIts, its_out, d_out, channel);
            Assert(thisPdf >= 0);
            thePdf += m_weights[j] * thisPdf;
            Assert(std::isfinite(thePdf));
        }
        return thePdf;
    }

    Float pdf(const std::vector<Intersection> &intersections,
            const Intersection &newIts, const Intersection &its_out,
            const Vector &d_out, int channel) const {
        Float thePdf = 0;
        for (size_t j = 0; j < m_weights.size(); j++) {
            Float thisPdf = m_samplers[j]->pdf(
                    intersections, newIts, its_out, d_out, channel);
            Assert(thisPdf >= 0);
            thePdf += m_weights[j] * thisPdf;
            Assert(std::isfinite(thePdf));
        }
        return thePdf;
    }

protected:
    virtual ~MISIntersectionSampler() { }
    DiscreteDistribution m_weights;
    ref_vector<const IntersectionSampler> m_samplers;
    MTS_DECLARE_CLASS();
};

/// weight = f(its_in, its_out, d_out, spectralChannel)
typedef std::function<Float(const Intersection&, const Intersection&, const Vector&, int)> IntersectionWeightFunc;

// channel can be -1 (= MIS over all channels)
class MTS_EXPORT_RENDER WeightIntersectionSampler : public IntersectionSampler {
public:
    WeightIntersectionSampler(IntersectionWeightFunc intersectionWeight,
            Float itsDistanceCutoff) :
                IntersectionSampler(itsDistanceCutoff),
                m_intersectionWeight(intersectionWeight) { }
    virtual Float sample(const std::vector<Intersection> &intersections,
            Intersection &newIts,
            const Intersection &its_out, const Vector &d_out,
            int channel, Sampler *sampler) const ;
    virtual Float pdf(const std::vector<Intersection> &intersections,
            const Intersection &newIts,
            const Intersection &its_out, const Vector &d_out,
            int channel) const ;
    virtual ~WeightIntersectionSampler() { }

protected:
    MTS_DECLARE_CLASS();
    const IntersectionWeightFunc m_intersectionWeight;
};

/// weight = f(distance, spectralChannel), channel is valid (i.e. not -1)
typedef std::function<Float(Float,int)> DistanceWeightFunc;

inline IntersectionWeightFunc distanceWeightWrapper(DistanceWeightFunc f) {
    return [=] (const Intersection &its_in, const Intersection &its_out,
                const Vector &d_out, int i) {
        return f(distance(its_in.p, its_out.p), i);
    };
}

inline DistanceWeightFunc makeConstantDistanceWeight() {
    return [=] (Float d, int i) { return 1.0f; };
}

inline DistanceWeightFunc makeExponentialDistanceWeight(
        const Spectrum &sigmaTr) {
    return [=] (Float d, int i) { return math::fastexp(-sigmaTr[i] * d); };
}

DistanceWeightFunc makeExactDiffusionDipoleDistanceWeight(
        const Spectrum &sigmaA, const Spectrum &sigmaS,
        const Spectrum &g, Float eta);

/**
 * \brief Sampler for a (spectrally-dependent) 1D distribution
 */
class MTS_EXPORT_RENDER Sampler1D : public Object {
public:
    virtual bool sample(int channel, Float &x,
            Sampler *sampler, Float *pdf = NULL) const = 0;
    virtual Float pdf(int channel, Float x) const = 0;

    /// Accepts channel=-1 for MIS sampling over channels
    bool sampleMIS(int channel, Float &x,
            Sampler *sampler, Float *pdf = NULL) const {
        if (channel != -1)
            return sample(channel, x, sampler, pdf);
        Float chosenChannel = sampler->next1D() * SPECTRUM_SAMPLES;
        if (!sample(chosenChannel, x, sampler, NULL))
            return false;
        if (pdf)
            *pdf = pdfMIS(channel, x);
        return true;
    }

    /// Accepts channel=-1 for MIS sampling over channels
    bool pdfMIS(int channel, Float x) const {
        return channelMean(channel,
                [=] (int chan) { return pdf(chan, x); });
    }

    MTS_DECLARE_CLASS();
protected:
    virtual ~Sampler1D() { }
};

/**
 * \brief Sampler for a (spectrally-dependent) 2D distribution in some
 * (tangent) plane in an intersection point.
 *
 * This usually gets used within a \c ProjSurfaceSampler, where its
 * associated \c DSSProjFrame will determine the nature of the tangent
 * frame (and give meaning to the coordinates of the 2D vector \c x).
 */
class MTS_EXPORT_RENDER TangentSampler2D : public Object {
public:
    /**
     * \brief Sample a point x in the 2D plane.
     *
     * \param x The point to sample in the (tangent) plane, with respect to
     *          the intersection point. When this \c TangentSampler2D is
     *          used within a \c ProjSurfaceSampler, then meaning of the
     *          coordinates in the plane is determined by the
     *          \c DSSProjFrame that is associated with the
     *          \c ProjSurfaceSampler.
     * \param cosTheta The cosine between the query direction and the
     *                 plane normal. Taken to be positive.
     */
    virtual bool sample(int channel, Vector2 &x, Float cosTheta,
            const Vector2 &xLo, const Vector2 &xHi,
            Sampler *sampler, Float *pdf = NULL) const = 0;
    virtual Float pdf(int channel, Vector2 x, Float cosTheta,
            const Vector2 &xLo, const Vector2 &xHi) const = 0;
    MTS_DECLARE_CLASS();
protected:
    virtual ~TangentSampler2D() { }
};

class MTS_EXPORT_RENDER MISTangentSampler2D : public TangentSampler2D {
public:
    MISTangentSampler2D(const std::vector<std::pair<Float, const TangentSampler2D*> > &samplers) {
        if (samplers.size() < 1)
            Log(EError, "Trying to construct MISTangentSampler2D without "
                    "any samplers!");
        m_samplers.reserve(samplers.size());
        m_weights.reserve(samplers.size());
        for (auto p : samplers) {
            if (p.first <= 0)
                continue;
            m_weights.append(p.first);
            m_samplers.push_back(p.second);
        }
        if (m_samplers.size() < 1)
            Log(EError, "Trying to construct MISTangentSampler2D without "
                    "any strictly positive weight samplers!");
        m_weights.normalize();
    }

    virtual bool sample(int channel, Vector2 &x, Float cosTheta,
            const Vector2 &xLo, const Vector2 &xHi,
            Sampler *sampler, Float *thePdf = NULL) const {
        size_t i = m_weights.sample(sampler->next1D());
        if (!m_samplers[i]->sample(
                channel, x, cosTheta, xLo, xHi, sampler, NULL))
            return false;
        if (thePdf)
            *thePdf = pdf(channel, x, cosTheta, xLo, xHi);
        return true;
    }

    virtual Float pdf(int channel, Vector2 x, Float cosTheta,
            const Vector2 &xLo, const Vector2 &xHi) const {
        Float thePdf = 0;
        for (size_t i = 0; i < m_weights.size(); i++)
            thePdf += m_weights[i] * m_samplers[i]->pdf(
                    channel, x, cosTheta, xLo, xHi);
        return thePdf;
    }

    MTS_DECLARE_CLASS();

protected:
    virtual ~MISTangentSampler2D() { }
    DiscreteDistribution m_weights;
    ref_vector<const TangentSampler2D> m_samplers;
};



class MTS_EXPORT_RENDER RadialSampler2D : public TangentSampler2D {
public:
    RadialSampler2D(const Sampler1D* radialSampler)
            : m_radialSampler(radialSampler) { }
    inline virtual bool sample(int channel, Vector2 &x, Float cosTheta,
            const Vector2 &xLo, const Vector2 &xHi,
            Sampler *sampler, Float *pdf = NULL) const {
        Float r;
        if (!m_radialSampler->sample(channel, r, sampler, pdf))
            return false;

        Assert(std::isfinite(r) && r >= 0);
        Float s, c;
        math::sincos(TWO_PI * sampler->next1D(), &s, &c);
        x[0] = r*s;
        x[1] = r*c;
        return true;
    }

    inline virtual Float pdf(int channel, Vector2 x, Float cosTheta,
            const Vector2 &xLo, const Vector2 &xHi) const {
        return m_radialSampler->pdf(channel, x.length());
    }

    MTS_DECLARE_CLASS();
protected:
    virtual ~RadialSampler2D() { }

    /// normalized for full area measure in the 2D plane
    const ref<const Sampler1D> m_radialSampler;
};



/**
 * \brief Exponential distribution.
 */
class MTS_EXPORT_RENDER ExpSampler1D : public Sampler1D {
public:
    ExpSampler1D(const Spectrum &lambda) : m_lambda(lambda) { }
    virtual bool sample(int channel, Float &x, Sampler *sampler,
            Float *thePdf = NULL) const {
        if (m_lambda[channel] == 0)
            return false;
        Float u = sampler->next1D();
        x = -log(u) / m_lambda[channel];
        if (thePdf)
            *thePdf = pdf(channel, x);
        return true;
    }
    virtual Float pdf(int channel, Float x) const {
        return m_lambda[channel] * math::fastexp(-m_lambda[channel] * x);
    }

    MTS_DECLARE_CLASS();
protected:
    virtual ~ExpSampler1D() { }
    const Spectrum m_lambda;
};


/**
 * \brief Exactly sampling the classic dipole contribution in the tangent plane.
 *
 * Uses the sampling method from Mertens et al, 'Efficient Rendering of
 * Local Subsurface Scattering' (2005).
 */
class MTS_EXPORT_RENDER RadialExactDipoleSampler2D : public Sampler1D {
public:
    RadialExactDipoleSampler2D(const Spectrum &sigmaA,
            const Spectrum &sigmaS, const Spectrum &g, Float eta);
    virtual bool sample(int channel, Float &r, Sampler *sampler,
            Float *thePdf = NULL) const;
    virtual Float pdf(int channel, Float r) const;

    MTS_DECLARE_CLASS();
protected:
    Spectrum m_sigmaTr;
    Spectrum m_zr;
    Spectrum m_zv;
    Spectrum m_pdfNorm;
    Spectrum m_T; /// real vs virtual prob threshold
};





class MTS_EXPORT_RENDER SurfaceSampler : public Object {
public:
    // channel can be -1
    virtual Float sample(const Intersection &its,
            const Vector &d_out, const Scene *scene,
            const std::vector<Shape *> &shapes,
            Intersection &newIts, int channel,
            Sampler *sampler) const = 0;
    // channel can be -1
    virtual Float pdf(const Intersection &its,
            const Vector &d_out, const Scene *scene,
            const std::vector<Shape *> &shapes,
            const Intersection &newIts, int channel) const = 0;

    MTS_DECLARE_CLASS();
protected:
    /// Virtual destructor
    virtual ~SurfaceSampler() { };
};

class MTS_EXPORT_RENDER UniformSurfaceSampler : public SurfaceSampler {
public:
    UniformSurfaceSampler() { }
    virtual Float sample(const Intersection &its,
            const Vector &d_out, const Scene *scene,
            const std::vector<Shape *> &shapes,
            Intersection &newIts, int channel,
            Sampler *sampler) const;
    virtual Float pdf(const Intersection &its,
            const Vector &d_out, const Scene *scene,
            const std::vector<Shape *> &shapes,
            const Intersection &newIts, int channel) const;

    MTS_DECLARE_CLASS();
protected:
    /// Virtual destructor
    virtual ~UniformSurfaceSampler() { };
};

/** \brief Sets the incoming point to the outgoing point, as in the case of
 * a BRDF. */
class MTS_EXPORT_RENDER BRDFDeltaSurfaceSampler : public SurfaceSampler {
public:
    BRDFDeltaSurfaceSampler() { }
    virtual Float sample(const Intersection &its,
            const Vector &d_out, const Scene *scene,
            const std::vector<Shape *> &shapes,
            Intersection &newIts, int channel,
            Sampler *sampler) const;
    virtual Float pdf(const Intersection &its,
            const Vector &d_out, const Scene *scene,
            const std::vector<Shape *> &shapes,
            const Intersection &newIts, int channel) const;

    MTS_DECLARE_CLASS();
protected:
    /// Virtual destructor
    virtual ~BRDFDeltaSurfaceSampler() { };
};

/**
 * \brief SurfaceSampler that works by sampling a point in a 2D plane and
 * then projecting orthogonally to the surface. */
class MTS_EXPORT_RENDER ProjSurfaceSampler : public SurfaceSampler {
public:
    ProjSurfaceSampler(DSSProjFrame projFrame,
            const TangentSampler2D *planeSampler,
            const IntersectionSampler *itsSampler) :
        m_planeSampler(planeSampler), m_itsSampler(itsSampler),
        m_projFrame(projFrame) { }

    virtual Float sample(const Intersection &its,
            const Vector &d_out, const Scene *scene,
            const std::vector<Shape *> &shapes,
            Intersection &newIts, int channel,
            Sampler *sampler) const;

    virtual Float pdf(const Intersection &its,
            const Vector &d_out, const Scene *scene,
            const std::vector<Shape *> &shapes,
            const Intersection &newIts, int channel) const;

    MTS_DECLARE_CLASS();
protected:
    /// Virtual destructor
    virtual ~ProjSurfaceSampler() { };

    /// Get the projection frame
    void getProjFrame(Vector &u, Vector &v, Vector &projectionDir,
            const Intersection &its, const Vector &d_out) const;

    /// normalized for full area measure in the 2D plane
    const ref<const TangentSampler2D> m_planeSampler;

    const ref<const IntersectionSampler> m_itsSampler;

    const DSSProjFrame m_projFrame;
};


/**
 * \brief Subsurface scattering materials that directly sample the incoming
 * radiance.
 *
 * This interface provides a helpful framework to write SubSurface
 * Scattering (SSS) models that directly sample the incoming radiance on
 * the surface --- as opposed to working with an (ir)radiance cache.
 *
 * There are some helper functions to handle boundary conditions either
 * 'explicitly' or 'implicitly', see the documentation for \c bssrdf(),
 * \c Li() and \c handleImplicitBounds().
 * */
class MTS_EXPORT_RENDER DirectSamplingSubsurface : public Subsurface {
public:
    /** \brief This default implementation fills in the internal list of
     * radiance sources and normalizes the MIS weights. */
    virtual bool preprocess(const Scene *scene, RenderQueue *queue,
        const RenderJob *job, int sceneResID, int cameraResID,
        int samplerResID);

    /**
     * \brief Sample point on the surface where the incoming radiance
     * should be sampled.
     *
     * Given the query point at the provided intersection, return a new
     * intersection where the incoming radiance should be sampled.
     *
     * Ideally, this should sample the bssrdf, marginalized over incoming
     * directions and extra parameters.
     *
     * \param d_out The direction (pointing outwards) in which the outgoing
     * radiance will be calculated.
     *
     * \return The pdf in area measure.
     */
    virtual Float samplePointOnSurface(const Intersection &its,
            const Vector &d_out, const Scene *scene,
            Intersection &newIts, const Spectrum &throughput,
            Sampler *sampler) const {
        if (!m_allowIncomingOutgoingDirections)
            Assert(dot(d_out, its.shFrame.n) >= 0);
        // One sample MIS weighting (balance heuristic)
        int channel = throughputToChannel(throughput);
        Assert(m_weights.size() > 0);
        Assert(m_weights.isNormalized());
        size_t chosenSamplerIdx = m_weights.sample(sampler->next1D());
        Float p = m_weights[chosenSamplerIdx] *
                m_surfaceSamplers[chosenSamplerIdx]->sample(
                    its, d_out, scene, getShapes(), newIts, channel, sampler);
        if (p == 0)
            return 0.0f;

        for (size_t i = 0; i < m_surfaceSamplers.size(); i++) {
            if (i == chosenSamplerIdx)
                continue;
            p += m_weights[i] * m_surfaceSamplers[i]->pdf(
                its, d_out, scene, getShapes(), newIts, channel);
        }
        Assert(std::isfinite(p));
        return p;
    }

    virtual Float pdfPointOnSurface(const Intersection &its,
            const Vector &d_out, const Scene *scene,
            const Intersection &newIts, const Spectrum &throughput) const {
        if (!m_allowIncomingOutgoingDirections)
            Assert(dot(d_out, its.shFrame.n) >= 0);
        // One sample MIS weighting (balance heuristic)
        int channel = throughputToChannel(throughput);
        Assert(m_weights.size() > 0);
        Assert(m_weights.isNormalized());
        Float p = 0;
        for (size_t i = 0; i < m_surfaceSamplers.size(); i++) {
            p += m_weights[i] * m_surfaceSamplers[i]->pdf(
                its, d_out, scene, getShapes(), newIts, channel);
        }
        Assert(std::isfinite(p));
        return p;
    }

     /**
      * \brief The BSSRDF transport kernel between the given points.
      *
      * The given directions follow the flow of light, i.e. \c d_in points
      * into the medium and \c d_out points outwards.
      *
      * Mismatched boundaries:
      * ======================
      *
      * Case m_eta == 1:
      * ----------------
      * The boundary should be explicitly given through an appropriate
      * BSDF. The BSSRDF should be computed as for an index-matched
      * boundary.
      *
      * Case m_eta != 1:
      * ----------------
      * In this case, the BSDF *must* be of an index-matched or null type,
      * or the results will be nonsensical.
      * The directions that are provided are those 'outside' of the medium
      * (i.e. incoming before refraction and outgoing after refraction).
      * See the \c handleImplicitBounds helper function to handle the
      * directional changes due to refraction and the Fresnel transmittance
      * factor scaling.
      * The BSSRDF should internally compensate for the effect of the
      * internal Fresnel reflection on the boundary condition.
      *
      * \return The value of the BSSRDF.
      */
    virtual Spectrum bssrdf(const Scene *scene,
            const Point &p_in,  const Vector &d_in,  const Normal &n_in,
            const Point &p_out, const Vector &d_out, const Normal &n_out,
            const void *extraParams) const = 0;

    /**
     * Returns the number of bytes that is requested for the extra parameters.
     */
    virtual size_t extraParamsSize() const = 0;

    /**
     * \brief Sample optional extra params that are needed to evaluate the
     * BSSRDF.
     *
     * These extra parameters are 'sampled after the incoming point and
     * before the incoming direction', the only exception being when the
     * bssrdf is evaluated for a known \c RadianceSource (in which case the
     * incoming direction is also known in advance).
     *
     * The allocated space is (at least) \c extraParamsSize() bytes.
     *
     * \param d_in If \c d_in is \c NULL, then \c d_in is not (yet) known,
     * and the ideal case is to sample the BSSRDF marginalized over all
     * possible \c d_in. If the specific \c d_in is known, then it is
     * provided here.
     *
     * \return The pdf.
     */
    virtual Spectrum sampleExtraParams(const Scene *scene,
            const Intersection &its_out, const Vector &d_out,
            const Intersection &its_in,  const Vector *d_in,
            const Spectrum &throughput, void *extraParams,
            Sampler *sampler) const = 0;

    /// Returns the pdf for sampleExtraParams.
    virtual Spectrum pdfExtraParams(const Scene *scene,
            const Intersection &its_out, const Vector &d_out,
            const Intersection &its_in,  const Vector *d_in,
            const Spectrum &throughput, const void *extraParams) const = 0;

    /**
     * \brief Sample an incoming direction and return the pdf (in solid
     * angle measure).
     *
     * The default implementation simply samples the cosine weighted
     * incoming hemisphere.
     *
     * \return The pdf on the (regular, non-projected/non-cosine-weighted)
     * solid angle hemisphere. (I.e. the default cosine weighted sampling
     * explicitly has the cosine factor in the pdf.)
     */
    virtual Spectrum sampleBssrdfDirection(const Scene *scene,
            const Intersection &its_out, const Vector &d_out,
            Intersection &its_in,        Vector       &d_in,
            const void *extraParams, const Spectrum &throughput,
            Sampler *sampler) const = 0;

    /**
     * \brief Return the pdf on \c d_in that is used in \c
     * sampleBssrdfDirection().
     */
    virtual Spectrum pdfBssrdfDirection(const Scene *scene,
            const Intersection &its_out, const Vector &d_out,
            const Intersection &its_in,  const Vector &d_in,
            const void *extraParams, const Spectrum &throughput) const = 0;




    /**
     * \brief Sample the exitant radiance for a point on the surface.
     *
     * Currently not implemented (yet). [You should probably use Li() anyway.]
     */
    virtual Spectrum Lo(const Scene *scene, Sampler *sampler,
            const Intersection &its, const Vector &d_out,
            const Spectrum &throughput, int depth = 0) const;

    /**
     * \brief Analogous to Lo but for the incident radiance on the boundary
     * --- can be coupled to a real BSDF to explicitly take the boundaries
     *  into account.
     *
     * An 'index matched' (\c m_eta = 1) DirectSamplingSubsurface that uses
     * this Li() method in combination with a correct treatment of the BSDF
     * of the boundaries should be preferred above the usage of Lo().
     *
     * A more typical 'Lo' behaviour (where the boundaries are taken into
     * account implicitly) can be obtained by using a 'null' BSDF and
     * setting the \c m_eta value of this DirectSamplingSubsurface to the
     * appropriate (non-unity) value */
    Spectrum Li(const Scene *scene, Sampler *sampler,
            const Intersection &its, const Vector &d,
            const Spectrum &throughput, int &splits, int depth) const {
        return Li_internal(
                scene, sampler, its, d, throughput, splits, depth, 0);
    }

    bool supportsLi() const {
        return true;
    }

    virtual void bindUsedResources(ParallelProcess *proc) const;
    virtual void wakeup(ConfigurableObject *parent,
            std::map<std::string, SerializableObject *> &params);

    virtual void serialize(Stream *stream, InstanceManager *manager) const;

    MTS_DECLARE_CLASS();

protected:

    DirectSamplingSubsurface(const Properties &props);

    DirectSamplingSubsurface(Stream *stream, InstanceManager *manager);

    /// Virtual destructor
    virtual ~DirectSamplingSubsurface();

    Spectrum Li_internal(const Scene *scene, Sampler *sampler,
            const Intersection &its, const Vector &d,
            const Spectrum &throughput, int &splits, int depth,
            int numInternalRefl) const;

    /**
     * \brief Should be called by derived classes during the configure()
     * phase.
     *
     * Don't forget to call normalizeSamplers() afterwards! */
    void registerSampler(Float weight,
            const SurfaceSampler *surfaceSampler) {
        Assert(weight >= 0);
        if (weight == 0)
            return;
        m_surfaceSamplers.push_back(surfaceSampler);
        m_weights.append(weight);
    }

    /// To be called after registering all samplers.
    void normalizeSamplers() {
        Assert(m_weights.size() > 0);
        m_weights.normalize();
    }



    /**
     * Returns the channel if the throughput is already single-channel,
     * otherwise returns -1 */
    static inline int throughputToChannel(const Spectrum &throughput) {
        int channel = -1;
        if (throughput.numNonZeroChannels(&channel) != 1)
            return -1;
        return channel;
    }

    /**
     * Force single channel throughput if requested, returns the
     * appropriate weight */
    inline Spectrum handleSingleChannel(
            const Spectrum &throughput, Sampler *sampler) const {
        if (!m_singleChannel)
            return Spectrum(1.0f);

        Float channelWeight = 1;
        int channel = throughputToChannel(throughput);
        if (channel == -1) {
            channelWeight = throughput.numNonZeroChannels();
            channel = throughput.sampleNonZeroChannelUniform(sampler);
        }
        Spectrum result(0.0f);
        result[channel] = channelWeight;
        return result;
    }



    /**
     * Sample the \c d_in and \c rec_wi directions based on the BSSRDF.
     *
     * This is used for MIS weighting in combination with \c
     * sampleDirectionsDirect().
     *
     * \param pdf_d_in_and_rec_wi Pdf on internal direction d_in and final
     * recursive wi query direction (after having interacted with the
     * boundary) rec_wi. Both are taken to be in the regular hemispherical
     * domain (or partly in a discrete domain for e.g. null or perfectly
     * specular BSDFs).
     *
     * \param rec_wi Direction for recursive wi query, in world coordinates.
     *
     * \return The value of the BSDF, including both cosine factors (or
     * only the one relevant factor if the accompagnying BSDF is not
     * smooth).
     */

    Spectrum sampleDirectionsFromBssrdf(
            const Scene *scene,
            const Intersection &its_out, const Vector &d_out,
            Intersection       &its_in,  Vector       &d_in,
            Vector &rec_wi, EMeasure &bsdfMeasure,
            Spectrum &pdf_d_in_and_rec_wi,
            const Spectrum &throughput,
            const void *extraParams,
            Sampler *sampler) const;

    /// pdf of \c sampleDirectionsFromBssrdf()
    Spectrum pdfDirectionsFromBssrdf(
            const Scene *scene,
            const Intersection &its_out, const Vector &d_out,
            const Intersection &its_in,  const Vector &d_in,
            const Vector &rec_wi, EMeasure bsdfMeasure,
            const Spectrum &throughput,
            const void *extraParams) const;

    /**
     * Sample the \c d_in and \c rec_wi directions based on the emitters.
     *
     * This is used for MIS weighting in combination with \c
     * sampleDirectionsFromBssrdf().
     *
     * \return The value of the BSDF, including both cosine factors (or
     * only the one relevant factor if the accompagnying BSDF is not
     * smooth).
     */
    Spectrum sampleDirectionsDirect(
            const Scene *scene,
            const Intersection &its_out, const Vector &d_out,
            Intersection       &its_in,  Vector       &d_in,
            Vector &rec_wi, EMeasure &bsdfMeasure,
            EMeasure &lightSamplingMeasure,
            Spectrum &pdf_d_in_and_rec_wi,
            Spectrum &LiDirect,
            Sampler *sampler) const;

    /// pdf of \c sampleDirectionsFromBssrdf()
    Spectrum pdfDirectionsDirect(
            const Scene *scene,
            const Intersection &its_out, const Vector &d_out,
            const Intersection &its_in,  const Vector &d_in,
            const Vector &rec_wi, EMeasure bsdfMeasure) const;

    /**
     * \brief Samples the incoming directions and the extra parameters, for a
     * given incoming point \c its_in, by directly sampling a direction towards
     * a light source.
     *
     * \return pdf(d_in, rec_wi, extraParams)
     */
    Spectrum sampleDirect(const Scene *scene, Sampler *sampler,
            Intersection &its_in, const Intersection &its_out,
            const Vector &d_out,
            const Spectrum &effectiveThroughput,
            Vector &d_in, Vector &rec_wi, void *extraParams,
            EMeasure &bsdfMeasure, EMeasure &lightSamplingMeasure,
            Spectrum &bsdfVal, Spectrum &LiDirect) const;

    /// Returns pdf(d_in, rec_wi, extraParams)
    Spectrum pdfDirect(const Scene *scene,
            const Intersection &its_in, const Intersection &its_out,
            const Vector &d_out,
            const Spectrum &effectiveThroughput,
            const Vector &d_in, const Vector &rec_wi, const void *extraParams,
            EMeasure bsdfMeasure) const;

    /**
     * \brief Samples the incoming directions and the extra parameters, for a
     * given incoming point \c its_in, by importance sampling the angular
     * BSSRDF transport.
     *
     * \return pdf(d_in, rec_wi, extraParams)
     */
    Spectrum sampleIndirect(const Scene *scene, Sampler *sampler,
            Intersection &its_in, const Intersection &its_out,
            const Vector &d_out, const Spectrum &effectiveThroughput,
            Vector &d_in, Vector &rec_wi, void *extraParams,
            EMeasure &bsdfMeasure, Spectrum &bsdfVal) const;

    /// Returns pdf(d_in, rec_wi, extraParams)
    Spectrum pdfIndirect(const Scene *scene,
            const Intersection &its_in, const Intersection &its_out,
            const Vector &d_out, const Spectrum &effectiveThroughput,
            const Vector &d_in, const Vector &rec_wi, const void *extraParams,
            EMeasure bsdfMeasure) const;




    /**
     * Holds information about the parameters (incoming point, incoming
     * directions [before & after interacting with the boundary BSDF], and
     * extra parameters) that were sampled from 'indirect' sampling, i.e.
     * from sampling the BSSRDF itself (as opposed to a 'direct' sampling
     * of the light sources). */
    struct IndirectSamplingRecord {
        Intersection its_in;
        Vector d_in; // pointing inwards, on our side of medium
        Vector rec_wi;
        EMeasure bsdfMeasure; // NOTE: only here for checkSourcesOfVariance...

        /* Sample weight for direct portion of the 'indirectly sampled' Li
         * (handles optional MIS, e.g. for 2-sample MIS weighting with
         * direct+indirect sampling methods) */
        Spectrum weightForDirectContrib;

        /* Sample weight for indirect portion of the 'indirectly sampled'
         * Li, or the full 'indirectly sampled' Li, if dedicated
         * directSampling isn't enabled */
        Spectrum weightForIndirectContrib;
    };

    /**
     * Sample Importance Resampling (SIR) of direct&indirect MIS weighted
     * sampling (with full expected value estimator for the direct
     * contribution) */
    bool indirectSample_SIR(const Scene *scene, Sampler *sampler,
            const Intersection &its_out, const Vector &d,
            const Spectrum &channelWeight,
            const Spectrum &channelWeightedThroughput,
            Spectrum &LiContribution,
            IndirectSamplingRecord &indirectSample, void * extraParams) const;

    bool indirectSample_noSIR(const Scene *scene, Sampler *sampler,
            const Intersection &its_out, const Vector &d,
            const Spectrum &channelWeight,
            const Spectrum &channelWeightedThroughput,
            Spectrum &LiContribution,
            IndirectSamplingRecord &indirectSample, void * extraParams) const;



    /**
     * \brief Helper function that modifies directions for implicit bound
     * refraction.
     *
     * If eta != 1, then we are in a mode where boundaries are implicitly
     * taken into account (In this case: the explicit BSDF should be
     * index-matched or null!). Because Fresnel effects (transmission
     * scaling and refraction) are essentially shared by all BSSRDF models,
     * we provide a helper function at this level.
     *
     * \return The combined Fresnel transmitance of the incoming and
     * outgoing refractions.
     */
    Float handleImplicitBounds(
            Vector &d_in,  const Normal &n_in,
            Vector &d_out, const Normal &n_out) const;

    /**
     * \brief Debugging function that gives information on which sampling
     * steps are the largest source of variance.
     */
    void checkSourcesOfVariance(
            const Scene *scene, const Spectrum &throughput,
            Intersection its_out, Vector d_out,
            Intersection check_its_in,
            Vector check_d_in,
            Vector check_rec_wi,
            const void *check_extraParams,
            Spectrum check_bssrdfVal,
            Spectrum check_bsdfVal,
            EMeasure check_bsdfMeasure,
            Sampler *sampler,
            bool absify) const;



    Float m_eta; /// intIOR/extIOR
    size_t m_numSIR;
    ref<RadianceSources> m_sources;
    int m_sourcesIndex;
    int m_sourcesResID;
    bool m_directSampling;
    bool m_directSamplingMIS;
    bool m_singleChannel;
    bool m_allowIncomingOutgoingDirections;
    bool m_nonCollimatedLightSourcesPresent;
    int m_maxInternalReflections; /// Maximum number of subsequent internal reflections (<0 for unbounded)
    ref_vector<const SurfaceSampler> m_surfaceSamplers;
    DiscreteDistribution m_weights;
    /* itsDistanceCutoff is not actually used at this level, but it's added
     * here for convenience, because all subclasses will pretty much need
     * this for their IntersectionSamplers. */
    Float m_itsDistanceCutoff;
};

MTS_NAMESPACE_END

#endif /* __MITSUBA_RENDER_DSS_H_ */
