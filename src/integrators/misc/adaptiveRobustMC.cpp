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
#include <mitsuba/core/util.h>
#include <boost/math/distributions/normal.hpp>

MTS_NAMESPACE_BEGIN

/*!\plugin{adaptiveRobustMC}{Robust adaptive Monte Carlo integrator [CURRENTLY ONLY FOR BOX FILTER]}
 * \order{13}
 * \parameters{
 *     \parameter{maxError}{\Float}{Maximum relative error
 *         threshold\default{0.05}}
 *     \parameter{pValue}{\Float}{
 *         Required p-value to accept a sample \default{0.05}
 *     }
 *     \parameter{maxSampleFactor}{\Integer}{
 *         Maximum number of samples to be generated \emph{relative} to the
 *         number of configured pixel samples. The adaptive integrator
 *         will stop after this many samples, regardless of whether
 *         or not the error criterion was satisfied.
 *         A negative value will be interpreted as $\infty$.
 *         \default{32---for instance, when 64 pixel samples are configured in
 *         the \code{sampler}, this means that the adaptive integrator
 *         will give up after 32*64=2048 samples}
 *     }
 *     \parameter{numBatches}{\Integer}{
 *         The number of sub-estimates to compute. The final result is the 
 *         average of these sub-estimate, but excluding the lowest and 
 *         highest sub-estimate. Note: the sample count given in the 
 *         sub-integrator is for each such sub-estimate. \default{10}
 *     }
 * }
 *
 * This ``meta-integrator'' repeatedly invokes a provided sub-integrator
 * until the computed radiance values satisfy a specified relative error bound
 * (5% by default) with a certain probability (95% by default). Internally,
 * it uses a Z-test to decide when to stop collecting samples. While repeatedly
 * applying a Z-test in this manner is not good practice in terms of
 * a rigorous statistical analysis, it provides a useful mathematically
 * motivated stopping criterion.
 *
 * \begin{xml}[caption={An example how to make the \pluginref{path} integrator adaptive}]
 * <integrator type="adaptiveRobustMC">
 *     <integrator type="path"/>
 * </integrator>
 * \end{xml}
 *
 * \remarks{
 *    \item This integrator currently only works with the box reconstruction filter!
 *    \item The adaptive integrator needs a variance estimate to work
 *     correctly. Hence, the underlying sample generator should be set to a reasonably
 *     large number of pixel samples (e.g. 64 or higher) so that this estimate can be obtained.
 *    \item This plugin uses a relatively simplistic error heuristic that does not
 *    share information between pixels and only reasons about variance in image space.
 *    In the future, it will likely be replaced with something more robust.
 * }
 */
class AdaptiveIntegratorRobustMC : public MonteCarloIntegrator {
public:
    AdaptiveIntegratorRobustMC(const Properties &props) : MonteCarloIntegrator(props) {
        /* Maximum relative error threshold. */
        m_maxError = props.getFloat("maxError", 0.05f);
        if (m_maxError < 0)
            Log(EError, "Received a negative maxError!");
        /* Maximum number of samples to take (relative to the number of pixel samples
           that were configured in the sampler). The sample collection
           will stop after this many samples even if the variance is still
           too high. A negative value will be interpreted as infinity. */
        m_maxSampleFactor = props.getInteger("maxSampleFactor", 32);
        /* Required P-value to accept a sample. */
        m_pValue = props.getFloat("pValue", 0.05f);
        /* Number of batches for the samples. The batch with the highest 
         * and lowest mean value will be ignored in the final averaged 
         * result. */
        m_numBatches = props.getSize("numBatches", 10);
        if (m_numBatches < 4)
            Log(EError, "Need at least 4 batches, but got " SIZE_T_FMT, m_numBatches);
        m_verbose = props.getBoolean("verbose", false);
    }

    AdaptiveIntegratorRobustMC(Stream *stream, InstanceManager *manager)
     : MonteCarloIntegrator(stream, manager) {
        m_subIntegrator = static_cast<MonteCarloIntegrator *>(manager->getInstance(stream));
        m_maxSampleFactor = stream->readInt();
        m_maxError = stream->readFloat();
        m_quantile = stream->readFloat();
        m_averageLuminance = stream->readFloat();
        m_pValue = stream->readFloat();
        m_numBatches = stream->readSize();
        m_verbose = false;
    }

    void addChild(const std::string &name, ConfigurableObject *child) {
        const Class *cClass = child->getClass();

        if (cClass->derivesFrom(MTS_CLASS(Integrator))) {
            if (!cClass->derivesFrom(MTS_CLASS(MonteCarloIntegrator)))
                Log(EError, "The sub-integrator must be derived from the class MonteCarloIntegrator");
            m_subIntegrator = static_cast<MonteCarloIntegrator *>(child);
            Assert(m_subIntegrator);
        } else {
            Integrator::addChild(name, child);
        }
    }

    void configureSampler(const Scene *scene, Sampler *sampler) {
        MonteCarloIntegrator::configureSampler(scene, sampler);
        if (!m_subIntegrator)
            Log(EError, "No sub-integrator was specified!");
        m_subIntegrator->configureSampler(scene, sampler);
    }

    bool preprocess(const Scene *scene, RenderQueue *queue, const RenderJob *job,
            int sceneResID, int sensorResID, int samplerResID) {
        Assert(m_subIntegrator);
        if (!MonteCarloIntegrator::preprocess(scene, queue, job, sceneResID, sensorResID, samplerResID))
            return false;
        Sampler *sampler = static_cast<Sampler *>(Scheduler::getInstance()->getResource(samplerResID, 0));
        Sensor *sensor = static_cast<Sensor *>(Scheduler::getInstance()->getResource(sensorResID));
        if (sampler->getClass()->getName() != "IndependentSampler")
            Log(EError, "The error-controlling integrator should only be "
                "used in conjunction with the independent sampler");

        if (sensor->getFilm()->getReconstructionFilter()->getClass()->getName() != "BoxFilter") {
            Log(EError, "The robust adaptive integrator currently only "
                    "supports a box reconstruction filter!");
        }

        if (!m_subIntegrator->preprocess(scene, queue, job, sceneResID, sensorResID, samplerResID))
            return false;

        // Copy settings from subintegrator
        m_maxDepth = m_subIntegrator->getMaxDepth();
        m_rr = m_subIntegrator->getRR();
        m_strictNormals = m_subIntegrator->strictNormals();
        m_hideEmitters = m_subIntegrator->hideEmitters();


        Vector2i filmSize = sensor->getFilm()->getSize();
        bool needsApertureSample = sensor->needsApertureSample();
        bool needsTimeSample = sensor->needsTimeSample();
        const int nSamples = 10000;

        Point2 apertureSample(0.5f);
        Float timeSample = 0.5f;
        RadianceQueryRecord rRec(scene, sampler);

        /* Estimate the overall luminance on the image plane */
        Log (EInfo, "Estimating the overall luminance on the image plane...");
        ProgressReporter rep("Estimating luminance", nSamples, NULL);
        VarianceFunctor luminance;
        for (int i=0; i<nSamples; ++i) {
            sampler->generate(Point2i(0));

            rRec.newQuery(RadianceQueryRecord::ERadiance, sensor->getMedium());
            rRec.extra = RadianceQueryRecord::EAdaptiveQuery;

            Point2 samplePos(rRec.nextSample2D());
            samplePos.x *= filmSize.x;
            samplePos.y *= filmSize.y;

            if (needsApertureSample)
                apertureSample = rRec.nextSample2D();
            if (needsTimeSample)
                timeSample = rRec.nextSample1D();

            RayDifferential eyeRay;
            Spectrum sampleValue = sensor->sampleRay(
                eyeRay, samplePos, apertureSample, timeSample);

            sampleValue *= m_subIntegrator->Li(eyeRay, rRec);
            luminance.update(sampleValue.getLuminance());
            rep.update(i);
        }

        m_averageLuminance = luminance.mean();

        boost::math::normal dist(0, 1);
        m_quantile = (Float) boost::math::quantile(dist, 1-m_pValue/2);
        Log(EInfo, "Configuring for a %.1f%% confidence interval, quantile=%f, avg. luminance=%e +- %e, num. batches=%d",
            (1-m_pValue)*100, m_quantile, m_averageLuminance, luminance.errorOfMean(), m_numBatches);
        return true;
    }

    void renderBlock(const Scene *scene, const Sensor *sensor,
            Sampler *sampler, ImageBlock *block, const bool &stop,
            const std::vector< TPoint2<uint8_t> > &points) const {
        bool needsApertureSample = sensor->needsApertureSample();
        bool needsTimeSample = sensor->needsTimeSample();

        RayDifferential eyeRay;
        RadianceQueryRecord rRec(scene, sampler);

        Float diffScaleFactor = 1.0f /
            std::sqrt((Float) sampler->getSampleCount());

        Point2 apertureSample(0.5f);
        Float timeSample = 0.5f;

        size_t sampleCount;
        block->clear();

        for (size_t i=0; i<points.size(); ++i) {
            Point2i offset = Point2i(points[i]) + Vector2i(block->getOffset());
            sampler->generate(offset);

            Spectrum batch[m_numBatches];
            for (size_t b = 0; b < m_numBatches; b++)
                batch[b] = Spectrum(0.0f);

            sampleCount = 0;
            Spectrum result;
            while (true) {
                if (stop)
                    return;

                const Point2 samplePos(Point2(offset) + Vector2(rRec.nextSample2D()));
                if (needsApertureSample)
                    apertureSample = rRec.nextSample2D();
                if (needsTimeSample)
                    timeSample = rRec.nextSample1D();

                // Add an extra sample in all batches
                ++sampleCount;
                for (size_t b = 0; b < m_numBatches; b++) {
                    rRec.newQuery(RadianceQueryRecord::ESensorRay, sensor->getMedium());
                    rRec.extra = RadianceQueryRecord::EAdaptiveQuery;

                    Spectrum sampleValue = sensor->sampleRayDifferential(
                        eyeRay, samplePos, apertureSample, timeSample);
                    eyeRay.scaleDifferential(diffScaleFactor); // HUH?

                    sampleValue *= m_subIntegrator->Li(eyeRay, rRec);

                    if (sampleValue.isFinite()) {
                        Spectrum delta = sampleValue - batch[b];
                        batch[b] += delta / sampleCount;
                    } else {
                        Log(EWarn, "Bad sample value: %s",
                                sampleValue.toString().c_str());
                    }

                    sampler->advance();
                }


                if (sampleCount * m_numBatches >= m_maxSampleFactor * sampler->getSampleCount()) {
                    result = robustAverage(batch);
                    break;
                } else if (sampleCount * m_numBatches >= sampler->getSampleCount()) {
                    /* Standard error of the primary estimator */
                    Float mean, stdErr;
                    Spectrum avg = robustAverage(batch, &mean, &stdErr);

                    /* Half width of the confidence interval */
                    Float ciWidth = stdErr * m_quantile;

                    /* Relative error heuristic */
                    Float base = std::max(mean, m_averageLuminance * 0.01f);

                    if (m_verbose && (sampleCount % 100) == 0)
                        Log(EDebug, "%i samples, mean=%f, std error=%f, ci width=%f, max allowed=%f", sampleCount, mean,
                            stdErr, ciWidth, base * m_maxError);

                    if (m_maxError > 0 && ciWidth <= m_maxError * base) {
                        result = avg;
                        break;
                    }
                }
            }

            /* TODO: THIS ONLY WORKS CORRECTLY WITH BOX FILTER!
             * Alternative solution is to store all batches as full bitmaps 
             * and put the samples in there at their correct positions so 
             * it works with all filters, but that incurs a large memory 
             * cost.
             * Better solution: fake it with a small, offset sub-bitmap to 
             * catch filter support and accumulate those pixels in the 
             * final bitmap at the correct position.
             */
            if (!block->put(Point2(offset) + Vector2(0.5f), result, rRec.alpha)) {
                Log(EWarn, "Had trouble submitting our final result: %s",
                        result.toString().c_str());
            }
        }
    }

    // Averages the spectra, excluding the smallest and largest sample
    Spectrum robustAverage(const Spectrum *batches,
            Float *meanLuminance = NULL, Float *stdErrLuminance = NULL) const {
        size_t minIdx = 0, maxIdx = 0;
        Float minim = 1./0.;
        Float maxim = -1./0.;
        for (size_t i = 0; i < m_numBatches; i++) {
            Float x = batches[i].getLuminance();
            if (x < minim) {
                minIdx = i;
                minim = x;
            }
            if (x > maxim) {
                maxIdx = i;
                maxim = x;
            }
        }
        size_t sampleCount = 0;
        Spectrum sumSpec(0.0f);
        Float meanLum = 0, meanSqrLum = 0;
        for (size_t i = 0; i < m_numBatches; i++) {
            if (i == minIdx || i == maxIdx)
                continue;
            Spectrum spec = batches[i];
            sumSpec += spec;

            Float x = spec.getLuminance();
            ++sampleCount;
            const Float delta = x - meanLum;
            meanLum += delta / sampleCount;
            meanSqrLum += delta * (x - meanLum);
        }
        Assert(sampleCount == m_numBatches - 2  ||  (sampleCount == m_numBatches - 1 && minIdx == maxIdx));
        Assert(sampleCount >= 2);
        const Float lumVar = meanSqrLum / (sampleCount-1);
        if (meanLuminance)
            *meanLuminance = meanLum;
        if (stdErrLuminance)
            *stdErrLuminance = sqrt(lumVar / sampleCount); // standard error of mean
        Spectrum meanSpec = sumSpec / sampleCount;
        Assert(meanLum == 0 || math::abs((meanSpec.getLuminance() - meanLum)) / meanLum < Epsilon);
        return sumSpec / sampleCount;
    }

    Spectrum Li(const RayDifferential &ray, RadianceQueryRecord &rRec) const {
        return m_subIntegrator->Li(ray, rRec);
    }

    Spectrum E(const Scene *scene, const Intersection &its, const Medium *medium,
            Sampler *sampler, int nSamples, bool includeIndirect) const {
        return m_subIntegrator->E(scene, its, medium,
            sampler, nSamples, includeIndirect);
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        MonteCarloIntegrator::serialize(stream, manager);
        manager->serialize(stream, m_subIntegrator.get());

        stream->writeInt(m_maxSampleFactor);
        stream->writeFloat(m_maxError);
        stream->writeFloat(m_quantile);
        stream->writeFloat(m_averageLuminance);
        stream->writeFloat(m_pValue);
        stream->writeSize(m_numBatches);
    }

    void bindUsedResources(ParallelProcess *proc) const {
        m_subIntegrator->bindUsedResources(proc);
    }

    void wakeup(ConfigurableObject *parent,
            std::map<std::string, SerializableObject *> &params) {
        m_subIntegrator->wakeup(this, params);
    }

    void cancel() {
        MonteCarloIntegrator::cancel();
        m_subIntegrator->cancel();
    }

    const Integrator *getSubIntegrator(int idx) const {
        if (idx != 0)
            return NULL;
        return m_subIntegrator.get();
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "AdaptiveIntegratorRobustMC[" << endl
            << "  maxSamples = " << m_maxSampleFactor << "," << endl
            << "  maxError = " << m_maxError << "," << endl
            << "  quantile = " << m_quantile << "," << endl
            << "  pvalue = " << m_pValue << "," << endl
            << "  subIntegrator = " << indent(m_subIntegrator->toString()) << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ref<MonteCarloIntegrator> m_subIntegrator;
    Float m_maxError, m_quantile, m_pValue, m_averageLuminance;
    int m_maxSampleFactor;
    size_t m_numBatches;
    bool m_verbose;
};

MTS_IMPLEMENT_CLASS_S(AdaptiveIntegratorRobustMC, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(AdaptiveIntegratorRobustMC, "Robost adaptive integrator for MonteCarloIntegrators");
MTS_NAMESPACE_END
