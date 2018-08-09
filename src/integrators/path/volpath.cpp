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

static StatsCounter avgNumSplits("Volumetric path tracer",
		"Average number of path splits", EAverage);
static StatsCounter avgPathLength("Volumetric path tracer",
		"Average path length", EAverage);

/*!\plugin{volpath}{Extended volumetric path tracer}
 * \order{4}
 * \parameters{
 *     \parameter{maxDepth}{\Integer}{Specifies the longest path depth
 *         in the generated output image (where \code{-1} corresponds to $\infty$).
 *	       A value of \code{1} will only render directly visible light sources.
 *	       \code{2} will lead to single-bounce (direct-only) illumination,
 *	       and so on. \default{\code{-1}}
 *	   }
 *	   \parameter{rrDepth}{\Integer}{Specifies the minimum path depth, after
 *	      which the implementation will start to use the ``russian roulette''
 *	      path termination criterion. \default{\code{5}}
 *	   }
 *	   \parameter{rrForcedDepth}{\Integer}{Specifies the minimum path depth, after
 *	      which the implementation will force the ``russian roulette'' path
 *	      termination probabilities to be less than unity. A value of \code{-1}
 *	      corresponds to $\infty$.\default{\code{-1}}
 *	   }
 *	   \parameter{rrTargetThroughput}{\Float}{The ``russian roulette'' path
 *	      termination criterion will try to keep the path weights at or
 *	      above this value. When the interesting parts of the scene end up
 *	      being much less bright than the light sources, setting this to a
 *	      lower value can be beneficial.
 *	      \default{\code{1.0}}
 *	   }
 *     \parameter{explicitSubsurfBoundary}{\Boolean}{
 *        When encauntering shapes with a subsurface scattering model that 
 *        supports explicit boundaries (i.e.\ models that can return
 *        $L_\mathrm{i}$ instead of $L_\mathrm{o}$, and thus can be coupled 
 *        to arbitrary BSDFs as boundaries), should we use this capability? 
 *        Usually, this would be the right thing to do.
 *        \default{\code{true}}
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
 *     \parameter{onlyPathsThatEnteredAVolume}{\Boolean}{
 *        If set to \true, then we only render contributions of paths that 
 *        entered a participating medium.
 *        \default{\code{false}}
 *     }
 *	   \parameter{minMediumScatteringChain}{\Integer}{
 *	      If paths travel through a participating medium, then only 
 *	      concider paths that have at least this many scattering 
 *	      interactions (inclusive). Setting this to $-1$ is equivalent to 
 *	      0, i.e.\ all paths are allowed.
 *	      \default{\code{-1}}
 *	   }
 *	   \parameter{maxMediumScatteringChain}{\Integer}{
 *	      If paths travel through a participating medium, then only 
 *	      concider paths that have at most this many scattering 
 *	      interactions (inclusive). Setting this to $-1$ is equivalent to 
 *	      $\infty$, i.e.\ all paths are allowed.
 *	      \default{\code{-1}}
 *	   }
 * }
 *
 * This plugin provides a volumetric path tracer that can be used to
 * compute approximate solutions of the radiative transfer equation.
 * Its implementation makes use of multiple importance sampling to
 * combine BSDF and phase function sampling with direct illumination
 * sampling strategies. On surfaces, it behaves exactly
 * like the standard path tracer.
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
 *    \item This integrator will generally perform poorly when rendering
 *      participating media that have a different index of refraction compared
 *      to the surrounding medium.
 *    \item This integrator has poor convergence properties when rendering
 *      caustics and similar effects. In this case, \pluginref{bdpt} or
 *      one of the photon mappers may be preferable.
 * }
 */
class VolumetricPathTracer : public MonteCarloIntegrator {
protected:
	bool m_explicitSubsurfBoundary;
	bool m_onlyPathsThatEnteredAVolume;

	/* Paths that enter medium should have {at least, at most} this number 
	 * of medium scatterings (-1 to disable, bounds are inclusive): */
	int m_minMediumScatteringChain;
	int m_maxMediumScatteringChain;

public:
	VolumetricPathTracer(const Properties &props) : MonteCarloIntegrator(props) {
		m_onlyPathsThatEnteredAVolume = props.getBoolean("onlyPathsThatEnteredAVolume", false);
		m_minMediumScatteringChain = props.getInteger("minMediumScatteringChain", -1);
		m_maxMediumScatteringChain = props.getInteger("maxMediumScatteringChain", -1);
		m_explicitSubsurfBoundary = props.getBoolean("explicitSubsurfBoundary", true);

		if (m_minMediumScatteringChain >= 0 && m_maxMediumScatteringChain >= 0
				&& m_minMediumScatteringChain > m_maxMediumScatteringChain) {
			Log(EError, "Conflicting options: "
					"minMedimuScatteringChain > maxMedimuScatteringChain!");
		}
	}

	/// Unserialize from a binary data stream
	VolumetricPathTracer(Stream *stream, InstanceManager *manager)
			 : MonteCarloIntegrator(stream, manager) {
		m_onlyPathsThatEnteredAVolume = stream->readBool();
		m_minMediumScatteringChain = stream->readInt();
		m_maxMediumScatteringChain = stream->readInt();
		m_explicitSubsurfBoundary = stream->readBool();
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		MonteCarloIntegrator::serialize(stream, manager);
		stream->writeBool(m_onlyPathsThatEnteredAVolume);
		stream->writeInt(m_minMediumScatteringChain);
		stream->writeInt(m_maxMediumScatteringChain);
		stream->writeBool(m_explicitSubsurfBoundary);
	}

	Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
		RayDifferential ray(r);
		Float eta = 1.0f;
		Spectrum throughput(1.0f);
		int mediumInteractionChain = 0;

		/* A word on throughputs: we explicitly pass around a throughput 
		 * value which is the troughput starting 'here', i.e. at the call 
		 * of Li(). The throughput in the rRec is *only* used for RR/path 
		 * splitting decisions! */
		Spectrum internalThroughput(1.0f);

		/* Don't do RR yet, but *do* check if we need to split already, 
		 * based on the historical throughput of the path that we got from 
		 * the rRec. */
		int initial_n = 1 + m_rr.split(rRec.splits, rRec.throughput, eta, rRec.sampler);
		bool hasEnteredAVolume = false;
		Spectrum Li = LiPathSteps(ray, rRec, eta, internalThroughput, 
				mediumInteractionChain, hasEnteredAVolume, initial_n);
		return Li;
	}

	/**
	 *
	 * \param initialThroughput The throughput *before* the path splitting 
	 * (i.e. the combined weight of all n paths which we will trace), but 
	 * possibly rescaled by a Russian Roulette weight if applicable. Unit 
	 * throughput is the point where Li() called. The throughput in the 
	 * rRec is also updated along the way, and used for path splitting 
	 * decisions, but *not* for computing Li.
	 *
	 * WARNING: rRec depth becomes meaningless at return! (due to path 
	 * splitting!), the splits field does remains valid
	 */
	Spectrum LiPathSteps(const RayDifferential &origRay, RadianceQueryRecord 
			&rRec, const Float origEta, const Spectrum &initialThroughput,
			const int origMediumInteractionChain,
			const bool origHadVolumeInteraction,
			const int n) const {
		if (rRec.depth > m_maxDepth && m_maxDepth > 0) {
			avgPathLength.incrementBase();
			avgPathLength += rRec.depth;
			return Spectrum(0.0f);
		}

		// Set up original variables and rescale throughputs
		RadianceQueryRecord rRecOrig(rRec);
		rRecOrig.throughput /= n;
		Spectrum rescaledInitialThroughput = initialThroughput / n;

		int totalNumSplits = rRecOrig.splits; // includes the current n already
		Spectrum Li(0.0f);
		for (int i = 0; i < n; i++) {
			Spectrum thisThroughput = rescaledInitialThroughput;
			RayDifferential ray(origRay);
			int mediumInteractionChain = origMediumInteractionChain;
			bool hasEnteredAVolume = origHadVolumeInteraction;
			Float eta = origEta;
			rRec = rRecOrig;
			rRec.splits = totalNumSplits;

			/* To avoid overflowing the stack, we only explicitly call 
			 * ourselves recursively if we *have* to split into 2 or more 
			 * paths. If we don't split the path, we loop internally until 
			 * the path stops or it has to be split. */
			int nRecursive; /* number of paths for next step (0 = stop, 1 = 
							   no split, > 1 = split in this many) */
			do {
				Spectrum throughputBeforeStep = thisThroughput;
				if (!LiPathStep(ray, rRec, eta, Li, thisThroughput, 
						mediumInteractionChain, hasEnteredAVolume)) {
					avgPathLength.incrementBase();
					avgPathLength += rRec.depth;
					nRecursive = 0;
					break;
				}
				// Update 'global' throughput
				rRec.throughput *= thisThroughput
						* throughputBeforeStep.invertButKeepZero();

				int numSplitsHere = 0;
				Float q = m_rr.roulette(
						rRec.depth, rRec.throughput, eta, rRec.sampler);
				if (q == 0.0f) {
					avgPathLength.incrementBase();
					avgPathLength += rRec.depth;
					nRecursive = 0;
					break;
				}
				if (q == 1.0f) // There was no RR -> possibly try splitting
					numSplitsHere = m_rr.split(rRec.splits, 
							rRec.throughput, eta, rRec.sampler);
				nRecursive = numSplitsHere + 1;
				/* Update the throughputs. If nRecursive != 1, then  
				 * LiPathSteps takes care of the 1/nRecursive factor itself 
				 * below. */
				rRec.throughput /= q;
				thisThroughput /= q;

				avgNumSplits.incrementBase();
				avgNumSplits += numSplitsHere;
			} while (nRecursive == 1);

			if (nRecursive == 0)
				continue; // End of this path, go to the next split

			/* We have nRecursive > 1, so we explicitly have to call 
			 * ourselves recursively */
			Li += LiPathSteps(ray, rRec, eta, thisThroughput, 
					mediumInteractionChain, hasEnteredAVolume,
					nRecursive);
			totalNumSplits = rRec.splits;
		}

		return Li;
	}

	bool includeMediumChainDepth(int mediumInteractionChain) const {
		return (m_minMediumScatteringChain < 0
		             || mediumInteractionChain >= m_minMediumScatteringChain)
		    && (m_maxMediumScatteringChain < 0
		             || mediumInteractionChain <= m_maxMediumScatteringChain);
	}

	/**
	 * We only update the explicit throughput param and leave 
	 * rRec.throughput unchanged!
	 */
	bool LiPathStep(RayDifferential &ray, RadianceQueryRecord &rRec,
			Float &eta, Spectrum &Li, Spectrum &throughput,
			int &mediumInteractionChain, bool &hasEnteredAVolume) const {
		/* Some aliases and local variables */
		const Scene *scene = rRec.scene;
		Intersection &its = rRec.its;
		MediumSamplingRecord mRec;

		/* Perform the first ray intersection (or ignore if the
		   intersection has already been provided). */
		rRec.rayIntersect(ray);

		bool scattered = rRec.depth != 1;
		hasEnteredAVolume = hasEnteredAVolume || (rRec.medium != NULL);

		/* ==================================================================== */
		/*                 Radiative Transfer Equation sampling                 */
		/* ==================================================================== */
		if (rRec.medium && rRec.medium->sampleDistance(Ray(ray, 0, its.t),
				mRec, rRec.sampler, &throughput)) {
			if (m_maxMediumScatteringChain >= 0
					&& mediumInteractionChain >= m_maxMediumScatteringChain)
				return false;
			mediumInteractionChain++;

			/* Sample the integral
			   \int_x^y tau(x, x') [ \sigma_s \int_{S^2} \rho(\omega,\omega') L(x,\omega') d\omega' ] dx'
			*/
			const PhaseFunction *phase = mRec.getPhaseFunction();

			if (rRec.depth >= m_maxDepth && m_maxDepth != -1) {
				// No more scattering events allowed
				return false;
			}

			throughput *= mRec.sigmaS * mRec.transmittance / mRec.pdfSuccess;

			/* ==================================================================== */
			/*                          Luminaire sampling                          */
			/* ==================================================================== */

			/* Estimate the single scattering component if this is requested */
			DirectSamplingRecord dRec(mRec.p, mRec.time);

			if (rRec.type & RadianceQueryRecord::EDirectMediumRadiance
					&& includeMediumChainDepth(mediumInteractionChain)) {
				int interactions = m_maxDepth - rRec.depth - 1;

				Spectrum value = scene->sampleAttenuatedEmitterDirect(
						dRec, rRec.medium, interactions,
						rRec.nextSample2D(), rRec.sampler);

				if (!value.isZero()) {
					const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

					/* Evaluate the phase function */
					PhaseFunctionSamplingRecord pRec(mRec, -ray.d, dRec.d);
					Float phaseVal = phase->eval(pRec);

					if (phaseVal != 0.0f) {
						/* Calculate prob. of having sampled that direction using
						   phase function sampling */
						Float phasePdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
								? phase->pdf(pRec) : (Float) 0.0f;

						/* Weight using the power heuristic */
						const Float weight = miWeight(dRec.pdf, phasePdf);
						if (!m_onlyPathsThatEnteredAVolume || hasEnteredAVolume)
							Li += throughput * value * phaseVal * weight;
					}
				}
			}

			/* ==================================================================== */
			/*                         Phase function sampling                      */
			/* ==================================================================== */

			Float phasePdf;
			PhaseFunctionSamplingRecord pRec(mRec, -ray.d);
			Float phaseVal = phase->sample(pRec, phasePdf, rRec.sampler);
			if (phaseVal == 0.0f)
				return false;
			throughput *= phaseVal;

			/* Trace a ray in this direction */
			ray = Ray(mRec.p, pRec.wo, ray.time);
			ray.mint = 0;

			Spectrum value(0.0f);
			rayIntersectAndLookForEmitter(scene, rRec.sampler, rRec.medium,
				m_maxDepth - rRec.depth - 1, ray, its, dRec, value);

			/* If a luminaire was hit, estimate the local illumination and
			   weight using the power heuristic */
			if (!value.isZero() && (rRec.type & RadianceQueryRecord::EDirectMediumRadiance)
					&& includeMediumChainDepth(mediumInteractionChain)) {
				const Float emitterPdf = scene->pdfEmitterDirect(dRec);
				if (!m_onlyPathsThatEnteredAVolume || hasEnteredAVolume)
					Li += throughput * value * miWeight(phasePdf, emitterPdf);
			}

			/* ==================================================================== */
			/*                         Multiple scattering                          */
			/* ==================================================================== */

			/* Stop if multiple scattering was not requested */
			if (!(rRec.type & RadianceQueryRecord::EIndirectMediumRadiance))
				return false;
			rRec.type = RadianceQueryRecord::ERadianceNoEmission;
		} else {
			if (rRec.medium && !includeMediumChainDepth(mediumInteractionChain))
				return false;
			mediumInteractionChain = 0;

			/* Sample
				tau(x, y) (Surface integral). This happens with probability mRec.pdfFailure
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
						value *= rRec.medium->evalTransmittance(ray, rRec.sampler);
					if (!m_onlyPathsThatEnteredAVolume || hasEnteredAVolume)
						Li += value;
				}

				return false;
			}

			/* Possibly include emitted radiance if requested */
			if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
				&& (!m_hideEmitters || scattered))
				if (!m_onlyPathsThatEnteredAVolume || hasEnteredAVolume)
					Li += throughput * its.Le(-ray.d);

			/* Include radiance from a subsurface integrator if requested */
			if (its.hasSubsurface() && (!m_explicitSubsurfBoundary || !its.hasLiSubsurface())
					&& (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance)) {
				if (!m_onlyPathsThatEnteredAVolume || hasEnteredAVolume)
					Li += throughput * its.LoSub(
							scene, rRec.sampler, -ray.d, throughput, rRec.depth);
			}

			if (rRec.depth >= m_maxDepth && m_maxDepth != -1)
				return false;

			/* Prevent light leaks due to the use of shading normals */
			Float wiDotGeoN = -dot(its.geoFrame.n, ray.d),
				  wiDotShN  = Frame::cosTheta(its.wi);
			if (wiDotGeoN * wiDotShN < 0 && m_strictNormals)
				return false;

			/* ==================================================================== */
			/*                          Luminaire sampling                          */
			/* ==================================================================== */

			const BSDF *bsdf = its.getBSDF(ray);
			DirectSamplingRecord dRec(its);

			/* Estimate the direct illumination if this is requested */
			if ((rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) &&
				(bsdf->getType() & BSDF::ESmooth)) {
				int interactions = m_maxDepth - rRec.depth - 1;

				Spectrum value = scene->sampleAttenuatedEmitterDirect(
						dRec, its, rRec.medium, interactions,
						rRec.nextSample2D(), rRec.sampler);

				if (!value.isZero()) {
					const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

					/* Evaluate BSDF * cos(theta) */
					BSDFSamplingRecord bRec(its, its.toLocal(dRec.d));
					const Spectrum bsdfVal = bsdf->eval(bRec);

					Float woDotGeoN = dot(its.geoFrame.n, dRec.d);

					/* Prevent light leaks due to the use of shading normals */
					if (!bsdfVal.isZero() && (!m_strictNormals ||
						woDotGeoN * Frame::cosTheta(bRec.wo) > 0)) {
						/* Calculate prob. of having generated that direction
						   using BSDF sampling */
						Float bsdfPdf = (emitter->isOnSurface()
								&& dRec.measure == ESolidAngle)
								? bsdf->pdf(bRec) : (Float) 0.0f;

						/* Weight using the power heuristic */
						const Float weight = miWeight(dRec.pdf, bsdfPdf);
						if (!m_onlyPathsThatEnteredAVolume || hasEnteredAVolume)
							Li += throughput * value * bsdfVal * weight;
					}
				}
			}


			/* ==================================================================== */
			/*                            BSDF sampling                             */
			/* ==================================================================== */

			/* Sample BSDF * cos(theta) */
			BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
			Float bsdfPdf;
			Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
			if (bsdfWeight.isZero())
				return false;

			/* Prevent light leaks due to the use of shading normals */
			const Vector wo = its.toWorld(bRec.wo);
			Float woDotGeoN = dot(its.geoFrame.n, wo);
			if (woDotGeoN * Frame::cosTheta(bRec.wo) <= 0 && m_strictNormals)
				return false;

			/* Trace a ray in this direction */
			ray = Ray(its.p, wo, ray.time);

			/* Keep track of the throughput, medium, and relative
			   refractive index along the path */
			throughput *= bsdfWeight;
			eta *= bRec.eta;
			if (its.isMediumTransition())
				rRec.medium = its.getTargetMedium(ray.d);

			/* If we cross the surface and have a subsurface integrator 
			 * that can sample Li behind it, then return it if 
			 * requested or stop if it is not requested.
			 * Don't check for cosTheta<0 because the subsurface 
			 * integrator may allow 'incoming' outgoing directions, 
			 * (e.g. DirectSamplingSubsurface with 
			 * allowIncomingOutgoingDirections to check boundary 
			 * conditions) */
			if (its.hasSubsurface() && its.hasLiSubsurface() && m_explicitSubsurfBoundary) {
				if (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance) {
					if (!m_onlyPathsThatEnteredAVolume || hasEnteredAVolume)
						Li += throughput * its.LiSub(scene, rRec.sampler, wo,
										throughput, rRec.splits, rRec.depth);
				}

				/* If 'outgoing' direction is away from the subsurf medium 
				 * (wo pointing into the medium) -> then we are looking at 
				 * the subsurf medium from the outside. If there is no 
				 * explicit medium defined within, then we want the subsurf 
				 * medium to be opaque apart from its explicit Li, so we 
				 * stop the tracing.*/
				if (Frame::cosTheta(bRec.wo) < 0) {
					if (rRec.medium == NULL)
						return false;
					/* Here: medium!=NULL, i.e. there is an explicit medium 
					 * defined, this can be used for exact single 
					 * scattering, for instance. In that case: don't abort 
					 * but follow through */
				}

			}

			/* Handle index-matched medium transitions specially */
			if (bRec.sampledType == BSDF::ENull) {
				if (!(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
					return false;
				rRec.type = scattered ? RadianceQueryRecord::ERadianceNoEmission
					: RadianceQueryRecord::ERadiance;
				scene->rayIntersect(ray, its);
				rRec.depth++;
				return true;;
			}

			Spectrum value(0.0f);
			rayIntersectAndLookForEmitter(scene, rRec.sampler, rRec.medium,
				m_maxDepth - rRec.depth - 1, ray, its, dRec, value);

			/* If a luminaire was hit, estimate the local illumination and
			   weight using the power heuristic */
			if (!value.isZero() && (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {
				const Float emitterPdf = (!(bRec.sampledType & BSDF::EDelta)) ?
					scene->pdfEmitterDirect(dRec) : 0;
				if (!m_onlyPathsThatEnteredAVolume || hasEnteredAVolume)
					Li += throughput * value * miWeight(bsdfPdf, emitterPdf);
			}

			/* ==================================================================== */
			/*                         Indirect illumination                        */
			/* ==================================================================== */

			/* Stop if indirect illumination was not requested */
			if (!(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
				return false;

			rRec.type = RadianceQueryRecord::ERadianceNoEmission;
		}

		rRec.depth++;
		return true;
	}

	/**
	 * This function is called by the recursive ray tracing above after
	 * having sampled a direction from a BSDF/phase function. Due to the
	 * way in which this integrator deals with index-matched boundaries,
	 * it is necessarily a bit complicated (though the improved performance
	 * easily pays for the extra effort).
	 *
	 * This function
	 *
	 * 1. Intersects 'ray' against the scene geometry and returns the
	 *    *first* intersection via the '_its' argument.
	 *
	 * 2. It checks whether the intersected shape was an emitter, or if
	 *    the ray intersects nothing and there is an environment emitter.
	 *    In this case, it returns the attenuated emittance, as well as
	 *    a DirectSamplingRecord that can be used to query the hypothetical
	 *    sampling density at the emitter.
	 *
	 * 3. If current shape is an index-matched medium transition, the
	 *    integrator keeps on looking on whether a light source eventually
	 *    follows after a potential chain of index-matched medium transitions,
	 *    while respecting the specified 'maxDepth' limits. It then returns
	 *    the attenuated emittance of this light source, while accounting for
	 *    all attenuation that occurs on the wya.
	 */
	void rayIntersectAndLookForEmitter(const Scene *scene, Sampler *sampler,
			const Medium *medium, int maxInteractions, Ray ray, Intersection &_its,
			DirectSamplingRecord &dRec, Spectrum &value) const {
		Intersection its2, *its = &_its;
		Spectrum transmittance(1.0f);
		bool surface = false;
		int interactions = 0;

		while (true) {
			surface = scene->rayIntersect(ray, *its);

			if (medium)
				transmittance *= medium->evalTransmittance(Ray(ray, 0, its->t), sampler);

			if (surface && (interactions == maxInteractions ||
				!(its->getBSDF()->getType() & BSDF::ENull) ||
				its->isEmitter())) {
				/* Encountered an occluder / light source */
				break;
			}

			if (!surface)
				break;

			if (transmittance.isZero())
				return;

			if (its->hasSubsurface())
				return; /* SSS models are opaque */

			if (its->isMediumTransition())
				medium = its->getTargetMedium(ray.d);

			Vector wo = its->shFrame.toLocal(ray.d);
			BSDFSamplingRecord bRec(*its, -wo, wo, ERadiance);
			bRec.typeMask = BSDF::ENull;
			transmittance *= its->getBSDF()->eval(bRec, EDiscrete);

			ray.o = ray(its->t);
			ray.mint = Epsilon;
			its = &its2;

			if (++interactions > 100) { /// Just a precaution..
				Log(EWarn, "rayIntersectAndLookForEmitter(): round-off error issues?");
				return;
			}
		}

		if (surface) {
			/* Intersected something - check if it was a luminaire */
			if (its->isEmitter()) {
				dRec.setQuery(ray, *its);
				value = transmittance * its->Le(-ray.d);
			}
		} else {
			/* Intersected nothing -- perhaps there is an environment map? */
			const Emitter *env = scene->getEnvironmentEmitter();

			if (env && env->fillDirectSamplingRecord(dRec, ray))
				value = transmittance * env->evalEnvironment(RayDifferential(ray));
		}
	}

	inline Float miWeight(Float pdfA, Float pdfB) const {
		pdfA *= pdfA; pdfB *= pdfB;
		return pdfA / (pdfA + pdfB);
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "VolumetricPathTracer[" << endl
			<< "  explicitSubsurfBoundary = " << m_explicitSubsurfBoundary << "," << endl
			<< "  onlyPathsThatEnteredAVolume = " << m_onlyPathsThatEnteredAVolume << "," << endl
			<< "  minMediumScatteringChain = " << m_minMediumScatteringChain << "," << endl
			<< "  maxMediumScatteringChain = " << m_maxMediumScatteringChain << "," << endl
			<< "  maxDepth = " << m_maxDepth << "," << endl
			<< "  rr = " << m_rr.toString() << "," << endl
			<< "  strictNormals = " << m_strictNormals << endl
			<< "]";
		return oss.str();
	}

	MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_S(VolumetricPathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(VolumetricPathTracer, "Volumetric path tracer");
MTS_NAMESPACE_END
