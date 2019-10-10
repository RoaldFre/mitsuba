/* This directional dipole code is (quite directly) based on the sample
 * code provided by Jeppe Revall Frisvad, Toshiya Hachisuka and Thomas Kim
 * Kjeldsen with their paper 'Directional Dipole Model for Subsurface
 * Scattering', which in itself is based on smallpt, a path tracer by Kevin
 * Beason. */
#include <mitsuba/render/scene.h>
#include <mitsuba/render/dss.h>
#include <mitsuba/core/plugin.h>
#include "../medium/materials.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

MTS_NAMESPACE_BEGIN

class DirectionalDipole : public DirectSamplingSubsurface {
private:
    struct Vec {union {struct{double x, y, z;}; struct{double r, g, b;};}; // vector: position, also color (r,g,b)
        Vec(double x_ = 0, double y_ = 0, double z_ = 0) {x = x_; y = y_; z = z_;}
        inline double& operator[](const int i) {return *(&x + i);}
        inline const double& operator[](const int i) const {return *(&x + i);}
        inline Vec operator-() const {return Vec(-x, -y, -z);}
        inline Vec operator+(const Vec& b) const {return Vec(x + b.x, y + b.y, z + b.z);}
        inline Vec operator-(const Vec& b) const {return Vec(x - b.x, y - b.y, z - b.z);}
        inline Vec operator*(const Vec& b) const {return Vec(x * b.x, y * b.y, z * b.z);}
        inline Vec operator/(const Vec& b) const {return Vec(x / b.x, y / b.y, z / b.z);}
        inline Vec operator+(double b) const {return Vec(x + b, y + b, z + b);}
        inline Vec operator-(double b) const {return Vec(x - b, y - b, z - b);}
        inline Vec operator*(double b) const {return Vec(x * b, y * b, z * b);}
        inline Vec operator/(double b) const {return Vec(x / b, y / b, z / b);}
        inline friend Vec operator+(double b, const Vec& v) {return Vec(b + v.x, b + v.y, b + v.z);}
        inline friend Vec operator-(double b, const Vec& v) {return Vec(b - v.x, b - v.y, b - v.z);}
        inline friend Vec operator*(double b, const Vec& v) {return Vec(b * v.x, b * v.y, b * v.z);}
        inline friend Vec operator/(double b, const Vec& v) {return Vec(b / v.x, b / v.y, b / v.z);}
        inline double len() const {return sqrt(x * x + y * y + z * z);}
        inline Vec normalized() const {return (*this) / this->len();}
        inline friend Vec sqrt(const Vec& b) {return Vec(sqrt(b.x), sqrt(b.y), sqrt(b.z));}
        inline friend Vec exp(const Vec& b) {return Vec(exp(b.x), exp(b.y), exp(b.z));}
        inline double dot(const Vec& b) const {return x * b.x + y * b.y + z * b.z;}
        inline Vec operator%(const Vec& b) const {return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);}
    };

    struct Ray {Vec o, d; Ray() { }; Ray(Vec o_, Vec d_) : o(o_), d(d_) { }};

    inline double C1(const double n) {
        double r;
        if (n > 1.0) {
            r = -9.23372 + n * (22.2272 + n * (-20.9292 + n * (10.2291 + n * (-2.54396 + 0.254913 * n))));
        } else {
            r = 0.919317 + n * (-3.4793 + n * (6.75335 + n *  (-7.80989 + n *(4.98554 - 1.36881 * n))));
        }
        return r / 2.0;
    }
    inline double C2(const double n) {
        double r = -1641.1 + n * (1213.67 + n * (-568.556 + n * (164.798 + n * (-27.0181 + 1.91826 * n))));
        r += (((135.926 / n) - 656.175) / n + 1376.53) / n;
        return r / 3.0;
    }

    // constants
    Spectrum sigma_t;
    Spectrum sigma_sp;
    Spectrum sigma_tp;
    Spectrum albedo_p;
    Spectrum D;
    Spectrum sigma_tr;
    Spectrum de;
    double Cp_norm;
    double Cp;
    double Ce;
    double A;

    // directional dipole
    // --------------------------------
    inline double Sp_d(const Vec& x, const Vec& w, const double& r, const Vec& n, const int j) const {
        // evaluate the profile
        const double s_tr_r = sigma_tr[j] * r;
        const double s_tr_r_one = 1.0 + s_tr_r;
        const double x_dot_w = x.dot(w);
        const double r_sqr = r * r;

        const double t0 = Cp_norm * (1.0 / (4.0 * M_PI * M_PI)) * exp(-s_tr_r) / (r * r_sqr);
        const double t1 = r_sqr / D[j] + 3.0 * s_tr_r_one * x_dot_w;
        const double t2 = 3.0 * D[j] * s_tr_r_one * w.dot(n);
        const double t3 = (s_tr_r_one + 3.0 * D[j] * (3.0 * s_tr_r_one + s_tr_r * s_tr_r) / r_sqr * x_dot_w) * x.dot(n);

        return t0 * (Cp * t1 - Ce * (t2 - t3));
    }
    inline double bssrdf(const Vec& xi, const Vec& ni, const Vec& wi, const Vec& xo, const Vec& no, const Vec& wo, const int j) const {
        // distance
        const Vec xoxi = xo - xi;
        const double r = xoxi.len();

        // modified normal
        const Vec ni_s = (xoxi.normalized()) % ((ni % xoxi).normalized());

        // directions of ray sources
        const double nnt = 1.0 / m_eta, ddn = -wi.dot(ni);
        const Vec wr = (wi * -nnt - ni * (ddn * nnt + sqrt(1.0 - nnt * nnt * (1.0 - ddn * ddn)))).normalized();
        const Vec wv = wr - ni_s * (2.0 * wr.dot(ni_s));

        // distance to real sources
        const double cos_beta = -sqrt((r * r - xoxi.dot(wr) * xoxi.dot(wr)) / (r * r + de[j] * de[j]));
        double dr;
        const double mu0 = -no.dot(wr);
        if (mu0 > 0.0) {
            dr = sqrt((D[j] * mu0) * ((D[j] * mu0) - de[j] * cos_beta * 2.0) + r * r);
        } else {
            dr = sqrt(1.0 / (3.0 * sigma_t[j] * 3.0 * sigma_t[j]) + r * r);
        }

        // distance to virtual source
        const Vec xoxv = xo - (xi + ni_s * (2.0 * A * de[j]));
        const double dv = xoxv.len();

        // BSSRDF
        const double result = Sp_d(xoxi, wr, dr, no, j) - Sp_d(xoxv, wv, dv, no, j);

        // clamping to zero
        return (result < 0.0) ? 0.0 : result;
    }
    // --------------------------------

public:
    DirectionalDipole(const Properties &props)
        : DirectSamplingSubsurface(props) {

        m_reciprocal = props.getBoolean("reciprocal", false);
        lookupMaterial(props, m_sigmaS, m_sigmaA, m_g, &m_eta);
    }

    DirectionalDipole(Stream *stream, InstanceManager *manager)
     : DirectSamplingSubsurface(stream, manager) {
        m_sigmaS = Spectrum(stream);
        m_sigmaA = Spectrum(stream);
        m_g = Spectrum(stream);
        m_reciprocal = stream->readBool();
        configure();
    }

    void configure() {
        // setup some constants
        sigma_t = m_sigmaS + m_sigmaA;
        sigma_sp = m_sigmaS * (Spectrum(1.0f) - m_g);
        sigma_tp = sigma_sp + m_sigmaA;
        albedo_p = sigma_sp / sigma_tp;
        D = Spectrum(1.0f) / (3.0 * sigma_tp);
        sigma_tr = (m_sigmaA / D).sqrt();
        de = 2.131 * D / albedo_p.sqrt();
        Cp_norm = 1.0 / (1.0 - 2.0 * C1(1.0 / m_eta));
        Cp = (1.0 - 2.0 * C1(m_eta)) / 4.0;
        Ce = (1.0 - 3.0 * C2(m_eta)) / 2.0;
        A = (1.0 - Ce) / (2.0 * Cp);

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

    void serialize(Stream *stream, InstanceManager *manager) const {
        DirectSamplingSubsurface::serialize(stream, manager);
        m_sigmaS.serialize(stream);
        m_sigmaA.serialize(stream);
        m_g.serialize(stream);
        stream->writeBool(m_reciprocal);
    }

    virtual Spectrum bssrdf(const Scene *scene, const Point &p_in, const Vector &d_in, const Normal &n_in,
            const Point &p_out, const Vector &d_out, const Normal &n_out,
            const void *extraParams) const {
        Spectrum transport;
        Vec xi(p_in.x, p_in.y, p_in.z);
        Vec ni(n_in.x, n_in.y, n_in.z);
        Vec wi(-d_in.x, -d_in.y, -d_in.z);
        Vec xo(p_out.x, p_out.y, p_out.z);
        Vec no(n_out.x, n_out.y, n_out.z);
        Vec wo(d_out.x, d_out.y, d_out.z);
        for (int i = 0; i < SPECTRUM_SAMPLES; i++) {
            if (m_reciprocal) {
                transport[i] = 0.5 * (bssrdf(xi, ni, wi, xo, no, wo, i)
                                    + bssrdf(xo, no, wo, xi, ni, wi, i));
            } else {
                transport[i] = bssrdf(xi, ni, wi, xo, no, wo, i);
            }
        }

        /* For eta != 1: modulate with Fresnel transmission.
         * NOTE: The explicit boundary should be a index-matched or null
         * boundary anyway for this to make sense */
        if (m_eta != 1.0f) {
            transport *= 1.0f - fresnelDielectricExt(dot(n_out, d_out), m_eta);
            transport *= 1.0f - fresnelDielectricExt(dot(n_in,  -d_in), m_eta);
        }

        return transport;
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

    std::string toString() const {
        std::ostringstream oss;
        oss << "DirectionalDipole[" << endl;
        oss << "  sigmaS = " << m_sigmaS.toString() << endl;
        oss << "  sigmaA = " << m_sigmaA.toString() << endl;
        oss << "  g = " << m_g.toString() << endl;
        oss << "]" << endl;
        return oss.str();
    }

    MTS_DECLARE_CLASS()
protected:
    Spectrum m_sigmaS, m_sigmaA, m_g;
    bool m_reciprocal;
};

MTS_IMPLEMENT_CLASS_S(DirectionalDipole, false, DirectSamplingSubsurface)
MTS_EXPORT_PLUGIN(DirectionalDipole, "Directional (isotropic) dipole model");
MTS_NAMESPACE_END
