// MLX supplies the enclosing kernel and typed input/output arguments.
// One SIMD group per (sequence, query head). The first query of each
// GQA group is the sole owner of its shared K/V head and raw history.
uint lane = thread_position_in_threadgroup.x;
uint qid = threadgroup_position_in_grid.x;
uint b = qid / QH, qh = qid % QH;
uint kh = qh / (QH / KH);
uint kv = b * KH + kh;
bool owner = (qh % (QH / KH)) == 0;
constexpr int channels = (DIM + 31) / 32;
constexpr int rel_channels = (RDIM + 31) / 32;
float qs[channels], ks[channels], vs[channels], rs[rel_channels];
float qss = 0.0f, kss = 0.0f, vss = 0.0f, rss = 0.0f;
for (int i = 0; i < channels; i++) {
    uint d = 32 * i + lane;
    if (d >= DIM) continue;
    uint qi = qid * DIM + d;
    qs[i] = float(q[qi]) * float(qscale[qh * DIM + d]);
    qss += qs[i] * qs[i];
    if (!owner) continue;
    uint ki = kv * DIM + d;
    float raw_k = float(k[ki]), raw_v = float(v[ki]);
    float ksum = raw_k, vsum = raw_v;
    for (int lag = 0; lag < 4; lag++) {
        uint wi = (lag * KH + kh) * DIM + d;
        float wkey = float(wk[wi]), wval = float(wv[wi]);
        if (DYN) {
            wkey += float(dk[kv * 4 + lag]);
            wval += float(dv[kv * 4 + lag]);
        }
        float hk = raw_k, hv = raw_v;
        if (lag > 0) {
            uint history = (kv * 3 + lag - 1) * DIM + d;
            hk = float(histk[history]);
            hv = float(histv[history]);
        }
        ksum += wkey * hk;
        vsum += wval * hv;
        if (lag < 3) {
            uint history = (kv * 3 + lag) * DIM + d;
            hko[history] = half(hk);
            hvo[history] = half(hv);
        }
    }
    float norm = DYN ? (1.0f / 3.0f) : (1.0f / metal::sqrt(5.0f));
    ks[i] = float(half(ksum * norm));
    vs[i] = float(half(vsum * norm));
    kss += ks[i] * ks[i];
    vss += vs[i] * vs[i];
}
for (int i = 0; i < rel_channels; i++) {
    uint d = 32 * i + lane;
    if (d >= RDIM) continue;
    rs[i] = float(r[qid * RDIM + d]) * float(rscale[qh * RDIM + d]);
    rss += rs[i] * rs[i];
}
float qfactor = metal::min(1.0f, 22.627416998f / (metal::sqrt(metal::simd_sum(qss)) + 1.e-6f));
float kfactor = metal::min(1.0f, 22.627416998f / (metal::sqrt(metal::simd_sum(kss)) + 1.e-6f));
float vfactor = metal::rsqrt(metal::simd_sum(vss) / float(DIM) + 1.e-6f);
float rfactor = metal::min(1.0f, 8.0f / (metal::sqrt(metal::simd_sum(rss)) + 1.e-6f));
for (int i = 0; i < channels; i++) {
    uint d = 32 * i + lane;
    if (d >= DIM) continue;
    qo[qid * DIM + d] = half(float(half(qs[i] * qfactor)) * tau[0]);
    if (owner) {
        ko[kv * DIM + d] = half(ks[i] * kfactor);
        vo[kv * DIM + d] = half(vs[i] * vfactor);
    }
}
for (int i = 0; i < rel_channels; i++) {
    uint d = 32 * i + lane;
    if (d < RDIM) ro[qid * RDIM + d] = half(float(half(rs[i] * rfactor)) * tau[0]);
}
