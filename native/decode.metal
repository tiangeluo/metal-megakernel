#include <metal_stdlib>
using namespace metal;

// One SIMD group per query head. A threadgroup owns all query heads sharing a KV head.
// Up to four channels per lane covers d_head <= 128 with the probed SIMD width of 32.
struct Params {
    uint batch, queries, kv, dim, relative_dim, extent, capacity, position;
    uint window, dynamic, tile, chunks;
};
#define INPUTS                                                                                               \
    constant Params &p [[buffer(0)]], device const half *q [[buffer(1)]],                                    \
        device const half *k [[buffer(2)]], device const half *v [[buffer(3)]],                              \
        device const half *r [[buffer(4)]], device const half *sink [[buffer(5)]],                           \
        device const half *gate [[buffer(6)]], device const half *dk [[buffer(7)]],                          \
        device const half *dv [[buffer(8)]], device const half *wk [[buffer(9)]],                            \
        device const half *wv [[buffer(10)]], device const half *table [[buffer(11)]],                       \
        device half *ck [[buffer(12)]], device half *cv [[buffer(13)]], device half *hk [[buffer(14)]],      \
        device half *hv [[buffer(15)]], device half *qbuf [[buffer(16)]], device half *rbuf [[buffer(17)]],  \
        device half *ungated [[buffer(18)]], device half *output [[buffer(19)]],                             \
        device float *partials [[buffer(20)]], device const half *qscale [[buffer(21)]],                     \
        device const half *rscale [[buffer(22)]]
#define IDS                                                                                                  \
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],                   \
        uint sg [[simdgroup_index_in_threadgroup]]

inline float4 read_vector(device const half *src, uint dim, uint lane) {
    float4 val(0);
    for (uint j = 0; j < 4; ++j) {
        uint i = lane + 32 * j;
        if (i < dim)
            val[j] = float(src[i]);
    }
    return val;
}
inline void write_vector(device half *dst, uint dim, uint lane, float4 val) {
    for (uint j = 0; j < 4; ++j) {
        uint i = lane + 32 * j;
        if (i < dim)
            dst[i] = half(val[j]);
    }
}
inline float4 round16(float4 x) { return float4(half4(x)); }
inline float4 clip_vector(float4 x, float threshold) {
    float norm2 = simd_sum(dot(x, x));
    return x * min(1.0f, threshold / (sqrt(norm2) + 1.0e-6f));
}
inline float4 short_conv(constant Params &p, device const half *now, device const half *taps,
                         device const half *delta, device half *history, uint bh, uint lane) {
    uint h = bh % p.kv;
    float4 x = read_vector(now + bh * p.dim, p.dim, lane);
    float4 sum = x;
    for (uint lag = 0; lag < 4; ++lag) {
        float4 raw = lag ? read_vector(history + (bh * 3 + lag - 1) * p.dim, p.dim, lane) : x;
        float4 tap = read_vector(taps + (lag * p.kv + h) * p.dim, p.dim, lane);
        if (p.dynamic)
            tap += float(delta[bh * 4 + lag]);
        sum += tap * raw;
    }
    for (uint lag = 2; lag > 0; --lag) {
        float4 prior = read_vector(history + (bh * 3 + lag - 1) * p.dim, p.dim, lane);
        write_vector(history + (bh * 3 + lag) * p.dim, p.dim, lane, prior);
    }
    write_vector(history + bh * 3 * p.dim, p.dim, lane, x);
    return sum * (p.dynamic ? 1.0f / 3.0f : 0.44721359549995794f);
}
inline void prepare_kv(constant Params &p, device const half *k, device const half *v, device const half *wk,
                       device const half *wv, device const half *dk, device const half *dv, device half *hk,
                       device half *hv, device half *ck, device half *cv, uint bh, uint lane,
                       threadgroup half *current, bool keepCurrent) {
    float4 kval = round16(short_conv(p, k, wk, dk, hk, bh, lane));
    float4 vval = round16(short_conv(p, v, wv, dv, hv, bh, lane));
    kval = clip_vector(kval, 22.627416998f);
    vval *= rsqrt(simd_sum(dot(vval, vval)) / float(p.dim) + 1.0e-6f);
    uint slot = (bh * p.capacity + p.position % p.capacity) * p.dim;
    write_vector(ck + slot, p.dim, lane, kval);
    write_vector(cv + slot, p.dim, lane, vval);
    if (keepCurrent) {
        for (uint j = 0; j < 4; ++j) {
            uint i = lane + 32 * j;
            if (i < p.dim) {
                current[i] = half(kval[j]);
                current[p.dim + i] = half(vval[j]);
            }
        }
    }
}
inline uint first_pos(constant Params &p) {
    return p.window && p.position + 1 > p.window ? p.position + 1 - p.window : 0u;
}
struct Online {
    float maximum;
    float denominator;
    float4 weighted;
};
// The KV group loads an eight-row tile once for its query heads. In the fused
// schedule the current row comes directly from the rounded on-chip handoff.
#define TILE_ROWS 8u
inline Online scan_range(constant Params &p, device const half *ck, device const half *cv,
                         device const half *table, float4 query, float rel, uint bh, uint sg, uint lane,
                         uint start, uint stop, float initial_max, float initial_sum,
                         threadgroup half *tile_k, threadgroup half *tile_v, threadgroup half *current,
                         bool useCurrent) {
    Online a = {initial_max, initial_sum, float4(0)};
    for (uint base = start; base < stop; base += TILE_ROWS) {
        uint rows = min(TILE_ROWS, stop - base);
        for (uint idx = sg * 32 + lane; idx < rows * p.dim; idx += 32 * (p.queries / p.kv)) {
            uint pos = base + idx / p.dim, channel = idx % p.dim;
            uint offset = (bh * p.capacity + pos % p.capacity) * p.dim + channel;
            bool useHandoff = useCurrent && pos == p.position;
            tile_k[idx] = useHandoff ? current[channel] : ck[offset];
            tile_v[idx] = useHandoff ? current[p.dim + channel] : cv[offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint row = 0; row < rows; ++row) {
            float4 key(0), value(0);
            for (uint j = 0; j < 4; ++j) {
                uint ch = lane + 32 * j;
                if (ch < p.dim) {
                    key[j] = float(tile_k[row * p.dim + ch]);
                    value[j] = float(tile_v[row * p.dim + ch]);
                }
            }
            uint lag = p.position - (base + row);
            float rb = 0;
            if (lag < p.extent) {
                float term = lane < p.relative_dim ? rel * float(table[lag * p.relative_dim + lane]) : 0;
                rb = float(half(simd_sum(term)));
            }
            float z = simd_sum(dot(query, key)) * rsqrt(float(p.dim)) + rb;
            float m = max(z, a.maximum), old = exp(a.maximum - m), w = exp(z - m);
            a.weighted = a.weighted * old + value * w;
            a.denominator = a.denominator * old + w;
            a.maximum = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return a;
}
inline uint query_index(constant Params &p, uint bh, uint sg) { return bh * (p.queries / p.kv) + sg; }
inline float4 gated(constant Params &p, device const half *gate, uint head, uint lane, float4 y) {
    return round16(y) / (1.0f + exp(-read_vector(gate + head * p.dim, p.dim, lane)));
}
inline float4 prepare_query(device const half *raw, device const half *scale, uint head, uint qheads,
                            uint dim, uint lane, float clip, constant Params &p) {
    float4 x =
        read_vector(raw + head * dim, dim, lane) * read_vector(scale + (head % qheads) * dim, dim, lane);
    x = round16(clip_vector(x, clip));
    float tau = p.dynamic ? 1.0f + 0.1f * log(max(1.0f, float(p.position + 1) / 8192.0f)) : 1.0f;
    return round16(x * tau);
}
kernel void prepare(INPUTS, IDS) {
    threadgroup half unused[1];
    uint bh = group.x, head = query_index(p, bh, sg);
    if (sg == 0)
        prepare_kv(p, k, v, wk, wv, dk, dv, hk, hv, ck, cv, bh, lane, unused, false);
    write_vector(qbuf + head * p.dim, p.dim, lane,
                 prepare_query(q, qscale, head, p.queries, p.dim, lane, 22.627416998f, p));
    write_vector(rbuf + head * p.relative_dim, p.relative_dim, lane,
                 prepare_query(r, rscale, head, p.queries, p.relative_dim, lane, 8.0f, p));
}
kernel void single_decode(INPUTS, IDS) {
    threadgroup half tile_k[TILE_ROWS * 128], tile_v[TILE_ROWS * 128], current[2 * 128];
    uint bh = group.x, head = query_index(p, bh, sg);
    if (sg == 0)
        prepare_kv(p, k, v, wk, wv, dk, dv, hk, hv, ck, cv, bh, lane, current, true);
    float4 query = prepare_query(q, qscale, head, p.queries, p.dim, lane, 22.627416998f, p);
    float4 rr = prepare_query(r, rscale, head, p.queries, p.relative_dim, lane, 8.0f, p);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Online a = scan_range(p, ck, cv, table, query, rr.x, bh, sg, lane, first_pos(p), p.position + 1,
                          float(sink[head]), 1.0f, tile_k, tile_v, current, true);
    write_vector(output + head * p.dim, p.dim, lane, gated(p, gate, head, lane, a.weighted / a.denominator));
}
kernel void scan_staged(INPUTS, IDS) {
    threadgroup half tile_k[TILE_ROWS * 128], tile_v[TILE_ROWS * 128];
    uint bh = group.x, head = query_index(p, bh, sg);
    float4 query = read_vector(qbuf + head * p.dim, p.dim, lane);
    float rel = lane < p.relative_dim ? float(rbuf[head * p.relative_dim + lane]) : 0;
    Online a = scan_range(p, ck, cv, table, query, rel, bh, sg, lane, first_pos(p), p.position + 1,
                          float(sink[head]), 1.0f, tile_k, tile_v, tile_k, false);
    for (uint j = 0; j < 4; ++j)
        if (lane + j * 32 < p.dim)
            ungated[head * p.dim + lane + j * 32] = a.weighted[j] / a.denominator;
}
kernel void gate_staged(INPUTS, IDS) {
    uint head = query_index(p, group.x, sg);
    float4 y(0);
    for (uint j = 0; j < 4; ++j)
        if (lane + j * 32 < p.dim)
            y[j] = ungated[head * p.dim + lane + j * 32];
    write_vector(output + head * p.dim, p.dim, lane, gated(p, gate, head, lane, y));
}
kernel void scan_tile(INPUTS, IDS) {
    threadgroup half tile_k[TILE_ROWS * 128], tile_v[TILE_ROWS * 128];
    uint bh = group.x, head = query_index(p, bh, sg);
    uint start = first_pos(p) + group.y * p.tile;
    uint stop = min(start + p.tile, p.position + 1);
    float4 query = read_vector(qbuf + head * p.dim, p.dim, lane);
    float rel = lane < p.relative_dim ? float(rbuf[head * p.relative_dim + lane]) : 0;
    Online a = scan_range(p, ck, cv, table, query, rel, bh, sg, lane, start, stop, -INFINITY, 0.0f, tile_k,
                          tile_v, tile_k, false);
    device float *part = partials + (head * p.chunks + group.y) * (p.dim + 2);
    if (!lane) {
        part[0] = a.maximum;
        part[1] = a.denominator;
    }
    for (uint j = 0; j < 4; ++j)
        if (lane + 32 * j < p.dim)
            part[2 + lane + 32 * j] = a.weighted[j];
}
kernel void merge_tiles(INPUTS, IDS) {
    uint head = query_index(p, group.x, sg);
    Online a = {float(sink[head]), 1.0f, float4(0)};
    for (uint tile = 0; tile < p.chunks; ++tile) {
        device const float *part = partials + (head * p.chunks + tile) * (p.dim + 2);
        float m = max(a.maximum, part[0]), old = exp(a.maximum - m), w = exp(part[0] - m);
        a.denominator = a.denominator * old + part[1] * w;
        float4 y(0);
        for (uint j = 0; j < 4; ++j)
            if (lane + 32 * j < p.dim)
                y[j] = part[2 + lane + 32 * j];
        a.weighted = a.weighted * old + y * w;
        a.maximum = m;
    }
    for (uint j = 0; j < 4; ++j)
        if (lane + j * 32 < p.dim)
            ungated[head * p.dim + lane + j * 32] = half(a.weighted[j] / a.denominator);
}
