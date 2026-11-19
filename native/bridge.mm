// CPU-side Metal runtime. This C ABI is deliberately limited to shared, preallocated
// buffers; data preparation and readback are separate from the timed command buffer.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

struct Config {
    uint32_t batch, queries, kv, dim, relativeDim, extent, capacity, position;
    uint32_t window, dynamic, tile, chunks;
};
static_assert(sizeof(Config) == 48);
struct Metrics {
    double wallUs, gpuUs;
    uint32_t dispatches;
};

struct Runner {
    Config p;
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    std::array<id<MTLBuffer>, 23> buf{}, external{};
    std::array<size_t, 23> offsets{};
    std::array<id<MTLComputePipelineState>, 6> pipelines{};
};
static thread_local std::string lastError;
static const char *names[] = {"prepare",       "scan_staged", "gate_staged",
                              "single_decode", "scan_tile",   "merge_tiles"};
extern "C" const char *mm_error() { return lastError.c_str(); }
static bool fail(NSString *message) {
    lastError = message.UTF8String ?: "unspecified Metal error";
    return false;
}
extern "C" void *mm_open(const char *source, const Config *p, const void **data,
                         const size_t *sizes) {
    @autoreleasepool {
        lastError.clear();
        auto ctx = std::make_unique<Runner>();
        ctx->p = *p;
        ctx->dev = MTLCreateSystemDefaultDevice();
        if (!ctx->dev) {
            fail(@"MTLCreateSystemDefaultDevice returned nil. Run in a GPU-visible macOS session.");
            return nullptr;
        }
        ctx->queue = [ctx->dev newCommandQueue];
        if (!ctx->queue) {
            fail(@"Could not create command queue");
            return nullptr;
        }
        NSError *err = nil;
        MTLCompileOptions *opts = [MTLCompileOptions new];
        if (@available(macOS 15.0, *))
            opts.mathMode = MTLMathModeSafe;
        else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            opts.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        id<MTLLibrary> lib = [ctx->dev newLibraryWithSource:[NSString stringWithUTF8String:source]
                                                    options:opts
                                                      error:&err];
        if (!lib) {
            fail(err.localizedDescription ?: @"Could not compile Metal source");
            return nullptr;
        }
        for (size_t i = 0; i < ctx->pipelines.size(); ++i) {
            id<MTLFunction> fun =
                [lib newFunctionWithName:[NSString stringWithUTF8String:names[i]]];
            if (!fun) {
                fail(@"Compiled Metal function was missing");
                return nullptr;
            }
            ctx->pipelines[i] = [ctx->dev newComputePipelineStateWithFunction:fun error:&err];
            if (!ctx->pipelines[i]) {
                fail(err.localizedDescription ?: @"Could not create compute pipeline");
                return nullptr;
            }
            if (ctx->pipelines[i].threadExecutionWidth != 32 ||
                (p->queries / p->kv) * 32 > ctx->pipelines[i].maxTotalThreadsPerThreadgroup) {
                fail(@"Head group exceeds pipeline limits, or device SIMD width differs from 32");
                return nullptr;
            }
        }
        for (size_t i = 1; i < ctx->buf.size(); ++i) {
            ctx->buf[i] = [ctx->dev newBufferWithLength:sizes[i]
                                                options:MTLResourceStorageModeShared];
            if (!ctx->buf[i]) {
                fail(@"MTLBuffer allocation failed");
                return nullptr;
            }
            if (data[i])
                std::memcpy(ctx->buf[i].contents, data[i], sizes[i]);
        }
        return ctx.release();
    }
}
extern "C" int mm_transfer(void *raw, uint32_t idx, void *ptr, size_t bytes, bool write) {
    @autoreleasepool {
        auto &ctx = *static_cast<Runner *>(raw);
        if (!idx || idx >= ctx.buf.size() || ctx.buf[idx].length != bytes)
            return fail(@"Invalid transfer size or binding");
        id<MTLBuffer> buffer = ctx.external[idx] ?: ctx.buf[idx];
        if (buffer.storageMode != MTLStorageModeShared)
            return fail(@"Transfer requires shared storage");
        auto *address = static_cast<char *>(buffer.contents) + ctx.offsets[idx];
        if (write)
            std::memcpy(address, ptr, bytes);
        else
            std::memcpy(ptr, address, bytes);
        return 1;
    }
}
extern "C" void mm_close(void *ctx) {
    @autoreleasepool {
        delete static_cast<Runner *>(ctx);
    }
}
static const char *planNames[] = {"staged", "single", "split"};
extern "C" int mm_run(void *raw, int method, Metrics *metrics) {
    @autoreleasepool {
        auto &ctx = *static_cast<Runner *>(raw);
        if (method < 0 || method >= 3)
            return fail(@"Unknown schedule");
        static const int schedules[3][4] = {{0, 1, 2, -1}, {3, -1, -1, -1}, {0, 4, 5, 2}};
        auto t0 = std::chrono::steady_clock::now();
        id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
        if (!cb)
            return fail(@"Command buffer creation failed");
        cb.label = [NSString stringWithUTF8String:planNames[method]];
        id<MTLComputeCommandEncoder> e =
            [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        if (!e)
            return fail(@"Compute encoder creation failed");
        [e setBytes:&ctx.p length:sizeof(Config) atIndex:0];
        for (size_t i = 1; i < ctx.buf.size(); ++i)
            [e setBuffer:(ctx.external[i] ?: ctx.buf[i]) offset:ctx.offsets[i] atIndex:i];
        uint32_t count = 0;
        for (int ix : schedules[method]) {
            if (ix < 0)
                break;
            if (count)
                [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [e setComputePipelineState:ctx.pipelines[ix]];
            MTLSize grid = MTLSizeMake(ctx.p.batch * ctx.p.kv, ix == 4 ? ctx.p.chunks : 1, 1);
            [e dispatchThreadgroups:grid
                threadsPerThreadgroup:MTLSizeMake((ctx.p.queries / ctx.p.kv) * 32, 1, 1)];
            ++count;
        }
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        auto t1 = std::chrono::steady_clock::now();
        if (cb.status != MTLCommandBufferStatusCompleted)
            return fail(cb.error.localizedDescription ?: @"GPU work did not complete");
        metrics->wallUs = std::chrono::duration<double, std::micro>(t1 - t0).count();
        metrics->gpuUs = cb.GPUStartTime > 0 && cb.GPUEndTime >= cb.GPUStartTime
                             ? (cb.GPUEndTime - cb.GPUStartTime) * 1e6
                             : -1.0;
        metrics->dispatches = count;
        return 1;
    }
}
extern "C" int mm_position(void *raw, uint32_t position) {
    auto &ctx = *static_cast<Runner *>(raw);
    if (position == UINT32_MAX)
        return fail(@"Position exceeds native index range");
    uint32_t visible = ctx.p.window ? std::min(position + 1, ctx.p.window) : position + 1;
    if (visible > ctx.p.capacity)
        return fail(@"KV capacity is too small for this position");
    ctx.p.position = position;
    ctx.p.chunks = (visible + ctx.p.tile - 1) / ctx.p.tile;
    return 1;
}

// Accept only a validated kDLMetal data handle to a live MTLBuffer. The caller
// retains the DLPack owner, and supplies alignment, shape, strides and dtype checks.
extern "C" int mm_bind(void *raw, uint32_t idx, void *metalBuffer, size_t offset) {
    @autoreleasepool {
        auto &ctx = *static_cast<Runner *>(raw);
        if ((idx < 1 || idx > 8) && idx != 19)
            return fail(@"Only token inputs and gated output may be rebound");
        if (!metalBuffer) {
            ctx.external[idx] = nil;
            ctx.offsets[idx] = 0;
            return 1;
        }
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)metalBuffer;
        if (![buffer conformsToProtocol:@protocol(MTLBuffer)] ||
            buffer.device.registryID != ctx.dev.registryID)
            return fail(@"External storage must be a Metal buffer on the same GPU");
        if (offset % 4 != 0 || offset > buffer.length ||
            ctx.buf[idx].length > buffer.length - offset)
            return fail(
                @"External buffer does not cover the tensor or start at a four-byte boundary");
        ctx.external[idx] = buffer;
        ctx.offsets[idx] = offset;
        return 1;
    }
}
