#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
static int report(NSDictionary *data, int status) {
    NSData *json = [NSJSONSerialization dataWithJSONObject:data options:0 error:nil];
    fwrite(json.bytes,1,json.length,stdout);
    putchar('\n');
    return status;
}
int main(void) { @autoreleasepool {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    if(!d) return report(@{@"passed":@NO, @"metal_devices":@(MTLCopyAllDevices().count), @"reason":@"No Metal device visible to this process."},2);
    NSString *src=@"#include <metal_stdlib>\nusing namespace metal;\nkernel void probe(device uint *o [[buffer(0)]], threadgroup uint *m [[threadgroup(0)]], uint t [[thread_index_in_threadgroup]]){m[t]=t+1;m[5504+t]=t;threadgroup_barrier(mem_flags::mem_threadgroup);o[t]=m[(t+1)%128]+m[5504+t];}";
    NSError *err=nil;
    id<MTLLibrary> lib=[d newLibraryWithSource:src options:nil error:&err];
    id<MTLComputePipelineState> ps=lib?[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"probe"] error:&err]:nil;
    if(!ps) return report(@{@"passed":@NO,@"device":d.name,@"reason":err.localizedDescription?:@"Pipeline creation failed."},3);
    if(d.maxThreadgroupMemoryLength<22528||ps.maxTotalThreadsPerThreadgroup<128)
        return report(@{@"passed":@NO,@"device":d.name,@"reason":@"Insufficient threadgroup resources for probe."},4);
    id<MTLCommandQueue> q=[d newCommandQueue]; id<MTLCommandBuffer> cb=[q commandBuffer];
    id<MTLBuffer> buf=[d newBufferWithLength:512 options:MTLResourceStorageModeShared];
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e setComputePipelineState:ps]; [e setBuffer:buf offset:0 atIndex:0];
    [e setThreadgroupMemoryLength:22528 atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted) return report(@{@"passed":@NO,@"device":d.name,@"reason":cb.error.localizedDescription?:@"Command buffer failed."},5);
    const uint32_t *vals=buf.contents;
    for(uint32_t i=0;i<128;i++) if(vals[i]!=(i+1)%128+1+i)
        return report(@{@"passed":@NO,@"device":d.name,@"reason":@"Incorrect result from Metal."},6);
    return report(@{@"passed":@YES,@"device":d.name,@"verified_outputs":@128,
        @"threadgroup_bytes_used":@22528,@"threadgroup_memory_limit":@(d.maxThreadgroupMemoryLength),
        @"execution_width":@(ps.threadExecutionWidth),@"max_threads_per_group":@(ps.maxTotalThreadsPerThreadgroup),
        @"apple9":[NSNumber numberWithBool:[d supportsFamily:MTLGPUFamilyApple9]],
        @"thermal_state":@([NSProcessInfo processInfo].thermalState)},0);
} }
