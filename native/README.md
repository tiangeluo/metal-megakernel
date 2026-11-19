# Native attention schedules

The input is one token of projected features, FP16 weights and a prefilled cache. The public Python array contract is defined by `reference.attention.step`. The Objective-C++ bridge compiles the shared Metal shader at runtime and keeps pipelines, input, output, scratch and cache buffers allocated for the lifetime of each runner.

A threadgroup owns one sequence and one KV head. Its SIMD groups each own one query head. With six query heads per KV head this uses 192 threads. Each group stages up to eight historical K/V positions at a time in threadgroup memory. Each SIMD group maintains an FP32 online softmax maximum, denominator and value accumulator for its query head.

| Schedule | Dispatches per step | Data handoff |
|---|---:|---|
| Staged | 3 | Prepare clipped Q/R and rounded K/V, append cache and update histories. Scan KV in chronological tiles. Read the rounded head averages and apply the output gate. |
| Single | 1 | Use the same arithmetic. Forward the two rounded current K/V vectors through threadgroup memory to all query heads, while appending them to the cache. Keep prepared Q/R and softmax accumulators in registers and apply the gate before the final store. |
| Split | 4 | Prepare, scan independent 32-position ranges, then merge maxima, denominators and vector numerators. The merge introduces the sink exactly once. The same separate output-gate dispatch as staged follows the merge. The split length and the eight-position on-chip tile are separate parameters. |

At head dimension 128 the KV tile occupies 4,096 bytes. The single-dispatch schedule also stages 512 bytes for the newest K/V pair. These figures describe explicitly declared threadgroup arrays. They do not measure total shader resource use or device-memory traffic.

From the repository root:

```sh
python -m native.runner --build-only
python -m unittest native.test_runner -v
python -m native.runner --length 64 --samples 20
```

The first command verifies the CPU bridge build. The tests require a visible Metal device and compile and execute the shader. They check batched GQA, relative and local boundaries, extreme sink values, FP16 output agreement and cache updates across consecutive steps. The last command prints timing samples only after passing validation against NumPy. Use the project's interleaved benchmark harness for the published measurements.

`NativeRunner.run()` synchronizes the command buffer and returns its encoding-to-completion wall time, optional GPU interval and encoded dispatch count. `reset()`, `set_token()`, `set_position()`, and `outputs()` stay outside isolated timing. The stack adapter can bind input and output slices of existing Metal buffers using the DLPack handoff. The caller keeps the exporting arrays alive and synchronizes producer and consumer work. Each native command buffer waits for completion, so that synchronization is charged in stack latency.

A machine that cannot expose the GPU produces an explicit device-acquisition error. A shader compilation error, parity failure or API validation failure stops the run.
