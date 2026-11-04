"""Bandwidth of the skinny-GEMM ops: REP independent nodes over distinct weights (cold L2) in one ORT run, timed with CUDA events.

usage: BOLT_PLUGIN=... BOLT_OPS=... python tools/bench_gemv.py [R]
"""
import os, sys, torch, numpy as np, onnxruntime as ort
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
from test_bolt_ops import session, vi, BF, F32, U8, I32, helper

R = int(sys.argv[1]) if len(sys.argv) > 1 else 8
REP = 16
CASES = [("Nvfp4Gemv", 18560, 4096), ("Nvfp4Gemv", 4096, 8192), ("Q4Gemv", 5376, 4096), ("Q8Gemv", 1024, 4096), ("Q8Gemv", 131072, 4096), ("Q8Gemv", 4096, 4096)]


def build(op, N, K):
    inputs, feeds, nodes, outs = [("x", BF, [R, K])], {"x": (torch.randn(R, K) * 0.5).bfloat16()}, [], []
    for r in range(REP):
        if op == "Nvfp4Gemv":
            w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8); s = torch.randint(0x30, 0x48, (N, K // 16), dtype=torch.uint8)
            ins = [("x"), f"w{r}", f"s{r}", f"g{r}"]
            for n, t in ((f"w{r}", w), (f"s{r}", s), (f"g{r}", torch.tensor([0.37]))):
                inputs.append((n, {torch.uint8: U8, torch.float32: F32}[t.dtype], list(t.shape))); feeds[n] = t
        else:
            nb = K // 32; bits = 8 if op == "Q8Gemv" else 4
            w = torch.randint(0, 256, (N, nb, 32 if bits == 8 else 16), dtype=torch.uint8); s = (torch.rand(N, nb) * 0.02).bfloat16()
            ins = ["x", f"w{r}", f"s{r}"]
            for n, t in ((f"w{r}", w), (f"s{r}", s)):
                inputs.append((n, {torch.uint8: U8, torch.bfloat16: BF}[t.dtype], list(t.shape))); feeds[n] = t
        nodes.append(helper.make_node(op, ins, [f"y{r}"], domain="bolt")); outs.append(f"y{r}")
    g = helper.make_graph(nodes, "b", [vi(n, t, s) for n, t, s in inputs], [vi(o, BF, None) for o in outs])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("bolt", 1)]); m.ir_version = 10
    return m, feeds, outs


for op, N, K in CASES:
    m, feeds, outs = build(op, N, K)
    sess = session(m)
    b = sess.io_binding(); keep = []
    for n, t in feeds.items():
        ov = ort.OrtValue.from_dlpack(t.cuda().contiguous().__dlpack__()); keep.append(ov); b.bind_ortvalue_input(n, ov)
    for o in outs: b.bind_output(o, "cuda", 0)
    nbytes = sum(t.numel() * t.element_size() for n, t in feeds.items() if n != "x") / REP
    for _ in range(3): sess.run_with_iobinding(b)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(5): sess.run_with_iobinding(b)
    e1.record(); torch.cuda.synchronize()
    us = e0.elapsed_time(e1) / 5 / REP * 1e3
    print(f"{op:10s} R={R} N={N:6d} K={K}: {us:7.1f} us/call, {nbytes / us / 1e3:6.0f} GB/s ({nbytes/1e6:.1f} MB)")
