"""Unit tests of the bolt custom ops against torch references.

usage: BOLT_PLUGIN=<cuda plugin ep .so> BOLT_OPS=<libbolt_ops.so> python tests/test_bolt_ops.py
"""
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto as TP
from onnx import helper

BF, F32, U8, I32, I64 = TP.BFLOAT16, TP.FLOAT, TP.UINT8, TP.INT32, TP.INT64
DEV = "cuda:0"
_registered = False


def session(model):
    global _registered
    plugin = os.environ["BOLT_PLUGIN"]
    if not _registered:
        ort.register_execution_provider_library("CUDAExecutionProvider", plugin)
        _registered = True
    so = ort.SessionOptions()
    so.register_custom_ops_library(os.environ["BOLT_OPS"])
    devs = [d for d in ort.get_ep_devices() if d.ep_name == "CUDAExecutionProvider" and d.ep_metadata.get("cuda_device_id") == "0"]
    so.add_provider_for_devices(devs, {})
    return ort.InferenceSession(model.SerializeToString(), sess_options=so)


def vi(n, t, shape):
    return helper.make_tensor_value_info(n, t, shape)


def run(op, inputs, out_types, attrs=None, domain="bolt"):
    """inputs: list of (name, torch tensor on cuda or None for omitted optional input); returns list of torch tensors."""
    m = {torch.bfloat16: BF, torch.float32: F32, torch.uint8: U8, torch.int32: I32, torch.int64: I64}
    outs = [f"o{i}" for i in range(len(out_types))]
    node = helper.make_node(op, ["" if t is None else n for n, t in inputs], outs, domain=domain, **(attrs or {}))
    g = helper.make_graph([node], "t", [vi(n, m[t.dtype], list(t.shape)) for n, t in inputs if t is not None],
                          [vi(o, ot, None) for o, ot in zip(outs, out_types)])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid(domain, 1)])
    model.ir_version = 10
    sess = session(model)
    b = sess.io_binding()
    keep = []
    for n, t in inputs:
        if t is None:
            continue
        ov = ort.OrtValue.from_dlpack(t.contiguous().__dlpack__())
        keep.append(ov)
        b.bind_ortvalue_input(n, ov)
    for o in outs:
        b.bind_output(o, "cuda", 0)
    sess.run_with_iobinding(b)
    b.synchronize_outputs()
    return [torch.from_dlpack(o) for o in b.get_outputs()]


def rel(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


FP4 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


def test_nvfp4(R, N, K, XSCALE=2.0):
    torch.manual_seed(1)
    w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8)
    ws = torch.randint(0x30, 0x48, (N, K // 16), dtype=torch.uint8)  # e4m3 bytes, positive small range
    g = torch.tensor([0.37])
    x = torch.randn(R, K).bfloat16() * XSCALE
    deq = torch.stack([FP4[(w & 15).long()], FP4[(w >> 4).long()]], -1).reshape(N, K)
    s = ws.view(torch.float8_e4m3fn).float().repeat_interleave(16, 1)
    ref = x.float() @ (deq * s * g).T
    (y,) = run("Nvfp4Gemv", [("x", x.to(DEV)), ("w", w.to(DEV)), ("ws", ws.to(DEV)), ("g", g.to(DEV))], [BF])
    print(f"Nvfp4Gemv R={R} N={N} K={K} |x|~{XSCALE}: rel err {rel(y, ref):.2e}")
    assert rel(y, ref) < 1e-2


def test_qgemv(bits, R, N, K):
    torch.manual_seed(2)
    nb = K // 32
    if bits == 8:
        q = torch.randint(0, 256, (N, nb, 32), dtype=torch.uint8)
        deq = q.float() - 128
    else:
        q = torch.randint(0, 256, (N, nb, 16), dtype=torch.uint8)
        deq = torch.stack([(q & 15).float(), (q >> 4).float()], -1).reshape(N, nb, 32) - 8
    sc = (torch.rand(N, nb) * 0.02).bfloat16()
    x = torch.randn(R, K).bfloat16()
    ref = x.float() @ (deq * sc.float().unsqueeze(-1)).reshape(N, K).T
    (y,) = run("Q8Gemv" if bits == 8 else "Q4Gemv", [("x", x.to(DEV)), ("w", q.to(DEV)), ("s", sc.to(DEV))], [BF])
    print(f"Q{bits}Gemv R={R} N={N} K={K}: rel err {rel(y, ref):.2e}")
    assert rel(y, ref) < 1e-2


def test_relu2():
    x = torch.randn(3, 1000).bfloat16() * 3
    (y,) = run("Relu2", [("x", x.to(DEV))], [BF])
    ref = (torch.relu(x).float() ** 2).bfloat16()
    print("Relu2: max diff", (y.cpu().float() - ref.float()).abs().max().item())
    assert torch.equal(y.cpu(), ref)


def test_route(R, E, topk=9, K=4096):
    torch.manual_seed(3)
    x = torch.randn(R, K).bfloat16()
    w = (torch.randn(E, K) * 0.05).bfloat16()
    b = torch.randn(E) * 0.1
    idx, wt, _ = run("MoeRoute", [("x", x.to(DEV)), ("w", w.to(DEV)), ("b", b.to(DEV))], [I32, F32, F32],
                     {"topk": topk, "scale": 5.0})
    sc = (x.float() @ w.float().T).sigmoid()
    ridx = torch.topk(sc + b, topk, dim=-1)[1]
    rw = sc.gather(1, ridx)
    rw = rw / (rw.sum(-1, keepdim=True) + 1e-20) * 5.0
    same = all(set(idx[r].cpu().tolist()) == set(ridx[r].tolist()) for r in range(R))
    # compare weights per expert
    got = torch.zeros(R, E); ref = torch.zeros(R, E)
    got.scatter_(1, idx.cpu().long(), wt.cpu()); ref.scatter_(1, ridx, rw)
    print(f"MoeRoute R={R} E={E}: same experts {same}, weight rel err {rel(got, ref):.2e}")
    assert same and rel(got, ref) < 1e-3


def test_moe_decode(R, E, I, topk=9, op="MoeDecode"):
    torch.manual_seed(4)
    z = (torch.randn(R, 1024) * 0.5).bfloat16()
    idx = torch.stack([torch.randperm(E)[:topk] for _ in range(R)]).int()
    w = torch.rand(R, topk)
    w1 = torch.randint(0, 256, (E, I, 512), dtype=torch.uint8)
    w2 = torch.randint(0, 256, (E, 1024, I // 2), dtype=torch.uint8)
    s1 = (torch.rand(E, I, 32) * 0.003).bfloat16()
    s2 = (torch.rand(E, 1024, I // 32) * 0.003).bfloat16()

    def deq(q, s, n, k):
        v = torch.stack([(q & 15).float(), (q >> 4).float()], -1).reshape(q.shape[0], n, k // 32, 32) - 8
        return (v * s.float().unsqueeze(-1)).reshape(q.shape[0], n, k)

    d1, d2 = deq(w1, s1, I, 1024), deq(w2, s2, 1024, I)
    ref = torch.zeros(R, 1024)
    for t in range(R):
        for j in range(topk):
            e = idx[t, j]
            h = (z[t].float() @ d1[e].T).bfloat16().float()
            h = (torch.relu(h) ** 2).bfloat16().float()
            ref[t] += w[t, j] * (h @ d2[e].T).bfloat16().float()
    out = run(op, [("z", z.to(DEV)), ("i", idx.to(DEV)), ("w", w.to(DEV)), ("w1", w1.to(DEV)), ("s1", s1.to(DEV)),
                   ("w2", w2.to(DEV)), ("s2", s2.to(DEV))], [BF, BF, F32] + ([I32] if op == "MoeVerify" else []), {"topk": topk})[0]
    print(f"{op} R={R} E={E} I={I}: rel err {rel(out, ref):.2e}")
    assert rel(out, ref) < 1e-2


IQ2_VALUES = torch.tensor([8, 25, 43, 0, -8, -25, -43, 0], dtype=torch.float32)


def iq2_codes(*shape):
    mag = torch.randint(0, 3, shape)
    sign = torch.randint(0, 2, shape)
    return (mag + 4 * sign).to(torch.uint8)


def pack_nibbles(code):
    return (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous()


def test_q4_iq2(R, N, K):
    torch.manual_seed(11)
    nb = K // 32
    code = iq2_codes(N, K)
    q = pack_nibbles(code).reshape(N, nb, 16)
    sc = (torch.rand(N, nb) * 0.02).bfloat16()
    x = torch.randn(R, K).bfloat16()
    deq = IQ2_VALUES[code.long()].reshape(N, nb, 32) * sc.float().unsqueeze(-1)
    ref = x.float() @ deq.reshape(N, K).T
    (y,) = run("Q4Gemv", [("x", x.to(DEV)), ("w", q.to(DEV)), ("s", sc.to(DEV))], [BF], {"iq2": 1})
    print(f"Q4Gemv iq2 R={R} N={N} K={K}: rel err {rel(y, ref):.2e}")
    assert rel(y, ref) < 6e-3


def test_moe_iq2(R, E, I, topk=9):
    torch.manual_seed(12)
    z = (torch.randn(R, 1024) * 0.5).bfloat16()
    idx = torch.stack([torch.randperm(E)[:topk] for _ in range(R)]).int()
    w = torch.rand(R, topk)
    c1, c2 = iq2_codes(E, I, 1024), iq2_codes(E, 1024, I)
    w1, w2 = pack_nibbles(c1), pack_nibbles(c2)
    s1 = (torch.rand(E, I, 32) * 0.003).bfloat16()
    s2 = (torch.rand(E, 1024, I // 32) * 0.003).bfloat16()

    def deq(c, s, n, k):
        return (IQ2_VALUES[c.long()].reshape(c.shape[0], n, k // 32, 32) * s.float().unsqueeze(-1)).reshape(c.shape[0], n, k)

    d1, d2 = deq(c1, s1, I, 1024), deq(c2, s2, 1024, I)
    ref = torch.zeros(R, 1024)
    for t in range(R):
        for j in range(topk):
            e = idx[t, j]
            h = (z[t].float() @ d1[e].T).bfloat16().float()
            h = (torch.relu(h) ** 2).bfloat16().float()
            ref[t] += w[t, j] * (h @ d2[e].T).bfloat16().float()
    out = run("MoeVerify", [("z", z.to(DEV)), ("i", idx.to(DEV)), ("w", w.to(DEV)), ("w1", w1.to(DEV)), ("s1", s1.to(DEV)),
                            ("w2", w2.to(DEV)), ("s2", s2.to(DEV))], [BF, BF, F32, I32], {"topk": topk, "iq2": 1})[0]
    print(f"MoeVerify iq2 R={R} E={E} I={I}: rel err {rel(out, ref):.2e}")
    assert rel(out, ref) < 1e-2


def iq2_native_blocks(*lead, k):
    """Random valid ggml IQ2_XXS blocks [*lead, k/256, 66] (any grid / sign index / scale nibble, finite fp16 d)."""
    blk = torch.randint(0, 256, (*lead, k // 256, 66), dtype=torch.uint8)
    d = (torch.rand(*lead, k // 256) * 0.02 + 0.001).half().view(torch.uint8).reshape(*lead, k // 256, 2)
    blk[..., :2] = d
    return blk


def iq2_repack(blk):
    """[*lead, K/256, 66] ggml blocks -> (records uint8 [*lead, K/32, 8], fp16 d bits as a bf16-typed tensor [*lead, K/256])."""
    rec = blk[..., 2:].reshape(*blk.shape[:-2], blk.shape[-2] * 8, 8).contiguous()
    d = blk[..., :2].contiguous().view(torch.int16).reshape(*blk.shape[:-1][:-0 or None]) if False else blk[..., :2].contiguous().view(torch.int16).squeeze(-1)
    return rec, d.view(torch.bfloat16)


def iq2_native_dequant(blk):
    from gguf import GGMLQuantizationType as Q
    from gguf import quants

    flat = blk.reshape(-1, blk.shape[-2] * 66).numpy()
    return torch.tensor(quants.dequantize(flat, Q.IQ2_XXS)).reshape(*blk.shape[:-2], blk.shape[-2] * 256)


def test_q4_iq2_native(R, N, K):
    torch.manual_seed(21)
    blk = iq2_native_blocks(N, k=K)
    x = torch.randn(R, K).bfloat16()
    deq = iq2_native_dequant(blk).bfloat16().float()  # the kernel rounds the weight (bf16 block scale * exact magnitude) to bf16
    ref = x.float() @ deq.T
    wb, db = iq2_repack(blk)
    (y,) = run("Q4Gemv", [("x", x.to(DEV)), ("w", wb.to(DEV)), ("s", db.to(DEV))], [BF], {"iq2": 2})
    (y_gguf,) = run("Q4Gemv", [("x", x.to(DEV)), ("w", blk.reshape(N, -1).to(DEV))], [BF], {"iq2": 3})
    print(f"Q4Gemv iq2 native/gguf R={R} N={N} K={K}: rel err {rel(y_gguf, ref):.2e}, vs_repack={rel(y, y_gguf):.2e}")
    assert rel(y, ref) < 6e-3 and rel(y_gguf, ref) < 6e-3 and rel(y, y_gguf) < 2e-4


def test_moe_iq2_native(R, E, I, topk=9):
    torch.manual_seed(22)
    z = (torch.randn(R, 1024) * 0.5).bfloat16()
    idx = torch.stack([torch.randperm(E)[:topk] for _ in range(R)]).int()
    w = torch.rand(R, topk)
    b1, b2 = iq2_native_blocks(E, I, k=1024), iq2_native_blocks(E, 1024, k=I)
    d1, d2 = iq2_native_dequant(b1).bfloat16().float(), iq2_native_dequant(b2).bfloat16().float()
    ref = torch.zeros(R, 1024)
    for t in range(R):
        for j in range(topk):
            e = idx[t, j]
            h = (z[t].float() @ d1[e].T).bfloat16().float()
            h = (torch.relu(h) ** 2).bfloat16().float()
            ref[t] += w[t, j] * (h @ d2[e].T).bfloat16().float()
    (w1, s1), (w2, s2) = iq2_repack(b1), iq2_repack(b2)
    out = run("MoeVerify", [("z", z.to(DEV)), ("i", idx.to(DEV)), ("w", w.to(DEV)), ("w1", w1.to(DEV)), ("s1", s1.to(DEV)),
                            ("w2", w2.to(DEV)), ("s2", s2.to(DEV))], [BF, BF, F32, I32], {"topk": topk, "iq2": 2})[0]
    out_gguf = run("MoeVerify", [("z", z.to(DEV)), ("i", idx.to(DEV)), ("w", w.to(DEV)),
                                 ("w1", b1.reshape(E, I, -1).to(DEV)), ("s1", None),
                                 ("w2", b2.reshape(E, 1024, -1).to(DEV))], [BF, BF, F32, I32], {"topk": topk, "iq2": 3})[0]
    print(f"MoeVerify iq2 native/gguf R={R} E={E} I={I}: rel err {rel(out_gguf, ref):.2e}, vs_repack={rel(out, out_gguf):.2e}")
    assert rel(out, ref) < 1e-2 and rel(out_gguf, ref) < 1e-2 and rel(out, out_gguf) < 5e-4


def test_gguf_ops():
    torch.manual_seed(31)
    # 1. Nvfp4Gemv gguf=1 vs modelopt
    for R, N, K in ((8, 18560, 4096), (8, 4096, 8192), (1, 4096, 4096)):
        blk = torch.randint(0, 256, (N, K // 64, 36), dtype=torch.uint8)
        blk[:, :, :4] = torch.randint(0x30, 0x48, (N, K // 64, 4), dtype=torch.uint8)
        ws = blk[:, :, :4].reshape(N, K // 16).contiguous()
        qs = blk[:, :, 4:].reshape(N, K // 64, 4, 8)
        elems = torch.cat([qs & 0x0F, qs >> 4], dim=-1)
        w = (elems[..., 0::2] | (elems[..., 1::2] << 4)).reshape(N, K // 2).contiguous()
        g = torch.tensor([0.37], dtype=torch.float32)
        x = torch.randn(R, K).bfloat16()
        (y_gguf,) = run("Nvfp4Gemv", [("x", x.to(DEV)), ("w", blk.reshape(N, -1).to(DEV)), ("ws", None), ("g", g.to(DEV))], [BF], {"gguf": 1})
        (y_ref,) = run("Nvfp4Gemv", [("x", x.to(DEV)), ("w", w.to(DEV)), ("ws", ws.to(DEV)), ("g", g.to(DEV))], [BF])
        if R >= 2:
            assert torch.equal(y_gguf, y_ref), f"Nvfp4Gemv gguf mismatch R={R}"
        else:
            assert rel(y_gguf, y_ref) < 2e-3
    # 2. Q8Gemv gguf=1 vs split Q8
    for R, N, K in ((8, 4096, 4096), (8, 1024, 4096), (1, 1024, 4096)):
        nb = K // 32
        blk = torch.randint(0, 256, (N, nb, 34), dtype=torch.uint8)
        d = (torch.rand(N, nb) * 0.02 + 0.001).half()
        blk[:, :, :2] = d.view(torch.uint8).reshape(N, nb, 2)
        q = (blk[:, :, 2:].view(torch.int8).to(torch.int16) + 128).to(torch.uint8)
        sc = d.to(torch.bfloat16)
        x = torch.randn(R, K).bfloat16()
        (y_gguf,) = run("Q8Gemv", [("x", x.to(DEV)), ("w", blk.reshape(N, -1).to(DEV))], [BF], {"gguf": 1})
        (y_ref,) = run("Q8Gemv", [("x", x.to(DEV)), ("w", q.to(DEV)), ("s", sc.to(DEV))], [BF])
        if R >= 2:
            assert torch.equal(y_gguf, y_ref), f"Q8Gemv gguf mismatch R={R}"
        else:
            assert rel(y_gguf, y_ref) < 5e-3
    # 3. Q8Embed
    V, K = 512, 4096
    nb = K // 32
    blk = torch.randint(0, 256, (V, nb, 34), dtype=torch.uint8)
    d = (torch.rand(V, nb) * 0.02 + 0.001).half()
    blk[:, :, :2] = d.view(torch.uint8).reshape(V, nb, 2)
    ids = torch.randint(0, V, (1, 8), dtype=torch.int64)
    (emb,) = run("Q8Embed", [("w", blk.reshape(V, -1).to(DEV)), ("ids", ids.to(DEV))], [BF])
    ref_emb = (d.float().unsqueeze(-1) * blk[:, :, 2:].view(torch.int8).float()).reshape(V, K).bfloat16()[ids]
    assert torch.equal(emb.cpu(), ref_emb)
    # 4. MoeRouteGGUF, MambaVerifyGGUF, RmsNorm, SkipRmsNorm
    x = torch.randn(8, 4096).bfloat16()
    gw = torch.randn(240, 4096, dtype=torch.float32) * 0.05
    gb = torch.randn(240, dtype=torch.float32) * 0.1
    i1, w1, _ = run("MoeRoute", [("x", x.to(DEV)), ("w", gw.bfloat16().to(DEV)), ("b", gb.to(DEV))], [I32, F32, F32], {"topk": 9, "scale": 5.0})
    i2, w2, _ = run("MoeRouteGGUF", [("x", x.to(DEV)), ("w", gw.to(DEV)), ("b", gb.to(DEV))], [I32, F32, F32], {"topk": 9, "scale": 5.0})
    assert torch.equal(i1, i2) and torch.equal(w1, w2)
    skip = torch.randn(8, 4096).bfloat16()
    gamma = torch.rand(4096, dtype=torch.float32) + 0.5
    (rn,) = run("RmsNorm", [("x", x.to(DEV)), ("g", gamma.to(DEV))], [BF], {"epsilon": 1e-5})
    rn_ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5) * gamma.bfloat16().float()).bfloat16()
    assert rel(rn, rn_ref) < 1e-3
    srn, ssum = run("SkipRmsNorm", [("x", x.to(DEV)), ("s", skip.to(DEV)), ("g", gamma.to(DEV))], [BF, BF], {"epsilon": 1e-5})
    s_ref = (x + skip)
    srn_ref = (s_ref.float() * torch.rsqrt(s_ref.float().pow(2).mean(-1, keepdim=True) + 1e-5) * gamma.bfloat16().float()).bfloat16()
    assert torch.equal(ssum.cpu(), s_ref) and rel(srn, srn_ref) < 1e-3
    # General H=2048 path + optional 2nd output omitted + "gguf" domain
    x2 = torch.randn(4, 2048).bfloat16()
    skip2 = torch.randn(4, 2048).bfloat16()
    gamma2 = torch.rand(2048, dtype=torch.float32) + 0.5
    (rn2,) = run("RmsNorm", [("x", x2.to(DEV)), ("g", gamma2.to(DEV))], [BF], {"epsilon": 1e-5}, domain="gguf")
    rn2_ref = (x2.float() * torch.rsqrt(x2.float().pow(2).mean(-1, keepdim=True) + 1e-5) * gamma2.bfloat16().float()).bfloat16()
    assert rel(rn2, rn2_ref) < 1e-3
    (srn2_only,) = run("SkipRmsNorm", [("x", x2.to(DEV)), ("s", skip2.to(DEV)), ("g", gamma2.to(DEV))], [BF], {"epsilon": 1e-5}, domain="gguf")
    s2_ref = x2 + skip2
    srn2_ref = (s2_ref.float() * torch.rsqrt(s2_ref.float().pow(2).mean(-1, keepdim=True) + 1e-5) * gamma2.bfloat16().float()).bfloat16()
    assert rel(srn2_only, srn2_ref) < 1e-3
    # 5. MambaStepGGUF and MambaVerifyGGUF bit-exact vs MambaStep and MambaVerify
    heads, hd, ds, ng = 128, 64, 128, 8
    di = heads * hd
    cd = di + 2 * ng * ds
    proj = (torch.randn(8, di + cd + heads) * 0.7).bfloat16()
    cst = (torch.randn(8, cd, 3) * 0.5).bfloat16()
    sst = (torch.randn(8, heads, ds, hd) * 0.1).float()
    cw_f32 = (torch.randn(cd, 4) * 0.3).float()
    cb_f32 = (torch.randn(cd) * 0.1).float()
    dtb = (torch.randn(heads) * 0.3).float()
    an = (-torch.rand(heads) * 4 - 0.1).float()
    D = torch.randn(heads).float()
    nw = (torch.rand(di) + 0.5).float()
    sel = torch.tensor([2], dtype=torch.int32)
    attrs = {"heads": heads, "head_dim": hd, "d_state": ds, "n_groups": ng, "epsilon": 1e-5, "time_step_min": 0.0}
    step_bf16 = run("MambaStep", [("proj", proj.to(DEV)), ("cs", cst[:1].to(DEV)), ("ss", sst[:1].to(DEV)),
                                  ("cw", cw_f32.bfloat16().to(DEV)), ("cb", cb_f32.bfloat16().to(DEV)),
                                  ("dtb", dtb.to(DEV)), ("an", an.to(DEV)), ("D", D.to(DEV)), ("nw", nw.to(DEV))],
                    [BF, BF, F32, BF, F32], attrs)
    step_gguf = run("MambaStepGGUF", [("proj", proj.to(DEV)), ("cs", cst[:1].to(DEV)), ("ss", sst[:1].to(DEV)),
                                      ("cw", cw_f32.to(DEV)), ("cb", cb_f32.to(DEV)),
                                      ("dtb", dtb.to(DEV)), ("an", an.to(DEV)), ("D", D.to(DEV)), ("nw", nw.to(DEV))],
                    [BF, BF, F32, BF, F32], attrs, domain="gguf")
    assert all(torch.equal(a, b) for a, b in zip(step_bf16[:3], step_gguf[:3]))
    out_bf16 = run("MambaVerify", [("proj", proj.to(DEV)), ("cs", cst.to(DEV)), ("ss", sst.to(DEV)),
                                   ("cw", cw_f32.bfloat16().to(DEV)), ("cb", cb_f32.bfloat16().to(DEV)),
                                   ("dtb", dtb.to(DEV)), ("an", an.to(DEV)), ("D", D.to(DEV)), ("nw", nw.to(DEV)),
                                   ("sel", sel.to(DEV))], [BF, BF, F32, BF, F32], attrs)
    out_gguf = run("MambaVerifyGGUF", [("proj", proj.to(DEV)), ("cs", cst.to(DEV)), ("ss", sst.to(DEV)),
                                       ("cw", cw_f32.to(DEV)), ("cb", cb_f32.to(DEV)),
                                       ("dtb", dtb.to(DEV)), ("an", an.to(DEV)), ("D", D.to(DEV)), ("nw", nw.to(DEV)),
                                       ("sel", sel.to(DEV))], [BF, BF, F32, BF, F32], attrs)
    assert all(torch.equal(a, b) for a, b in zip(out_bf16[:3], out_gguf[:3]))
    print("GGUF ops (Nvfp4Gemv, Q8Gemv, Q8Embed, MoeRouteGGUF, MambaStepGGUF, MambaVerifyGGUF, RmsNorm, SkipRmsNorm) all passed")


def test_mamba(R, heads=128, hd=64, ds=128, ng=8):
    torch.manual_seed(5)
    di = heads * hd
    cd = di + 2 * ng * ds
    proj = (torch.randn(R, di + cd + heads) * 0.7).bfloat16()
    cst = (torch.randn(1, cd, 3) * 0.5).bfloat16()
    sst = torch.randn(1, heads, ds, hd) * 0.1
    cw = (torch.randn(cd, 1, 4) * 0.3).bfloat16()
    cb = (torch.randn(cd) * 0.1).bfloat16()
    dtb = torch.randn(heads) * 0.3
    an = -torch.rand(heads) * 4 - 0.1
    D = torch.randn(heads)
    nw = torch.rand(di) + 0.5
    ins = [("proj", proj), ("cs", cst), ("ss", sst), ("cw", cw), ("cb", cb), ("dtb", dtb), ("an", an), ("D", D), ("nw", nw)]
    y, cs_o, ss_o, _, _ = run("MambaStep", [(n, t.to(DEV)) for n, t in ins], [BF, BF, F32, BF, F32],
                              {"heads": heads, "head_dim": hd, "d_state": ds, "n_groups": ng, "epsilon": 1e-5, "time_step_min": 0.001})
    # reference
    gate, xbc, dt = proj.split([di, cd, heads], -1)
    inp = torch.cat([cst[0].T.float(), xbc.float()], 0)  # [3+R, cd]
    conv = torch.zeros(R, cd)
    for r in range(R):
        win = inp[r:r + 4]  # [4, cd]
        conv[r] = torch.nn.functional.silu((win * cw[:, 0, :].T.float()).sum(0) + cb.float())
    conv = conv.bfloat16().float()
    xs, Bm, Cm = conv.split([di, ng * ds, ng * ds], -1)
    st = sst[0].clone()
    ys = []
    for r in range(R):
        dts = torch.clamp(torch.nn.functional.softplus(dt[r].float() + dtb), min=0.001)
        dA = torch.exp(an * dts)
        xh = xs[r].reshape(heads, hd)
        Bh = Bm[r].reshape(ng, ds).repeat_interleave(heads // ng, 0)
        Ch = Cm[r].reshape(ng, ds).repeat_interleave(heads // ng, 0)
        st = st * dA[:, None, None] + (dts[:, None] * xh)[:, None, :] * Bh[:, :, None]
        ys.append((st * Ch[:, :, None]).sum(1) + D[:, None] * xh)
    yp = torch.stack(ys).reshape(R, di)
    g = yp * torch.nn.functional.silu(gate.float())
    g = g.reshape(R, ng, di // ng)
    g = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + 1e-5)
    ref = (g.reshape(R, di) * nw)
    print(f"MambaStep R={R}: y rel err {rel(y, ref):.2e}, ssm {rel(ss_o[0], st):.2e}, conv {rel(cs_o[0], inp[-3:].T):.2e}")
    assert rel(y, ref) < 1e-2 and rel(ss_o[0], st) < 1e-4 and rel(cs_o[0], inp[-3:].T) < 1e-6


def test_mamba_verify(R=8, sel=3, heads=128, hd=64, ds=128, ng=8):
    """MambaVerify: starts from state row `sel` of a [R, ...] buffer, writes the state after token r to row r."""
    torch.manual_seed(7)
    di = heads * hd
    cd = di + 2 * ng * ds
    proj = (torch.randn(R, di + cd + heads) * 0.7).bfloat16()
    cst = (torch.randn(R, cd, 3) * 0.5).bfloat16()
    sst = torch.randn(R, heads, ds, hd) * 0.1
    cw = (torch.randn(cd, 1, 4) * 0.3).bfloat16()
    cb = (torch.randn(cd) * 0.1).bfloat16()
    dtb = torch.randn(heads) * 0.3
    an = -torch.rand(heads) * 4 - 0.1
    D = torch.randn(heads)
    nw = torch.rand(di) + 0.5
    ins = [("proj", proj), ("cs", cst), ("ss", sst), ("cw", cw), ("cb", cb), ("dtb", dtb), ("an", an), ("D", D), ("nw", nw),
           ("sel", torch.tensor([sel], dtype=torch.int32))]
    outs = run("MambaVerify", [(n, t.to(DEV)) for n, t in ins], [BF, BF, F32, BF, F32],
               {"heads": heads, "head_dim": hd, "d_state": ds, "n_groups": ng, "epsilon": 1e-5, "time_step_min": 0.001})
    y_v, cs_all, ss_all = outs[0].cpu(), outs[1].cpu(), outs[2].cpu()
    # reference: the stock single-state op run row by row, then compared against every row's state
    st_conv, st_ssm = cst[sel:sel + 1].clone(), sst[sel:sel + 1].clone()
    ys = []
    for r in range(R):
        ins1 = [("proj", proj[r:r + 1]), ("cs", st_conv), ("ss", st_ssm), ("cw", cw), ("cb", cb), ("dtb", dtb), ("an", an), ("D", D), ("nw", nw)]
        y, cs_o, ss_o, _, _ = run("MambaStep", [(n, t.to(DEV)) for n, t in ins1], [BF, BF, F32, BF, F32],
                                  {"heads": heads, "head_dim": hd, "d_state": ds, "n_groups": ng, "epsilon": 1e-5, "time_step_min": 0.001})
        st_conv, st_ssm = cs_o.cpu(), ss_o.cpu()
        ys.append(y.cpu())
        assert rel(ss_all[r], st_ssm[0]) < 1e-5 and rel(cs_all[r], st_conv[0]) < 1e-6, r
    print(f"MambaVerify R={R} sel={sel}: y rel err {rel(y_v, torch.cat(ys)):.2e}, all per-row states match")
    assert rel(y_v, torch.cat(ys)) < 1e-4


if __name__ == "__main__":
    for R in (1, 2, 8):
        test_nvfp4(R, 18560, 4096)
    test_nvfp4(1, 4096, 8192)
    for R, N, K in ((2, 4096, 8192), (3, 4096, 8192), (5, 1024, 4096), (8, 4096, 8192), (7, 18560, 4096)):
        test_nvfp4(R, N, K)
    test_nvfp4(8, 18560, 4096, XSCALE=3000.0)
    test_nvfp4(1, 18560, 4096, XSCALE=3000.0)  # massive activations must not overflow the fp16 partial sums
    for bits in (8, 4):
        for R, N, K in ((1, 1024, 4096), (1, 4096, 1024), (4, 4096, 5376), (1, 131072, 4096), (8, 131072, 4096), (3, 1024, 4096),
                        (8, 4096, 1024), (6, 4096, 5376), (5, 2688, 4096)):
            test_qgemv(bits, R, N, K)
    test_relu2()
    for R, E in ((1, 240), (4, 496), (8, 304)):
        test_route(R, E)
    for R, E, I in ((1, 240, 2048), (2, 304, 2304), (1, 496, 2560)):
        test_moe_decode(R, E, I)
    for R, E, I in ((1, 240, 2048), (8, 304, 2304), (5, 496, 2560), (8, 12, 2048), (8, 240, 2048)):  # E=12: every expert shared by many rows
        test_moe_decode(R, E, I, topk=9 if E > 12 else 6, op="MoeVerify")
    for R in (1, 2, 8, 16):
        test_mamba(R)
    for R, N, K in ((8, 5376, 4096), (2, 4096, 5376), (5, 1024, 4096), (8, 131072 // 8, 4096)):
        test_q4_iq2(R, N, K)
    for R, E, I in ((8, 304, 2304), (3, 240, 2048), (8, 12, 2560)):
        test_moe_iq2(R, E, I, topk=9 if E > 12 else 6)
    for R, N, K in ((8, 5376, 4096), (1, 4096, 5376), (3, 1024, 4096), (8, 4096, 5376)):
        test_q4_iq2_native(R, N, K)
    for R, E, I in ((8, 304, 2304), (1, 240, 1536), (5, 496, 2560), (8, 12, 2048)):
        test_moe_iq2_native(R, E, I, topk=9 if E > 12 else 6)
    test_mamba_verify(8, 3)
    test_mamba_verify(5, 0)
    test_gguf_ops()
    print("all passed")
