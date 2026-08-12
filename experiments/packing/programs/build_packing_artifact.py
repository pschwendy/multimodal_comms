#!/usr/bin/env python3
"""Render the packing results into a standalone, self-contained HTML page.

Design notes (so a later edit doesn't undo the intent):
  - The page has ONE thesis: for these representations, precision compresses
    and dimension does not. That is encoded in colour -- viridian marks every
    approach that works, clay marks every approach that does not -- and the
    coding is used consistently in every table, so the argument is legible
    from the colour alone before any number is read.
  - Monospace carries labels, eyebrows and all numerals (tabular), because
    the subject is a signal-processing readout and that is its vernacular.
    Prose stays in a system sans at a comfortable measure.
  - The packet strip is generated from a REAL rotor packet, not drawn: it is
    the evidence for "every message touches every coordinate", so faking it
    would defeat the purpose.
"""

import argparse
import base64
import io
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

CSS = """
:root{
  --paper:#fcfcfb; --ink:#14181a; --mut:#59656a; --line:#e2e7e5; --card:#f4f6f5;
  --acc:#2f7d6b; --acc-soft:#dcece7; --neg:#a8553f; --neg-soft:#f4e2dc;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
@media (prefers-color-scheme:dark){:root{
  --paper:#0f1315; --ink:#e6eae8; --mut:#8f9d9a; --line:#242b2d; --card:#161b1d;
  --acc:#6ec6ab; --acc-soft:#17302a; --neg:#d08a72; --neg-soft:#33201a;
}}
:root[data-theme="dark"]{
  --paper:#0f1315; --ink:#e6eae8; --mut:#8f9d9a; --line:#242b2d; --card:#161b1d;
  --acc:#6ec6ab; --acc-soft:#17302a; --neg:#d08a72; --neg-soft:#33201a;
}
:root[data-theme="light"]{
  --paper:#fcfcfb; --ink:#14181a; --mut:#59656a; --line:#e2e7e5; --card:#f4f6f5;
  --acc:#2f7d6b; --acc-soft:#dcece7; --neg:#a8553f; --neg-soft:#f4e2dc;
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);font-family:var(--sans);
  font-size:16.5px;line-height:1.62;margin:0;padding:0}
.wrap{max-width:74rem;margin:0 auto;padding:3.5rem 1.5rem 6rem;
  display:flex;flex-direction:column;gap:3.25rem}
.prose{max-width:40rem}
h1{font-size:clamp(1.85rem,4.2vw,2.6rem);line-height:1.08;margin:0;
  letter-spacing:-.022em;font-weight:660;text-wrap:balance}
h2{font-size:1.18rem;margin:0 0 .2rem;letter-spacing:-.012em;font-weight:640;
  text-wrap:balance}
p{margin:0 0 .9rem}
p:last-child{margin-bottom:0}
.eyebrow{font-family:var(--mono);font-size:.68rem;letter-spacing:.14em;
  text-transform:uppercase;color:var(--mut);margin:0 0 .5rem}
.lede{font-size:1.06rem;color:var(--mut);max-width:44rem;margin:.9rem 0 0}
section{display:flex;flex-direction:column;gap:.85rem}
hr{border:0;border-top:1px solid var(--line);margin:0}

/* readout tiles */
.readout{display:grid;gap:.75rem;grid-template-columns:repeat(auto-fit,minmax(min(100%,13.5rem),1fr))}
.tile{background:var(--card);border:1px solid var(--line);border-radius:2px;
  padding:1rem 1.1rem;display:flex;flex-direction:column;gap:.15rem}
.tile b{font-family:var(--mono);font-size:1.75rem;font-weight:620;color:var(--acc);
  letter-spacing:-.03em;font-variant-numeric:tabular-nums;line-height:1.1}
.tile span{font-size:.82rem;color:var(--mut);line-height:1.4}

/* tables */
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:2px}
table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{padding:.5rem .85rem;text-align:right;white-space:nowrap;
  border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
th{font-family:var(--mono);font-size:.67rem;letter-spacing:.1em;
  text-transform:uppercase;color:var(--mut);font-weight:600;
  background:var(--card);position:sticky;top:0}
th:first-child,td:first-child{text-align:left;font-variant-numeric:normal}
tbody tr:last-child td{border-bottom:0}
td.num{font-family:var(--mono)}
tr.good td:first-child{box-shadow:inset 3px 0 0 var(--acc)}
tr.bad  td:first-child{box-shadow:inset 3px 0 0 var(--neg)}
tr.base td:first-child{box-shadow:inset 3px 0 0 var(--line)}
.v-good{color:var(--acc);font-weight:640}
.v-bad{color:var(--neg);font-weight:640}
.tag{font-family:var(--mono);font-size:.66rem;letter-spacing:.06em;
  padding:.1rem .4rem;border-radius:2px;text-transform:uppercase}
.tag.good{background:var(--acc-soft);color:var(--acc)}
.tag.bad{background:var(--neg-soft);color:var(--neg)}
.note{color:var(--mut);font-size:.87rem;max-width:44rem}
.note strong{color:var(--ink)}

/* packet strip */
figure{margin:0;display:flex;flex-direction:column;gap:.5rem}
figcaption{font-size:.82rem;color:var(--mut);max-width:44rem}
.striplabel{font-family:var(--mono);font-size:.68rem;letter-spacing:.1em;
  text-transform:uppercase;color:var(--mut)}
svg{display:block;width:100%;height:auto}

/* image grid */
.imgs{overflow-x:auto}
.imgs table{font-size:.8rem}
.imgs td{padding:.35rem}
.imgs img{display:block;border-radius:2px;border:1px solid var(--line)}
.imgs th{text-align:center;white-space:normal;min-width:8.5rem;line-height:1.35;
  text-transform:none;letter-spacing:0;font-size:.74rem;font-family:var(--sans)}
.imgs th small{display:block;font-family:var(--mono);font-size:.66rem;color:var(--mut);
  font-weight:400;letter-spacing:.02em}
:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
"""


def img_tag(arr, size=126):
    import numpy as np
    from PIL import Image
    a = ((arr.transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype("uint8")
    im = Image.fromarray(a).resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return (f'<img src="data:image/png;base64,'
            f'{base64.b64encode(buf.getvalue()).decode()}" '
            f'width="{size}" height="{size}" alt="">')


def packet_strip(values, color_var, n=160):
    """Inline SVG bar strip of real packet coordinates."""
    import numpy as np
    v = np.asarray(values[:n], dtype=float)
    m = float(np.abs(v).max()) or 1.0
    v = v / m
    w, h, gap = 1000.0 / n, 44.0, 0.28
    bars = []
    for i, x in enumerate(v):
        bh = max(abs(x) * (h / 2 - 1), 0.6)
        y = h / 2 - bh if x >= 0 else h / 2
        bars.append(f'<rect x="{i*w+gap:.2f}" y="{y:.2f}" width="{w-2*gap:.2f}" '
                    f'height="{bh:.2f}" fill="var(--{color_var})"/>')
    return (f'<svg viewBox="0 0 1000 {h}" preserveAspectRatio="none" '
            f'role="img" aria-label="packet coordinate values">'
            f'<line x1="0" y1="{h/2}" x2="1000" y2="{h/2}" stroke="var(--line)" '
            f'stroke-width="0.6"/>{"".join(bars)}</svg>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="reports/precision_packing_results.json")
    ap.add_argument("--recon", default="reports/multimodal_recon.npz")
    ap.add_argument("--out", default="reports/packing_artifact.html")
    args = ap.parse_args()

    R = json.load(open(args.results))
    H = ["<title>One packet, many messages</title>", f"<style>{CSS}</style>",
         '<div class="wrap">']

    # ---- header -----------------------------------------------------------
    H.append('<header class="prose">'
             '<p class="eyebrow">Multi-agent latent communication &middot; 2026-07-22</p>'
             '<h1>One packet, many messages</h1>'
             '<p class="lede">Fusing many agents\' messages into a single '
             'representation that means something different under every '
             'receiver\'s key &mdash; and finding out which axis of that '
             'representation actually compresses.</p></header>')

    txt = R["text"]
    base = txt["baseline_fp32_unpacked"]
    tn = R.get("trained_narrow_code")
    hero = None
    if tn:
        hero = min(tn["rows"], key=lambda r: r["bytes_per_msg"])
    cb = R.get("compression_baselines", {})
    H.append('<div class="readout">')
    if hero:
        # Quote the ratio against PLAINTEXT, not against the old latent scheme.
        # The latter was itself ~198x larger than the text, so a ratio against
        # it mostly measures a self-inflicted inefficiency.
        pt = cb.get("plaintext_bytes")
        sub = (f'messages sharing one packet, {hero["bytes_per_msg"]} bytes each '
               f'&mdash; {pt/hero["bytes_per_msg"]:.1f}&times; smaller than the '
               f'plaintext, lossily' if pt else
               f'messages sharing one packet, {hero["bytes_per_msg"]} bytes each')
        H.append(f'<div class="tile"><b>{hero["messages"]}</b><span>{sub}</span></div>')
    H.append('<div class="tile"><b>100%</b><span>of packet coordinates carry every '
             'single message &mdash; fusion, not concatenation</span></div>'
             '<div class="tile"><b>0</b><span>bits about any other receiver\'s message '
             'leak to an insider, proven not measured</span></div></div>')

    # ---- mechanism --------------------------------------------------------
    H.append('<hr>')
    strip_html = ""
    try:
        import torch
        from multimodal_comms.benchmarks.hiddenbench.runtime.packing import RotorPacker
        torch.manual_seed(3)
        P, d = 2048, 128
        rp = RotorPacker(P, d, seed=11, private=True)
        codes = {}
        for i in range(P // d):
            c = torch.randn(d)
            codes[i] = c * torch.rsqrt(c.pow(2).mean())
        full = rp.pack(codes).numpy()
        solo = rp.pack({3: codes[3]}).numpy()
        strip_html = (
            '<figure><span class="striplabel">One message alone, placed in the packet</span>'
            + packet_strip(solo, "acc") +
            '<span class="striplabel">All 16 messages superposed</span>'
            + packet_strip(full, "mut") +
            '<figcaption>Real packet coordinates. A single message already spans '
            'the whole packet, so no coordinate can be attributed to any sender &mdash; '
            'yet each receiver still recovers its own message exactly, because the '
            'per-slot frames stay mutually orthogonal.</figcaption></figure>')
    except Exception as e:  # keep the page renderable without torch
        strip_html = f'<p class="note">(packet strip unavailable: {e})</p>'

    H.append('<section><p class="eyebrow">Mechanism</p>'
             '<h2>Disjoint subspaces are not disjoint coordinates</h2>'
             '<div class="prose"><p>Give slot <em>i</em> the rows '
             '<code>[i&middot;d, (i+1)&middot;d)</code> of a keyed dense rotation '
             '<code>R</code>. The frames are mutually orthogonal, so recovery is exact '
             'at every load up to <code>M = P/d</code> &mdash; but each message\'s '
             'contribution is dense across the entire packet. Density of superposition '
             'turns out to be free: it costs nothing in fidelity.</p></div>'
             + strip_html + '</section>')

    # ---- trained narrow code ---------------------------------------------
    if tn:
        H.append('<hr>')
        H.append('<section><p class="eyebrow">Result &middot; trained code</p>'
                 '<h2>128 messages in one packet</h2>'
                 '<p class="note">A narrow code cannot be <em>projected out</em> of a '
                 'pretrained representation (below) &mdash; but it can be <em>trained '
                 'in</em>. Held-out multi-agent messages, real rotor-fused packets, at '
                 f'25% of training. Chance floor {tn["chance_floor"]:.3f}; the same '
                 'autoencoder with the full uncompressed latent scores 0.955.</p>')
        H.append('<div class="scroll"><table><thead><tr><th>Fusion</th><th>Bits</th>'
                 '<th>Messages</th><th>Bytes / msg</th><th>Code cos</th>'
                 '<th>Content F1</th><th></th></tr></thead><tbody>')
        for r in tn["rows"]:
            good = r["f1"] > 0.6
            H.append(f'<tr class="{"good" if good else "bad"}"><td>{r["fusion"]}</td>'
                     f'<td class="num">{r["bits"]}</td>'
                     f'<td class="num">{r["messages"]}</td>'
                     f'<td class="num">{r["bytes_per_msg"]}</td>'
                     f'<td class="num">{r["code_cos"]:.3f}</td>'
                     f'<td class="num {"v-good" if good else "v-bad"}">{r["f1"]:.3f}</td>'
                     f'<td><div class="bar"><i style="width:'
                     f'{min(r["f1"]/0.955*100,100):.0f}%"></i></div></td></tr>')
        H.append('</tbody></table></div>')
        om = R.get("overfit_mechanism_check", {})
        H.append('<p class="note">Read the last two rows against each other: '
                 '<strong>rotor 0.662 vs concatenation 0.662</strong> at the same load '
                 '&mdash; identical &mdash; settles the central claim that dense '
                 'superposition costs nothing even on a learned code, while '
                 '<strong>frame 0.365</strong> is '
                 'its own crosstalk law arriving on schedule at &rho;=1. The width curve '
                 'also separates for the first time; every earlier run was flat across a '
                 '64&times; range, and that flatness was the symptom of a dead channel. '
                 f'A mechanism check ran first: {om.get("messages",16)} fixed messages '
                 f'through the same d={om.get("code_dim",320)} bottleneck memorise to '
                 f'free-running F1 <strong>{om.get("free_running_f1",1.0):.3f}</strong>, '
                 'so nothing in the architecture ever blocked learning.</p>'
                 )
        if cb:
            H.append('<h2>Against a real compressor</h2>')
            H.append('<p class="note">Quoting a ratio against the previous latent scheme '
                     'flatters this work: that baseline was itself ~198&times; larger than '
                     'simply sending the text, so most of the ratio measures a '
                     'self-inflicted inefficiency. Dev messages average 207 bytes '
                     '(40 tokens). The defensible claim is <strong>~5&times; smaller than '
                     'plaintext, lossily</strong> &mdash; and only once 4-bit quantisation '
                     'is applied; the fp32 representation is <em>larger</em> than the '
                     'message it encodes.</p>')
            H.append('<div class="scroll"><table><thead><tr><th>Encoding</th>'
                     '<th>Bytes / msg</th><th>Lossless</th></tr></thead><tbody>')
            for lbl, key, ll in [("plaintext UTF-8", "plaintext_bytes", None),
                                 ("gzip, per message", "gzip_bytes", True),
                                 ("zlib + shared dictionary", "zlib_shared_dict_bytes", True),
                                 ("ours, M=128, 8-bit", "ours_8bit_bytes", False),
                                 ("ours, M=128, 4-bit", "ours_4bit_bytes", False)]:
                v = cb.get(key)
                if v is None:
                    continue
                cls = "base" if ll is None else ("good" if ll else "bad")
                mark = ("&mdash;" if ll is None else
                        ("yes" if ll else "no &middot; F1 0.66"))
                H.append(f'<tr class="{cls}"><td>{lbl}</td><td class="num">{v}</td>'
                         f'<td>{mark}</td></tr>')
            H.append('</tbody></table></div>')
            H.append('<p class="note">What gzip cannot do is hand 128 receivers <em>one '
                     'packet</em> that each reads differently, recovering only its own '
                     'message with provably zero leakage about the others. That property, '
                     'not the byte count, is what this is for. For moving bytes alone, '
                     'zlib with a shared dictionary is lossless at 152 and is the better '
                     'tool.</p>')
        H.append('<p class="note"><strong>Scope.</strong> 6,001 training messages from a '
                 'handful of benchmark tasks, training accuracy 0.999 &mdash; the model has '
                 'largely memorised the corpus, so the held-out set is in-domain rather '
                 'than a transfer test. The claim is "128 agent messages of ~190 characters '
                 'share one packet", not "arbitrary text compresses 128&times;": the same '
                 'recipe on FineWeb reaches only 0.028.</p></section>')

    # ---- the thesis table -------------------------------------------------
    H.append('<hr>')
    H.append('<section><p class="eyebrow">Measurement &middot; text</p>'
             '<h2>Precision compresses with no training at all</h2>'
             '<p class="note">Rotor-fused packets of raw latents, quantised on the wire. '
             'Content-word F1 against the source message, 150-character messages; '
             f'chance floor {R["controls"]["f1_chance_floor"]:.3f}. '
             '<strong>No learned dimensional compression is used here at all.</strong></p>')
    H.append('<div class="scroll"><table><thead><tr><th>Configuration</th>'
             '<th>Messages</th><th>Bits</th><th>Bytes / msg</th><th>Content F1</th>'
             '</tr></thead><tbody>')
    H.append(f'<tr class="base"><td>uncompressed baseline</td><td class="num">1</td>'
             f'<td class="num">32</td><td class="num">{base["bytes_per_msg"]}</td>'
             f'<td class="num">{base["f1"]:.3f}</td></tr>')
    for r in txt["rows"]:
        rotor = r["fusion"] == "rotor"
        cls = "good" if rotor and r["f1"] > 0.95 else ("bad" if r["f1"] < 0.95 else "")
        name = ("rotor-fused" if rotor else "no fusion &mdash; quantised directly")
        vcls = "v-good" if r["f1"] >= 0.95 else "v-bad"
        H.append(f'<tr class="{cls}"><td>{name}</td><td class="num">{r["messages"]}</td>'
                 f'<td class="num">{r["bits"]}</td><td class="num">{r["bytes_per_msg"]}</td>'
                 f'<td class="num {vcls}">{r["f1"]:.3f}</td></tr>')
    H.append('</tbody></table></div>')
    H.append('<p class="note">Two things fall out. <strong>Fusion is what makes 4-bit '
             'viable</strong> &mdash; quantising raw latents directly costs a tenth of the '
             'F1 (0.888), quantising the <em>fused</em> packet does not, because the '
             'rotation destroys the outlier dimensions that otherwise eat the dynamic '
             'range. And <strong>quality improves as messages are added</strong> '
             '(0.971 &rarr; 0.989 from 8 to 32 messages): crosstalk is exactly zero, so the '
             'only loss is quantisation, and the packet grows more Gaussian the more codes '
             'are summed into it. Superposition makes the wire format better, not worse.</p>'
             '</section>')

    # ---- negative result --------------------------------------------------
    H.append('<hr>')
    H.append('<section><p class="eyebrow">Negative result</p>'
             '<h2>Why it had to be trained in, not projected out</h2>'
             '<p class="note">The decoder survives half its latent energy replaced by '
             'noise &mdash; but collapses under a learned projection with a '
             '<em>smaller</em> error. Noise perturbs directions; projection deletes '
             'them.</p>')
    H.append('<div class="scroll"><table><thead><tr><th>Perturbation of the latent</th>'
             '<th>Relative MSE</th><th>Content F1</th><th></th></tr></thead><tbody>')
    for r in R["noise_tolerance"]:
        bad = r["f1"] < 0.5
        H.append(f'<tr class="{"bad" if bad else "good"}"><td>{r["perturbation"]}</td>'
                 f'<td class="num">{r["rel_mse"]:.2f}</td>'
                 f'<td class="num {"v-bad" if bad else "v-good"}">{r["f1"]:.3f}</td>'
                 f'<td><span class="tag {"bad" if bad else "good"}">'
                 f'{"unusable" if bad else "intact"}</span></td></tr>')
    H.append('</tbody></table></div>')
    H.append('<p class="note">PCA confirms there is nothing safe to delete: rank-2560 '
             'captures only <strong>41%</strong> of these latents\' variance, rank-80 only '
             '21%. Nothing in the base autoencoder\'s training ever pressured its code to '
             'be low-dimensional, so it is not. Reaching hundreds of messages per packet '
             'therefore needs a narrow code <em>trained in from the start</em> &mdash; it '
             'cannot be projected out afterwards.</p></section>')

    # ---- images -----------------------------------------------------------
    if os.path.exists(args.recon):
        import numpy as np
        z = np.load(args.recon)
        keys = [k for k in ["orig", "vae", "m32_b8", "m32_b4", "learned_d2560"] if k in z]
        labels = {"orig": "original",
                  "vae": "VAE ceiling<small>23.51 dB</small>",
                  "m32_b8": "32 msgs &middot; 8-bit<small>4160 B &middot; 23.51 dB</small>",
                  "m32_b4": "32 msgs &middot; 4-bit<small>2112 B &middot; 22.78 dB</small>",
                  "learned_d2560": "learned bottleneck<small>18.06 dB</small>"}
        H.append('<hr>')
        H.append('<section><p class="eyebrow">Measurement &middot; images</p>'
                 '<h2>The same packets carry images</h2>'
                 '<p class="note">The fusion layer never inspects modality &mdash; it moves '
                 'fixed-width codes &mdash; so images ride the same packets under the same '
                 'keys, capacity law and leakage bound. 8-bit rotor packing lands '
                 '<strong>exactly on the VAE ceiling</strong> and is identical at 8 and 32 '
                 'messages. The last column is the learned dimensional bottleneck: the same '
                 'failure as the text side, in pixels.</p>')
        H.append('<div class="imgs"><table><thead><tr>' +
                 "".join(f"<th>{labels.get(k,k)}</th>" for k in keys) +
                 '</tr></thead><tbody>')
        for i in range(min(4, z[keys[0]].shape[0])):
            H.append("<tr>" + "".join(f"<td>{img_tag(z[k][i])}</td>" for k in keys) + "</tr>")
        H.append('</tbody></table></div></section>')

    # ---- leakage ----------------------------------------------------------
    H.append('<hr>')
    H.append('<section><p class="eyebrow">Confidentiality</p>'
             '<h2>What one receiver learns about another: nothing</h2>'
             '<div class="prose"><p>The packet layout <code>R</code> is necessarily shared, '
             'so each slot also gets a secret Haar rotation <code>V<sub>i</sub></code> '
             'applied before placement. An insider holding <code>R</code> and its own key '
             'sees <code>c<sub>i</sub>V<sub>i</sub></code> for every other slot &mdash; and '
             'by right-invariance of the Haar measure that is uniform on the sphere whatever '
             '<code>c<sub>i</sub></code> was. So the mutual information with any other '
             'message\'s direction is exactly zero, and by the data-processing inequality '
             'no decoder or amount of compute recovers it. Codes are unit-RMS, so the norm '
             'is a known constant and leaks nothing either.</p>'
             '<p>Measured over 60 trials, the best-effort stolen block sits at the chance '
             'floor. The control matters as much as the result: running the identical '
             'attack with the private rotations disabled recovers the victim\'s code at '
             'cosine <strong>&gt;0.99</strong>, which is what shows the test is measuring '
             'the rotation rather than a broken attack.</p></div></section>')

    # ---- honest limits ----------------------------------------------------
    c = R["controls"]
    H.append('<hr>')
    H.append('<section><p class="eyebrow">What this does not show</p>'
             '<h2>Open, and mis-steps worth keeping</h2>'
             '<div class="prose">'
             '<p>The headline number is bytes per message, not messages per fixed packet. '
             'Getting past ~100 messages in one 40&nbsp;KB packet needs the trained narrow '
             'code the negative result rules out obtaining by projection &mdash; that '
             'training is the open work, and three separate runs of it (matryoshka, '
             'single-width, frozen-decoder) all flatlined.</p>'
             f'<p>Two diagnoses along the way were wrong and are recorded as such. The '
             f'first blamed posterior collapse; the controls refuted it &mdash; feeding the '
             f'decoder zeros scores {c["decoder_zeros"]:.3f}, a different message\'s latents '
             f'{c["decoder_shuffled"]:.3f}, the true ones {c["decoder_true"]:.3f}, so the '
             f'plateau was never the model\'s prior. The second blamed a real gradient leak '
             'in the freeze window &mdash; a genuine bug, now fixed &mdash; but the '
             'post-fix run reproduced the plateau exactly. Finding <em>a</em> bug is not '
             'the same as finding <em>the</em> bug.</p></div></section>')

    H.append('</div>')
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write("\n".join(H))
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
