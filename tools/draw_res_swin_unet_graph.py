import argparse
import shutil
import subprocess
from pathlib import Path


def build_res_swin_unet_dot() -> str:
    return r"""
digraph ResSwinUNet {
  graph [rankdir=LR, splines=true, nodesep=0.35, ranksep=0.6, fontname="Helvetica"];
  node [shape=box, style="rounded,filled", fillcolor="#F7FAFC", color="#334155", fontname="Helvetica", fontsize=10];
  edge [color="#475569", fontname="Helvetica", fontsize=9];

  in [label="Input\nB x 3 x H x W", fillcolor="#E2E8F0"];

  subgraph cluster_enc {
    label="ResNetShallowEncoder";
    color="#93C5FD";
    style="rounded";
    x0 [label="conv1+bn+relu\nx0: B x 64 x H/2 x W/2"];
    x1 [label="maxpool + layer1\nx1: B x 64 x H/4 x W/4"];
    x2 [label="layer2\nx2: B x 128 x H/8 x W/8"];
    x0 -> x1 -> x2;
  }

  subgraph cluster_swin {
    label="Swin Encoder";
    color="#86EFAC";
    style="rounded";
    s3 [label="SwinStage3\n(downsample + SwinBlocks)\nx3: B x 256 x H/16 x W/16"];
    s4 [label="SwinStage4\n(downsample + SwinBlocks)\nx4: B x 512 x H/32 x W/32"];
    wb [label="WaveletDecoupledBottleneck\n(optional)", fillcolor="#ECFCCB"];
    s3 -> s4;
  }

  subgraph cluster_dec {
    label="UNet Decoder (UpBlock + AttentionGate)";
    color="#FDBA74";
    style="rounded";
    d4 [label="dec4: up 512->256 + AG(x3) + ConvBlock\nd4: B x 256 x H/16 x W/16"];
    d3 [label="dec3: up 256->128 + AG(x2) + ConvBlock\nd3: B x 128 x H/8 x W/8"];
    d2 [label="dec2: up 128->64 + AG(x1) + ConvBlock\nd2: B x 64 x H/4 x W/4"];
    d1 [label="dec1: up 64->64 + AG(x0) + ConvBlock\nd1: B x 64 x H/2 x W/2"];
    fu [label="final_up ConvTranspose2d 64->32\nf: B x 32 x H x W"];
    seg [label="seg_head 1x1\nseg_logits: B x C x H x W", fillcolor="#FDE68A"];
    bnd [label="boundary_head 1x1\nboundary_logits: B x 1 x H x W\n(optional)", fillcolor="#FDE68A"];
    d4 -> d3 -> d2 -> d1 -> fu -> seg;
    fu -> bnd [style=dashed, color="#EA580C", label="use_boundary=True"];
  }

  subgraph cluster_aux {
    label="Deep Supervision (optional)";
    color="#A78BFA";
    style="rounded";
    aux2 [label="aux_head_d2(d2)\nupsample -> H x W"];
    aux3 [label="aux_head_d3(d3)\nupsample -> H x W"];
    aux4 [label="aux_head_d4(d4)\nupsample -> H x W"];
  }

  in -> x0;
  x2 -> s3;

  s4 -> d4 [label="default path"];
  s4 -> wb [style=dashed, color="#16A34A", label="if use_wavelet_bottleneck=True"];
  wb -> d4 [style=dashed, color="#16A34A"];

  s3 -> d4 [style=dashed, color="#EA580C", label="skip x3 + AG"];
  x2 -> d3 [style=dashed, color="#EA580C", label="skip x2 + AG"];
  x1 -> d2 [style=dashed, color="#EA580C", label="skip x1 + AG"];
  x0 -> d1 [style=dashed, color="#EA580C", label="skip x0 + AG"];

  d2 -> aux2 [style=dashed, color="#7C3AED", label="deep_supervision=True"];
  d3 -> aux3 [style=dashed, color="#7C3AED", label="deep_supervision=True"];
  d4 -> aux4 [style=dashed, color="#7C3AED", label="deep_supervision=True"];
}
""".strip()


def render_with_dot(dot_path: Path, render_format: str) -> Path | None:
    dot_binary = shutil.which("dot")
    if dot_binary is None:
        return None

    output_path = dot_path.with_suffix(f".{render_format}")
    cmd = [dot_binary, f"-T{render_format}", str(dot_path), "-o", str(output_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "Graphviz render failed.")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw ResSwinUNet architecture graph from current implementation.")
    parser.add_argument("--output-dir", type=str, default="assets/diagrams")
    parser.add_argument("--name", type=str, default="res_swin_unet_structure")
    parser.add_argument("--render-format", type=str, default="png", choices=["png", "svg", "pdf"])
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Only export .dot file. Do not call Graphviz renderer.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dot_path = output_dir / f"{args.name}.dot"
    dot_text = build_res_swin_unet_dot()
    dot_path.write_text(dot_text + "\n", encoding="utf-8")
    print(f"[saved] DOT file: {dot_path}")

    if args.skip_render:
        print("[info] Skip rendering image (--skip-render).")
        return

    rendered_path = render_with_dot(dot_path, args.render_format)
    if rendered_path is None:
        print("[warn] Graphviz 'dot' binary not found. Install Graphviz to render image.")
        print(f"[hint] Example: dot -T{args.render_format} {dot_path} -o {dot_path.with_suffix('.' + args.render_format)}")
        return

    print(f"[saved] Rendered graph: {rendered_path}")


if __name__ == "__main__":
    main()
