const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, ExternalHyperlink,
  HeadingLevel, AlignmentType,
} = require("docx");

const ARIAL = "Arial";
const SZ = 24;        // half-points -> 12 pt

const txt = (text, opts = {}) =>
  new Paragraph({
    spacing: { after: opts.after === undefined ? 120 : opts.after },
    alignment: opts.align,
    children: [new TextRun({ text, font: ARIAL, size: SZ, bold: !!opts.bold,
                             italics: !!opts.italics, color: opts.color })],
  });

const label = (text) =>
  new Paragraph({
    spacing: { before: 160, after: 60 },
    children: [new TextRun({ text, font: ARIAL, size: 20, bold: true, color: "666666" })],
  });

// Equation images: read the true pixel dimensions out of the PNG IHDR chunk and
// preserve the aspect ratio. Hardcoding it squashed every equation by up to 60%.
const pngSize = (file) => {
  const b = fs.readFileSync(__dirname + "/" + file);
  if (b.toString("ascii", 12, 16) !== "IHDR") throw new Error("not a PNG: " + file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
};

const eq = (file, widthIn) => {
  const { w: pw, h: ph } = pngSize(file);
  const w = Math.round(widthIn * 96);
  return new Paragraph({
    spacing: { after: 140 },
    alignment: AlignmentType.CENTER,
    children: [new ImageRun({
      type: "png",
      data: fs.readFileSync(__dirname + "/" + file),
      transformation: { width: w, height: Math.round(w * ph / pw) },
    })],
  });
};

const code = (text) =>
  new Paragraph({
    spacing: { after: 60 },
    children: [new TextRun({ text, font: "Courier New", size: 18 })],
  });

const ref = (url, what) =>
  new Paragraph({
    spacing: { after: 60 },
    bullet: { level: 0 },
    children: [
      new ExternalHyperlink({
        children: [new TextRun({ text: url, font: ARIAL, size: SZ,
                                 color: "1155CC", underline: {} })],
        link: url,
      }),
      new TextRun({ text: "  —  " + what, font: ARIAL, size: SZ }),
    ],
  });

const heading = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 320, after: 160 },
    children: [new TextRun({ text, font: ARIAL, size: 32, bold: true, color: "000000" })],
  });

const children = [];

children.push(new Paragraph({
  spacing: { after: 80 },
  children: [new TextRun({ text: "Loss functions", font: ARIAL, size: 40, bold: true })],
}));
children.push(txt("All weights below are read from the trained checkpoints, not from script defaults. "
  + "Ω = ocean mask (land excluded from every term).", { color: "666666" }));

// ---------------------------------------------------------------- CorrDiff
children.push(heading("CorrDiff"));
children.push(eq("eq_corrdiff_sym.png", 6.5));
children.push(label("WITH OUR VALUES"));
children.push(eq("eq_corrdiff_val.png", 5.2));
children.push(label("LATEX"));
children.push(code("\\mathcal{L} = \\frac{\\min(\\mathrm{SNR}_t,\\,5)}{\\mathrm{SNR}_t+1}"));
children.push(code("  \\left\\|\\hat{v}-v\\right\\|^{2}_{\\Omega}"));
children.push(code("\\qquad v=\\sqrt{\\bar{\\alpha}_t}\\,\\varepsilon-\\sqrt{1-\\bar{\\alpha}_t}\\,x_0"));
children.push(label("EXPLANATION"));
children.push(txt("One term. Plain squared error, but on v instead of on the field directly. "
  + "Predicting the noise breaks down when the image is almost clean, and predicting the clean "
  + "field breaks down when it is almost pure noise; v is a mix of the two that stays well behaved "
  + "at both ends."));
children.push(txt("The weight w exists because a randomly drawn timestep can be trivially easy — "
  + "denoising something barely noisy. Left alone those easy steps soak up most of the gradient. "
  + "The weight caps them so the middle of the schedule, where the model actually decides the "
  + "field's structure, gets the attention. γ = 5 is the value recommended in the paper it comes from."));
children.push(txt("There is no physics term and no observation term. This is the whole loss."));
children.push(label("REFERENCES"));
children.push(ref("https://arxiv.org/abs/2202.00512", "v-prediction"));
children.push(ref("https://arxiv.org/abs/2303.09556", "the Min-SNR-γ weight, and γ = 5 as its default"));
children.push(ref("https://arxiv.org/abs/2102.09672", "cosine noise schedule"));
children.push(ref("https://arxiv.org/abs/2309.15214", "the deterministic-mean + residual-diffusion design (arXiv preprint, no journal version)"));

// ------------------------------------------------------------------ Stream
children.push(heading("Stream"));
children.push(txt("Two separately trained networks, so two losses. A diffusion model predicts "
  + "direction; a plain regression predicts speed; the two are combined and Helmholtz-reprojected.",
  { color: "666666" }));

children.push(label("DIRECTION NETWORK"));
children.push(eq("eq_streamdir_sym.png", 6.5));
children.push(label("WITH OUR VALUES"));
children.push(eq("eq_streamdir_val.png", 6.5));
children.push(label("LATEX"));
children.push(code("\\mathcal{L} = w_t\\left\\|\\hat{x}_0-x_0\\right\\|^{2}_{\\Omega}"));
children.push(code("  + 1.0\\left(1-\\cos\\theta\\right)_{\\Omega}"));
children.push(code("  + 0.2\\left(\\frac{\\mathrm{rms}(\\hat{x}_0)}{\\mathrm{rms}(x_0)}-1\\right)^{2}"));
children.push(code("\\qquad w_t=\\frac{\\min(\\mathrm{SNR}_t,\\,5)}{\\mathrm{mean}_t\\left[\\min(\\mathrm{SNR}_t,\\,5)\\right]}"));

children.push(label("MAGNITUDE NETWORK"));
children.push(eq("eq_streammag_sym.png", 5.6));
children.push(label("WITH OUR VALUES"));
children.push(eq("eq_streammag_val.png", 5.6));
children.push(label("LATEX"));
children.push(code("\\mathcal{L} = \\tfrac{1}{2}\\left(\\log\\sigma^{2}"));
children.push(code("  + \\frac{(y-\\mu)^{2}}{\\sigma^{2}}\\right) + 0.05\\,\\mathrm{TV}(\\log\\sigma^{2})"));

children.push(label("EXPLANATION"));
children.push(txt("Direction network. Same squared-error base as CorrDiff, plus two additions. "
  + "The angle term scores which way the water points, separately from how fast it goes. The "
  + "magnitude term is an explicit patch against a known failure: squared error rewards hedging "
  + "toward the average, which shrinks the field, so this compares the prediction's overall "
  + "amplitude to the truth's and penalises shrinkage. "));
children.push(txt("A fourth term was dropped on 2 Sept 2026. It correlated the model's "
  + "spread of directions across draws against an empirical spread map, and existed to make the "
  + "model's stated uncertainty line up with where it is actually wrong. Measured against an "
  + "otherwise identical control it did the opposite — the correlation between predicted "
  + "uncertainty and real error was 0.223 with the term and 0.296 without — and it also made the "
  + "vorticity field worse and the intervals about 10% wider, for no gain in RMSE. No weight "
  + "between 0 and 1 helped. The shipped weights no longer use it.", { italics: true }));
children.push(txt("Magnitude network. It outputs a speed and its own confidence for every cell. "
  + "Being wrong while confident is punished hard by the second part; claiming huge uncertainty "
  + "everywhere is punished by the first. The balance forces an honest per-cell confidence instead "
  + "of one number for the whole map. Without the log term the model would just declare infinite "
  + "uncertainty and the loss would collapse to zero. The final term keeps the confidence map "
  + "smooth rather than speckled."));
children.push(txt("Two caveats worth stating. A vorticity term exists in other stream loss "
  + "variants in the codebase and was not used here. And λ = 0.05 on the smoothness term is the "
  + "script default — that checkpoint does not record it, so it is inferred rather than read.", { italics: true }));

children.push(label("REFERENCES"));
children.push(ref("https://arxiv.org/abs/2303.09556", "the Min-SNR-γ weight"));
children.push(ref("https://doi.org/10.1109/ICNN.1994.374138", "predicting a mean and a variance together (the magnitude loss)"));
children.push(ref("https://arxiv.org/abs/1703.04977", "per-pixel learned uncertainty in deep networks"));
children.push(ref("https://doi.org/10.1016/0167-2789(92)90242-F", "total-variation smoothness"));
children.push(ref("https://arxiv.org/abs/2203.09168", "caveat: this loss under-fits exactly where error is largest"));
children.push(txt("The magnitude-matching term is ours — no citation exists, it needs describing "
  + "from scratch. The spread-calibration term was also ours and has been removed.", { italics: true, color: "666666" }));

// ---------------------------------------------------------------- DistAttn
children.push(heading("DistAttn"));
children.push(eq("eq_distattn_sym.png", 6.5));
children.push(label("WITH OUR VALUES"));
children.push(eq("eq_distattn_val.png", 6.5));
children.push(label("LATEX"));
children.push(code("\\mathcal{L} = \\left\\|\\hat{\\varepsilon}-\\varepsilon\\right\\|^{2}_{\\Omega}"));
children.push(code("  + 0.002\\left\\|\\Phi(\\hat{x}_0)-\\Phi(x_0)\\right\\|_{\\Omega}"));
children.push(code("  + 1.0\\,\\frac{\\sum_{i\\in P}\\left(\\hat{x}_0-x_0\\right)^{2}}{2N_P}"));
children.push(code("\\qquad \\Phi(x)=\\left[\\nabla\\times x,\\;\\nabla\\cdot x\\right]"));
children.push(label("EXPLANATION"));
children.push(txt("Three terms. The base is standard noise-prediction squared error."));
children.push(txt("The second term compares the curl and divergence of the prediction against the "
  + "truth's — that is, whether the water rotates the right way and whether it appears or vanishes "
  + "anywhere. It is a physics sanity check on the shape of the field, not on its values."));
children.push(txt("The third term forces the prediction to agree with the readings at the cells the "
  + "vehicle actually drove through. It is averaged over visited cells only; averaging over the whole "
  + "grid would dilute it to nothing, since the track touches under 5% of the domain."));
children.push(txt("One thing to flag: the observation term was designed to be weightable by how old "
  + "each reading is, but the shipped checkpoint predates that option, so every reading was weighted "
  + "equally. The age-aware version was never trained.", { italics: true }));
children.push(label("REFERENCES"));
children.push(ref("https://arxiv.org/abs/2006.11239", "the noise-prediction base objective"));
children.push(ref("https://doi.org/10.1016/j.jcp.2018.10.045", "precedent for physics-residual terms in a loss"));
children.push(txt("The curl/divergence form and the observation-consistency term are ours — no "
  + "citation, describe from scratch.", { italics: true, color: "666666" }));

const doc = new Document({
  styles: { default: { document: { run: { font: ARIAL, size: SZ } } } },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
                          margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 } } },
    children,
  }],
});

Packer.toBuffer(doc).then((b) => {
  fs.writeFileSync(__dirname + "/loss_functions.docx", b);
  console.log("wrote loss_functions.docx");
});
