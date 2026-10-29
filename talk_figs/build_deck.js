// Draft sponsor deck: "Keep It Local" (numbers are placeholders; figures from
// results/figures/talk/). Build: node talk_figs/build_deck.js
const pptxgen = require("pptxgenjs");
const path = require("path");

const F = (n) => path.resolve("results/figures/talk", n);
const NAVY = "003057", GOLD = "B3A369", WHITE = "FFFFFF", INK = "1A2430", GREY = "6B7480";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
pres.theme = { headFontFace: "Cambria", bodyFontFace: "Calibri" };

function titleSlide() {
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("Keep It Local", { x: 0.8, y: 2.1, w: 11.7, h: 1.1, fontFace: "Cambria",
    fontSize: 52, bold: true, color: WHITE, isTextBox: true });
  s.addText("What a shared seismic model needs from you, and what it gives away",
    { x: 0.85, y: 3.25, w: 11.5, h: 0.7, fontSize: 22, color: GOLD, italic: true, isTextBox: true });
  s.addText("Federated learning for seismic facies  ·  privacy–personalization tradeoff  ·  sponsor update",
    { x: 0.85, y: 6.6, w: 11.5, h: 0.5, fontSize: 13, color: "AAB4C0", isTextBox: true });
}

function summarySlide() {
  const s = pres.addSlide();
  s.addText("Three findings", { x: 0.8, y: 0.55, w: 11.5, h: 0.8, fontFace: "Cambria",
    fontSize: 34, bold: true, color: NAVY, isTextBox: true });
  const items = [
    ["Sharing is safe for personalization", "A shared model matches single-survey training on every survey — no accuracy lost to collaboration (within seed noise).", "01"],
    ["Your data gives itself away", "The model's internal statistics alone identify which operator sent an update: 99 in 100 vs 33 by guessing.", "02"],
    ["There is a boundary", "Cross-survey sharing helps the smallest survey and hurts the largest — a shared body is not free.", "03"],
  ];
  let y = 1.7;
  for (const [h, b, n] of items) {
    s.addText(n, { x: 0.9, y: y, w: 1.1, h: 1.1, fontSize: 40, bold: true, color: GOLD, isTextBox: true });
    s.addText(h, { x: 2.1, y: y + 0.05, w: 10.2, h: 0.55, fontSize: 21, bold: true, color: INK, isTextBox: true });
    s.addText(b, { x: 2.1, y: y + 0.62, w: 10.2, h: 0.9, fontSize: 15, color: GREY, isTextBox: true });
    y += 1.75;
  }
}

function figSlide(title, img, punch, note) {
  const s = pres.addSlide();
  s.addText(title, { x: 0.6, y: 0.35, w: 12.1, h: 0.7, fontFace: "Cambria",
    fontSize: 30, bold: true, color: NAVY, isTextBox: true });
  s.addImage({ path: F(img), x: 1.15, y: 1.2, w: 11.0, h: 4.7 });
  if (punch) s.addText(punch, { x: 0.8, y: 6.05, w: 11.7, h: 0.7, fontSize: 18,
    bold: true, color: NAVY, italic: true, isTextBox: true });
  if (note) s.addText(note, { x: 0.8, y: 6.75, w: 11.7, h: 0.5, fontSize: 12, color: GREY, isTextBox: true });
}

function closing() {
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("What this means for us", { x: 0.8, y: 0.9, w: 11.7, h: 0.9, fontFace: "Cambria",
    fontSize: 36, bold: true, color: WHITE, isTextBox: true });
  const bullets = [
    "Federated training costs us no accuracy — a shared model is as good as training on one survey.",
    "Keeping BatchNorm + the head local removes the two easy leak channels.",
    "Sharing the body across very different surveys is not free: large, distinct surveys lose out.",
    "Recommendation: keep BN + head local; consider secure aggregation before pooling many partners.",
    "Next: confirm on re-trained Parihaka (orientation fix) + a 3rd survey; report means ± sd.",
  ];
  s.addText(bullets.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < bullets.length - 1, paraSpaceAfter: 12 } })),
    { x: 1.0, y: 2.1, w: 11.2, h: 4.2, fontSize: 18, color: "E8EDF2", isTextBox: true });
}

titleSlide();
summarySlide();
figSlide("Meet the operators", "f1_sections_n3.png",
  "Three surveys, three facies mixes — and only Operator C holds the Zechstein.",
  "Setup: contiguous blocks of one survey each. N=3 shown.");
figSlide("One shared model is as good as one trained locally", "f2_triptych_n3.png",
  "Shared vs single-survey: no accuracy lost to collaboration.",
  "Crossline, interpretation, federated prediction, errors. (Numbers to finalise.)");
figSlide("Averaged BatchNorm loses the geology", "f3_blackout_strip.png",
  "Average the normalization across surveys and the model forgets how to see one of them — fix it by keeping BN local.",
  "Same weights, same crossline: (a) averaged BN vs (b) local BN vs (c) interpretation.");
figSlide("How the model learns the layers", "f4_rounds_grid.png",
  "Facies structure appears in the first few rounds and refines over 20.",
  "Rounds 1 / 5 / 10 / 20, one fixed crossline.");
figSlide("Your data gives itself away", "f5_fingerprint_barcodes.png",
  "One upload — no seismic attached — and the statistics still say who sent it.",
  "Correct 99 in 100 (3 operators) vs 33 by guessing. Presence + identity only; not amounts.");
figSlide("The head update reveals who has the Zechstein", "f6_sections_n3.png",
  "Silence is a signal: a facies a site lacks shows up as a dead row in the update.",
  "Presence + ranking only — not measured amounts.");
figSlide("What stays home", "f9_unet_keep_home.png",
  "We share the conv body; each site keeps its BatchNorm and the segmentation head.",
  "BatchNorm running statistics are direct summaries of local data — keeping them local means those summaries never leave the site.");
figSlide("What secure aggregation hides — and what it doesn't", "f8_secure_agg.png",
  "It hides who contributed and how much, not what is in the pot — a rare facies is still a fingerprint.",
  "Top: without secure aggregation. Bottom: with it.");
figSlide("One shared body, three basins", "f11_three_basins.png",
  "A shared body helps the smallest survey and hurts the largest — there is a boundary.",
  "Per-survey heads + local BN. Parihaka shown rotated to true orientation. (Numbers to finalise.)");
closing();

pres.writeFile({ fileName: "results/figures/talk/Keep_It_Local_DRAFT.pptx" })
  .then(() => console.log("deck written: results/figures/talk/Keep_It_Local_DRAFT.pptx"))
  .catch((e) => { console.error(e); process.exit(1); });
