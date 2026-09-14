# DFlow-LM Paper Review — modification.md

## Summary (Phase 3 — Final Assessment)

### What was improved
- **Critical fix**: Added `\input{tables}` to main.tex — all 11 tables now render correctly (previously showed "Table ??")
- **Critical fix**: Removed duplicate macro definitions in tables.tex that would have caused compilation errors
- **Abstract**: Clarified DiffuGPT as hybrid (not discrete diffusion); now precisely compares against both MDLM and DiffuGPT separately
- **Introduction**: Removed "We hypothesize" opener; rewording to "Both limitations plausibly share a common cause" is more direct
- **Related Work**: Removed overclaim "first work to successfully adapt"; replaced with factual differentiation
- **Results sections**: Tightened all 11 subsections — removed formulaic superlatives ("establishing a new state-of-the-art", "demonstrating that"), converted ablation bullet list to compact prose
- **Discussion**: Completely rewritten — now analyzes frozen-backbone ablation to argue the benefit is in representation reuse (not just LM head warmstart); explains non-monotonic scaling gap; limitations are honest and concise
- **Conclusion**: No longer repeats intro; focuses on the central insight (quality gap = training gap)
- **Throughout**: Reduced "We ..." sentence monotony; fixed venue header (NeurIPS/ICML → COLM)

### What remains (cannot fix without experiments)
- **[E1]** All * marked results are preliminary — need finalization
- **[E2]** No multi-seed statistics reported
- **[S1]** §5 Results still has 11 subsections — some are thin but each covers a distinct experiment, so consolidation is a judgment call
- **[F3]** ~~Venue header uses neurips_2025.sty~~ FIXED — now uses `colm2026_conference.sty`

### Overall readiness
The paper is well-written and clearly structured. It has real figures with actual data visualizations (unlike TBG). The main blocker is finalizing the * marked preliminary results. Once those are confirmed, the paper is ready for COLM submission.

---

## Phase 1: Initial Review (2026-03-26)

---

### [WRITING] Issues

1. **W1** (Related Work, L124): "To our knowledge, DFlow-LM is the first work to successfully adapt" — overclaim; hedged but still sounds AI-generated
2. **W2** (Results §5.1, L242): "establishing a new state-of-the-art among non-autoregressive language models" — formulaic
3. **W3** (Results §5.2, L254): "demonstrating that pretrained representations provide a strong advantage for the denoising task" — filler conclusion tacked onto an otherwise clean paragraph
4. **W4** (Results §5.3, L260): "consistently outperforms all discrete diffusion baselines on every benchmark, achieving the best non-AR results" — formulaic superlative
5. **W5** (Results §5.9, L326-327): "indicating that its generated distribution closely approximates the true data distribution" — explains what MAUVE measures; unnecessary for venue audience
6. **W6** (Discussion, L365-369): Repeats hypothesis from intro almost verbatim
7. **W7** (Limitations, L376): "While DFlow-LM significantly advances the state of the art" — unnecessary self-praise before limitations
8. **W8** (Conclusion, L394-397): Repeats intro claims; "Our comprehensive evaluation" is filler
9. **W9** (Throughout): "We" sentence monotony — ~15+ "We ..." openers
10. **W10** (Results §5.4, L267): "Key findings:" followed by bullet list — could be tighter prose
11. **W11** (Abstract, L63): "compared to 42.8 for the best prior non-autoregressive method" — DiffuGPT is hybrid/continuous, not purely non-AR. Imprecise.

### [STRUCTURE] Issues

12. **S1**: §5 Results has **11 subsections** (5.1-5.11) — several are 2-3 sentences. Consolidate thin subsections (5.2+5.3, 5.7+5.8+5.9).
13. **S2**: §6 Discussion is thin — three paragraphs, one repeating the intro hypothesis.
14. **S3**: No explanation of *why* the hybrid attention mask preserves pretrained representations beyond "it preserves causal attention for the prompt." Could add a sentence about KV-cache compatibility or weight reuse.

### [FORMAT] Issues

15. **F1** (CRITICAL): `tables.tex` is NOT included via `\input{tables}` — all table references show "Table ??". Tables are defined but never loaded.
16. **F2**: `tables.tex` redefines `\tbd`, `\best`, `\secondbest` (L9-11) — will conflict with main.tex definitions. Must remove duplicates.
17. **F3**: Venue header says "NeurIPS/ICML style" (L5) and uses neurips_2025.sty — should target COLM.
18. **F4**: Page 10 is entirely blank — wasted space.

### [DATA] Issues

19. **D1**: Abstract says "42.8 for the best prior non-autoregressive method" but DiffuGPT is hybrid continuous-discrete. Best prior *discrete* diffusion is MDLM at 51.2. Needs precision.
20. **D2**: Scaling gap numbers check: 38.2-29.4=8.8 (S), 28.7-13.9=14.8 (M), 18.4-10.8=7.6 (L), 12.1-8.6=3.5 (XL). The M gap (14.8) is larger than S (8.8) — non-monotonic. Should note this.

### [EXPERIMENT] Issues (Cannot Fix)

21. **E1**: All * marked results are preliminary — need finalization
22. **E2**: No multi-seed statistics reported

---

## Priority

### Round 1 (Critical):
- **F1**: Include tables.tex and fix duplicate macro definitions
- **F3**: Fix venue reference

### Round 2 (Writing):
- Fix W1-W9: Remove formulaic language, tighten prose, reduce "We" monotony
- Fix W11/D1: Clarify DiffuGPT is hybrid, not discrete diffusion

### Round 3 (Structure):
- Fix S1: Consolidate thin results subsections
- Fix S2: Strengthen discussion

---

## Experiments Needed for COLM Acceptance

### Tier 1: Must-Do (rejection risk)
1. **Finalize all * marked results**: All preliminary results across Tables 1-11 must be finalized with proper hyperparameter tuning
2. **Multi-seed statistics**: Run 3-5 seeds for DFlow-LM and key baselines (MDLM, SEDD, DiffuGPT), report mean±std
3. **Verify scaling claims**: Confirm DFlow-LM-L (783M) and DFlow-LM-XL (1.56B) results — these are critical for the scaling narrative

### Tier 2: Strongly Recommended
4. **Human evaluation**: Add human preference evaluation (at least 100 samples rated by 3 annotators) comparing DFlow-LM vs. MDLM vs. AR GPT-2
5. **Longer sequence evaluation**: Test on sequences >256 tokens to quantify the repetition issue mentioned in limitations
6. **Multi-domain training**: Train on a combination of WikiText-103 + OpenWebText to address the cross-domain weakness

### Tier 3: Nice to Have
7. **Comparison with latest methods**: Add comparison with any discrete diffusion methods published in 2025 (e.g., improved MDLM variants)
8. **Ablation on backbone choice**: Test with a non-GPT-2 pretrained backbone (e.g., Llama-3) to show the method generalizes beyond GPT-2
9. **Inference speed benchmark**: Detailed wall-clock comparison of DFlow-LM vs. AR generation at various sequence lengths
