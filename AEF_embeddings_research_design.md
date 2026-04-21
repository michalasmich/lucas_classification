# AlphaEarth / Satellite Embedding Notes and Fusion Design for LUCAS

Date: 2026-04-20

This note summarizes what I found about the Google / DeepMind AlphaEarth Foundations (AEF) Satellite Embedding dataset and then turns that into a practical fusion design recommendation for your LUCAS image classification pipeline.

Primary sources reviewed:

- Google Earth Engine dataset catalog: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
- AlphaEarth Foundations paper: https://arxiv.org/abs/2507.22291
- Intro tutorial: https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-01-introduction
- Supervised classification tutorial: https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-03-supervised-classification

Additional official source reviewed:

- GCS README for the released embedding files: https://developers.google.com/earth-engine/guides/aef_on_gcs_readme

Additional fusion papers reviewed:

- Baltrusaitis et al., TPAMI 2019: https://doi.org/10.1109/TPAMI.2018.2798607
- Joze et al., CVPR 2020 (MMTM): https://openaccess.thecvf.com/content_CVPR_2020/html/Joze_MMTM_Multimodal_Transfer_Module_for_CNN_Fusion_CVPR_2020_paper.html
- Perez et al., AAAI 2018 (FiLM): https://doi.org/10.1609/aaai.v32i1.11671
- Poelsterl et al., MICCAI 2021 / Wolf et al., NeuroImage 2022 (DAFT): https://doi.org/10.1007/978-3-030-87240-3_66 and https://doi.org/10.1016/j.neuroimage.2022.119505
- Hayat et al., MLRH 2022 (MedFuse): https://proceedings.mlr.press/v182/hayat22a.html
- Wang et al., MIDL 2024: https://proceedings.mlr.press/v250/wang24c.html
- Parikh et al., AMIA 2024: https://pmc.ncbi.nlm.nih.gov/articles/PMC11141810/
- Du et al., ECCV 2024 (TIP): https://doi.org/10.1007/978-3-031-72633-0_27
- Chen et al., ISPRS JPRS 2017: https://doi.org/10.1016/j.isprsjprs.2016.12.008

## 1. What the released AEF dataset is

The released Earth Engine dataset is a global annual embedding field at 10 m resolution where each pixel is represented by a 64-dimensional vector rather than physical spectral bands. Google describes these vectors as learned geospatial embeddings that summarize annual surface conditions and temporal trajectories from multiple Earth observation sources rather than directly measurable channels. See the dataset catalog and intro tutorial:

- https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
- https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-01-introduction

Key properties of the released annual dataset:

- Spatial resolution: 10 m per pixel.
- Feature dimension: 64 bands, named `A00` to `A63`.
- Geometry: each 64D vector should be treated as one coordinate in embedding space; individual bands are not independently interpretable.
- Temporal granularity: annual summaries.
- Intended downstream uses: clustering, classification, regression, change detection, similarity search.

The Earth Engine catalog says the vectors are unit-length and do not require extra normalization for downstream analysis. It also states they are linearly composable, meaning they can be aggregated to coarser spatial units while preserving useful distance relationships:

- https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL

That last point is important for your use case: these are not just arbitrary latent activations from an unknown model dump. Google is explicitly positioning them as analysis-ready features intended for shallow downstream models.

## 2. What AEF is learning from

The AlphaEarth Foundations paper describes AEF as a general geospatial embedding model that assimilates spatial, temporal, and measurement context across multiple sources:

- https://arxiv.org/abs/2507.22291

From the paper and catalog, AEF uses multiple data streams, including:

- Optical data such as Sentinel-2 and Landsat.
- Radar data such as Sentinel-1 and PALSAR-2.
- LiDAR via GEDI.
- Environmental variables.
- Spatially precise alignment with geotagged text.

Relevant sources:

- Paper main text and supplement: https://arxiv.org/abs/2507.22291
- Dataset catalog summary: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL

Why this matters for LUCAS:

- Your RGB orthophoto or VHR crop is a high-resolution single-date visual observation.
- AEF is a lower-resolution but multi-sensor and multi-temporal summary.
- So the AEF vector is best understood as contextual evidence, not as a replacement for the image.

## 3. Important release details that matter operationally

### 3.1 Earth Engine product versus raw GCS files

The Earth Engine dataset is analysis-ready, but the raw files on GCS are stored differently. The GCS README says:

- Each COG file stores signed 8-bit values per channel.
- These raw values must be de-quantized.
- The de-quantization is not simple `/127`; it is:
  `((values / 127.5) ** 2) * sign(values)`
- `-128` is reserved as `NoData`.

Source:

- https://developers.google.com/earth-engine/guides/aef_on_gcs_readme

Practical implication:

- If you sampled points directly from the Earth Engine image collection, you likely already have float embeddings in `[-1, 1]`.
- If you downloaded raw COG pixels from GCS, you must de-quantize first.

### 3.2 The released product is annual

The annual release summarizes one calendar year. The catalog says the annual image uses `system:time_start` / `system:time_end` to represent the summarized year:

- https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL

This is good for stable or phenological signals, but it introduces a real modeling question for your task because your image crop is single-date and your labels are visually interpreted at a point.

### 3.3 Current versioning nuance

There is a small documentation nuance worth remembering.

- The dataset catalog page I read says the annual collection spans 2017-01-01 to 2025-01-01 and notes a 2025-11-17 update where dataset version 1.1 regenerated the 2017 layer with additional Sentinel-1 acquisitions.
- The GCS README, last updated 2026-01-29, says the bucket contains annual embeddings for 2017 through 2025 inclusive.

Sources:

- https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
- https://developers.google.com/earth-engine/guides/aef_on_gcs_readme

My interpretation:

- The product is evolving, and the GCS and catalog docs are not perfectly synchronized.
- If you publish with it, record the exact year and product version you used.

## 4. How Google suggests using the embeddings

The tutorials are very consistent about the intended workflow:

- Treat the 64 dimensions as predictors.
- Sample embedding vectors at training points.
- Use shallow downstream models.

The supervised classification tutorial explicitly demonstrates point sampling from the annual embedding mosaic and then training a classifier directly on the 64 features, using kNN as a good low-shot baseline:

- https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-03-supervised-classification

The intro tutorial emphasizes that the embeddings capture the yearly trajectory, so two pixels that look similar at one moment but differ in timing can still be separated in embedding space:

- https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-01-introduction

The regression tutorial uses the same 64D embeddings as predictors in a Random Forest regression pipeline:

- https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-04-regression

The similarity-search tutorial uses dot products between embedding vectors to retrieve similar sites:

- https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-05-similarity-search

So the official and quasi-official story is very clear: these embeddings are intended to work as compact transferable predictors without heavy downstream feature engineering.

## 5. What the AEF paper says about low-shot transfer and LUCAS

This part is especially relevant for you.

The paper evaluates AEF under very light transfer methods, specifically k-nearest neighbors and linear layers fitted to the features:

- https://arxiv.org/abs/2507.22291

That is strongly aligned with your instinct not to heavily process the 64D vector.

The paper also includes LUCAS-derived evaluations. In the supplement, the authors state:

- LUCAS land cover was treated as an instantaneous observation of state, so the valid time was set to the observation time.
- LUCAS land use was treated as an integrated observation of state, so the valid period was a one-year window centered on the observation time.

Source:

- https://arxiv.org/abs/2507.22291

This is a very important methodological clue.

Interpretation:

- The AEF authors themselves distinguish between tasks that should use point-in-time summaries and tasks that benefit from annual integration.
- Your released annual embeddings are likely closer to their land-use style setup than to a strict single-date land-cover observation.

So if you use the annual product for your image-based land-cover classifier, there is a real chance of temporal mismatch between:

- what your high-resolution image sees on one date,
- what the annual embedding summarizes over the year,
- and what the photointerpreted label is intended to represent.

That does not make the annual embedding useless, but it does mean it should be treated as auxiliary context, not as the primary evidence.

## 6. Does your proposed fusion idea make sense?

Short answer: yes, as a proof of concept it makes good sense.

Your proposed idea was:

- keep the current image backbone and image head,
- concatenate the raw 64D embedding with the final 256D image feature,
- replace the final classifier input dimension `256 -> 320`,
- predict 8 classes from that concatenated vector.

I think this is a sound baseline.

Why it makes sense:

- It respects the intended use of AEF as a shallow-transfer feature.
- It avoids running the 64D vector through a deeper nonlinear branch that could distort its geometry.
- It lets the classifier decide whether the embedding adds complementary evidence for each class.
- It is easy to ablate against image-only and embedding-only baselines.

### Important observation

This is an inference from model algebra, not a claim from a paper:

`Linear([h_img ; e_aef])` is mathematically equivalent to:

`W_img * h_img + W_aef * e_aef + b`

So your idea can be implemented in a cleaner and more interpretable way as:

- one linear head on image features,
- one linear head on the raw 64D AEF vector,
- sum the two logits.

That gives the same functional family as concatenation followed by one linear layer, but it is easier to inspect and regularize.

## 7. My recommended first fusion design

For your first experiment, I would not use extra activations or extra hidden layers on the AEF branch.

Recommended formulation:

- Image branch: your current backbone and current `... -> 256 -> ReLU -> dropout` feature pathway.
- AEF branch: one single linear layer `64 -> 8`.
- Fusion: add the image logits and embedding logits.

Optional but useful addition:

- Add one learnable scalar gate `lambda` on the AEF logits:
  `final_logits = img_logits + lambda * aef_logits`

Why I like this best:

- It keeps the AEF side as close as possible to a linear probe.
- It preserves interpretability.
- It lets the model automatically downweight the embedding branch if it is not useful.
- It avoids the training instability that can come from forcing 64D context features and learned 256D image features into the same numeric regime too early.

### Do you need an activation on the AEF branch?

My answer: not for the first experiment.

Reasons:

- The AEF release is explicitly meant for shallow downstream use.
- Google states the vectors are unit-length and already analysis-ready:
  https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
- The AEF paper evaluates linear probes directly on the features:
  https://arxiv.org/abs/2507.22291

Adding ReLU or a hidden MLP on the embedding branch may help later, but it stops being the conservative test you want.

### Do you need to normalize the embeddings?

For the released Earth Engine float embeddings, I would start with no extra normalization.

Reason:

- Google explicitly says the vectors are unit-length and do not require additional normalization:
  https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL

The only exception is if your downloaded values are raw GCS int8 pixels. In that case you must de-quantize them first:

- https://developers.google.com/earth-engine/guides/aef_on_gcs_readme

## 8. What the literature says about fusion strategies

### 8.1 Late fusion is a legitimate and strong baseline

Baltrusaitis et al. describe the classic early / late / hybrid taxonomy and note that late fusion offers flexibility, supports heterogeneous models, and handles missing modalities more easily, though it gives up low-level cross-modal interaction:

- https://doi.org/10.1109/TPAMI.2018.2798607

That matches your case very well because:

- the modalities are heterogeneous,
- the image branch is already strong,
- you do not want to distort the embedding.

### 8.2 Simple image-tabular baselines are common

The ECCV 2024 TIP paper notes that many prior image-tabular methods used shallow MLPs and simple fusion strategies, and it explicitly compares against concatenation fusion, max fusion, channel-wise multiplication, and DAFT:

- https://doi.org/10.1007/978-3-031-72633-0_27

That does not mean simple fusion is optimal, but it does mean your plan is absolutely standard as a baseline.

### 8.3 When more interaction helps, modulation often beats raw concatenation

Several papers suggest that if you go beyond a basic baseline, the next step is usually not a deep MLP on the tabular vector but some form of feature-wise conditioning or modulation:

- FiLM uses feature-wise affine modulation conditioned by another modality:
  https://doi.org/10.1609/aaai.v32i1.11671
- MMTM uses squeeze/excitation-style transfer between modality streams and emphasizes minimal changes to pretrained branches:
  https://openaccess.thecvf.com/content_CVPR_2020/html/Joze_MMTM_Multimodal_Transfer_Module_for_CNN_Fusion_CVPR_2020_paper.html
- DAFT dynamically rescales and shifts CNN feature maps using tabular data and outperformed competing methods in image+tabular medical tasks:
  https://doi.org/10.1016/j.neuroimage.2022.119505

My interpretation:

- If the conservative late-fusion baseline is too weak, a modulation-style method is the most principled next step.
- Among those, DAFT is especially relevant because it is explicitly about fusing image and low-dimensional tabular information without replacing the CNN backbone.

### 8.4 Recent image+metadata studies do show benefit from auxiliary non-image features

Examples:

- Wang et al. 2024 found that fusion models outperformed image-only skin malignancy models, and a multiplicative fusion variant performed best:
  https://proceedings.mlr.press/v250/wang24c.html
- Parikh et al. 2024 compared late, early, and joint fusion for chest X-ray plus EHR and found fusion better than single-modality models, with joint fusion best in their setup:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11141810/
- Hayat et al. 2022 proposed a simple multimodal fusion module and reported gains over more complex strategies in their asynchronous medical setting:
  https://proceedings.mlr.press/v182/hayat22a.html

The consistent lesson is not that one fusion method always wins. The lesson is:

- auxiliary structured information often helps,
- simple baselines are worth trying first,
- but the best method depends on whether the auxiliary modality acts more like context, calibration, or a true interacting signal.

## 9. My assessment for your specific LUCAS setup

### 9.1 Is this likely to help at all?

My honest view: yes, it could help, but I would expect modest and class-dependent gains rather than a dramatic jump.

Why it could help:

- AEF carries annual phenology and multi-sensor context not present in a single RGB crop.
- AEF includes radar and other signals that may add structure where RGB alone is ambiguous.
- The AEF paper shows strong land-cover performance under low-parameter transfer, including LUCAS-derived evaluations:
  https://arxiv.org/abs/2507.22291

Why it may help only a little, or even hurt in some cases:

- Your label is tied to a specific point and a specific interpreted imagette.
- Your image crop is high-resolution and visually centered around the classification target.
- The annual AEF vector is 10 m and summarizes the whole year.
- Therefore it may encode broader contextual or seasonal information that is not always the decisive evidence for your label.

### 9.2 Where I think it is most likely to help

This is my inference, not a direct statement from a source:

- Permanent crops: likely candidate, because annual phenology and context can help.
- Grassland / shrubs / bare: maybe helpful, because annual multi-sensor evidence may add separability where RGB texture is unstable.
- Artificial / water / wooded areas: maybe small gains, but these may already be easy from imagery and leave less headroom.

### 9.3 Where I think it may be least helpful

Also my inference:

- Cases where the central point label depends on fine local geometry visible only in the orthophoto or VHR crop.
- Cases where the annual embedding reflects a broader land-use regime but the image shows a local exception.
- Cases where the single-date image and the annual summary genuinely disagree because of management, disturbance, or seasonal timing.

### 9.4 Overall judgment

I would treat AEF as a context branch.

I would not let it dominate the decision.

I would not use it to replace or strongly modulate the image features on the first attempt.

## 10. Recommended experiment ladder

I would run the experiments in this order.

### Stage 0: essential baselines

- Image-only model: your current pipeline.
- AEF-only model: multinomial logistic regression or a single linear layer `64 -> 8`.
- Fused model: image branch plus AEF linear branch with additive logit fusion.

Why:

- If the AEF-only model is weak, fusion may still help, but expectations should stay moderate.
- If the AEF-only model is surprisingly strong, then the auxiliary signal is real.

### Stage 1: safest proof of concept

- `img_logits = head_img(h_img)`
- `aef_logits = head_aef(e_aef)`
- `final_logits = img_logits + lambda * aef_logits`

Settings:

- Initialize `lambda = 0` or `1`.
- No activation on the AEF branch.
- No hidden layer on the AEF branch.
- Keep the image branch unchanged.

### Stage 2: still conservative, but slightly more flexible

Only if Stage 1 helps a little:

- replace scalar `lambda` with per-class gates,
- or use a diagonal scaling on the 64 embedding channels before the linear probe.

This still preserves the idea that the branch remains mostly linear.

### Stage 3: modulation-style interaction

Only if Stage 1 shows promise but plateaus:

- FiLM-style modulation,
- DAFT-style affine modulation,
- MMTM-style channel recalibration.

These are much more expressive, but they also move away from your original desire to preserve embedding semantics.

### Stage 4: only if you want a stronger research paper section

Compare:

- additive logit fusion,
- concatenation + linear classifier,
- multiplicative fusion,
- one modulation model.

That would give you a clean "simple-to-advanced" ablation table for a paper.

## 11. Experimental cautions specific to your project

### 11.1 Use the same split logic by `lucas_id`

Since `lucas_id` is your primary key, every branch must use exactly the same point-level split.

### 11.2 Match the year as closely as possible

If your image date and AEF year are mismatched, the fusion result becomes harder to interpret.

### 11.3 Report class-wise deltas, not only OA

Because I expect gains to be class-specific, overall accuracy alone may hide the real behavior.

### 11.4 Run a permutation sanity check

This is not from a paper; it is a practical recommendation.

Do one control where you randomly shuffle AEF vectors across training samples while keeping labels fixed. If the fused model still "improves," then the gain is probably coming from regularization effects or leakage in the pipeline, not real complementary signal.

### 11.5 Keep the image branch dominant at first

Your scientific story is still image-based land-cover classification.

So the right question is:

- does AEF add a useful side-signal?

not:

- can AEF redefine the task?

## 12. Bottom line

My recommendation is:

- Yes, test AEF fusion.
- Start with a very conservative design.
- Do not put an MLP on the embedding branch in the first round.
- Do not add an activation on the embedding branch in the first round.
- Prefer additive logit fusion with a linear AEF probe over a less interpretable concatenation block.

If it helps, the next most defensible upgrade is a modulation method such as DAFT or FiLM-style conditioning, not a generic deep tabular branch.

My best prior for your task is:

- the method is plausible,
- the gains are likely modest,
- the gains may be stronger for certain classes than for overall accuracy,
- and the biggest risk is temporal/contextual mismatch between annual AEF summaries and single-date high-resolution imagettes.

## 13. Source list

Official dataset and tutorials:

- Google Earth Engine dataset catalog:
  https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
- AlphaEarth Foundations paper:
  https://arxiv.org/abs/2507.22291
- Introduction tutorial:
  https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-01-introduction
- Supervised classification tutorial:
  https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-03-supervised-classification
- Regression tutorial:
  https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-04-regression
- Similarity-search tutorial:
  https://developers.google.com/earth-engine/tutorials/community/satellite-embedding-05-similarity-search
- GCS README:
  https://developers.google.com/earth-engine/guides/aef_on_gcs_readme

Fusion literature:

- Baltrusaitis, Ahuja, Morency. Multimodal Machine Learning: A Survey and Taxonomy. TPAMI 2019.
  https://doi.org/10.1109/TPAMI.2018.2798607
- Joze et al. MMTM: Multimodal Transfer Module for CNN Fusion. CVPR 2020.
  https://openaccess.thecvf.com/content_CVPR_2020/html/Joze_MMTM_Multimodal_Transfer_Module_for_CNN_Fusion_CVPR_2020_paper.html
- Perez et al. FiLM: Visual Reasoning with a General Conditioning Layer. AAAI 2018.
  https://doi.org/10.1609/aaai.v32i1.11671
- Poelsterl et al. Combining 3D Image and Tabular Data via the Dynamic Affine Feature Map Transform. MICCAI 2021.
  https://doi.org/10.1007/978-3-030-87240-3_66
- Wolf, Poelsterl, Wachinger. DAFT: A universal module to interweave tabular data and 3D images in CNNs. NeuroImage 2022.
  https://doi.org/10.1016/j.neuroimage.2022.119505
- Hayat, Geras, Shamout. MedFuse: Multi-modal fusion with clinical time-series data and chest X-ray images. MLRH 2022.
  https://proceedings.mlr.press/v182/hayat22a.html
- Wang et al. Skin Malignancy Classification Using Patients' Skin Images and Meta-data: Multimodal Fusion for Improving Fairness. MIDL 2024.
  https://proceedings.mlr.press/v250/wang24c.html
- Parikh et al. Comparative Analysis of Fusion Strategies for Imaging and Non-imaging Data. AMIA 2024.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11141810/
- Du et al. TIP: Tabular-Image Pre-training for Multimodal Classification with Incomplete Data. ECCV 2024.
  https://doi.org/10.1007/978-3-031-72633-0_27
- Chen, Huang, Xu. Multi-source remotely sensed data fusion for improving land cover classification. ISPRS JPRS 2017.
  https://doi.org/10.1016/j.isprsjprs.2016.12.008
