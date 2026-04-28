# PersonCapture HDR — target-person crops and high-quality HDR-to-SDR PNG stills from video

PersonCapture HDR builds image datasets of one specific person from video and includes a dedicated HDR-to-SDR still export path for high-quality PNG/JPG crops from HDR footage.

The current primary workflow is the Qt GUI in `person_capture.gui_app`; the older `person_capture.main` CLI is still present, but it is a simpler legacy path and does not cover the GUI pre-scan cache, HDR export, advanced crop composition, curation, or most runtime controls.

Two things are central to the current project:

- **Target-person dataset capture:** face-driven ArcFace identity matching, pre-scan span pruning, crop composition, and optional curation.
- **HDR still export:** full-resolution HDR source-frame crop export to SDR PNG/JPG, with a Windows WIC conversion path and cleanup for false saturated speckles seen in some HDR AVIF still render paths.

The current high-precision preset is face-driven: ArcFace is enabled, ReID is disabled, and SCRFD is the default face detector. This keeps identity decisions anchored to face embeddings instead of body-feature drift.

---

## Current pipeline

1. **Reference loading**
   - Loads the target reference image and builds an ArcFace reference bank.
   - Optional bank growth is normally done during pre-scan under strict gates.
   - Runtime bank growth is off by default to avoid drift.

2. **Pre-scan**
   - Samples the video at `prescan_stride` using an optional low-resolution decode path (`prescan_decode_max_w`) and analysis downscale (`prescan_max_width`).
   - Detects candidate faces, compares them to the reference bank with ArcFace distance, and builds kept time spans.
   - Can reuse a persistent cache in `prescan_cache_dir` according to `prescan_cache_mode` (`auto`, `refresh`, `off`).

3. **Main pass**
   - Processes only the kept spans when pre-scan is enabled.
   - Uses face identity first. YOLO person detection is still available for person boxes and composition, but `skip_yolo_when_faceonly` can avoid YOLO work when a visible face is already sufficient.
   - Uses temporal lock settings to stabilize short runs of accepted frames.

4. **Crop composition**
   - Scores the configured ratio list per accepted target (`ratio`, for example `2:3,3:2,1:1`).
   - Builds close/upper/body crops from identity boxes when `compose_crop_enable` is on.
   - Applies smart lateral search and final guards for face scale, side margins, top headroom, body inclusion, black borders, and minimum crop height.

5. **Export**
   - Writes primary crops under the selected output directory.
   - For HDR footage, can export full-resolution SDR PNG/JPG crops through the HDR still path instead of relying on the low-resolution preview path.
   - Optional annotations, debug dumps, HDR source archives, and curated subsets are controlled independently.

---

## Outputs

Default output root: `output`

- Primary crops: `output/crops`
- Optional annotated previews: `output/annot`
- Optional debug data: `output/debug` or the configured `debug_dir`
- Optional pre-scan cache: `prescan_cache` or the configured `prescan_cache_dir`
- Optional TensorRT engines/timing data: `trt_cache` or the configured `trt_cache_root`
- Optional curated selection: produced by the Curate tab / curator path when enabled

---

## Quick start on Windows

```bat
cd /d C:\Users\marre\source\repos\person_capture
py -3.12 -m venv env
call env\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
python -m person_capture.gui_app
```

The repository also includes `start_person_capture.bat`, which activates `env`, sets the TensorRT/CUDA DLL path assumptions used by the current Windows setup, writes `last_run.log`, checks that ONNX Runtime exposes `TensorrtExecutionProvider`, and launches the GUI module.

Optional TensorRT/ONNX Runtime setup is documented in `README_onnxruntime_1.24_trt_windows.md`.

---

## Legacy CLI path

The CLI still exists for a minimal OpenCV-style run:

```bat
python -m person_capture.main ^
  --video path\to\video.mp4 ^
  --ref path\to\person.jpg ^
  --out output ^
  --ratio 2:3 ^
  --frame-stride 2 ^
  --min-det-conf 0.35 ^
  --face-thresh 0.32 ^
  --device cuda
```

Do not use this CLI as documentation for the GUI behavior. It does not implement the current GUI pre-scan, HDR, cache, advanced cropper, or curator pipeline.

---

## Detection and identity

### Person detector

`PersonDetector` uses Ultralytics YOLO for class `person`, with `yolo_model`, `min_det_conf`, and `device` controlling the model, threshold, and CPU/CUDA placement. The loader keeps Ultralytics cache/settings under the repo-local `.ultralytics` directory where possible and uses CUDA half precision when available.

### Face detector

`FaceEmbedder` supports:

- `scrfd_10g_bnkps` / `scrfd_2.5g_bnkps` through InsightFace SCRFD ONNX; this is the current preferred path.
- YOLOv8-face weights when a YOLO face model is selected.

Important face settings:

- `face_det_conf`: candidate face detector confidence.
- `face_det_pad`: expands the person box before face detection.
- `face_fullframe_when_missed`: tries full-frame face detection when per-person detection misses.
- `face_fullframe_imgsz`: full-frame detector image size.
- `rot_adaptive`, `rot_every_n`, `rot_after_hit_frames`, `fast_no_face_imgsz`: adaptive rotated/low-size passes to recover hard faces without paying the full cost every frame.

### Identity decision

ArcFace distance is the primary identity signal. Lower distance is better.

Current recommended behavior:

- `use_arcface = true`
- `disable_reid = true`
- `learn_bank_runtime = false`
- `require_face_if_visible = true`
- `drop_reid_if_any_face_match = true`

`match_mode` can still be `either`, `both`, `face_only`, or `reid_only`, but with `disable_reid = true` the practical path is face-driven.

---

## Pre-scan behavior

Pre-scan is controlled by `prescan_enable`. When enabled, the main pass is restricted to the spans found by pre-scan.

Core controls:

- `prescan_stride`: sample cadence in source frames.
- `prescan_decode_max_w`: optional decoder-level downscale for pre-scan.
- `prescan_max_width`: analysis-frame downscale after decode.
- `prescan_hdr_preview`: whether the Vulkan HDR preview is driven during pre-scan. Keep off for speed unless debugging preview behavior.
- `prescan_face_conf`: face detector candidate threshold, not target-person probability.
- `prescan_fd_enter`: ArcFace distance gate to enter a span.
- `prescan_fd_add`: stricter ArcFace distance gate for adding a face to the reference bank.
- `prescan_fd_exit`: ArcFace distance hysteresis gate for leaving a span.
- `prescan_add_cooldown_samples`: minimum sample gap between bank additions.
- `prescan_bank_max`, `prescan_diversity_dedup_cos`, `prescan_replace_margin`, `prescan_weights`: bank selection and replacement behavior.
- `prescan_pad_sec`, `prescan_bridge_gap_sec`, `prescan_min_segment_sec`: span padding, merge, and minimum duration.
- `prescan_boundary_refine_sec`, `prescan_refine_stride_min`, `prescan_trim_pad`, `prescan_refine_budget_sec`, `prescan_skip_trailing_refine`: boundary refinement controls.
- `prescan_cache_mode`: `auto` to reuse matching cache, `refresh` to rebuild, `off` to disable cache.

`prescan_face_conf` is clamped internally to a sane detector range. Setting it to `1.0` is effectively unreachable for SCRFD-style detector confidence and can suppress candidates. Very low values can admit noisy boxes, which then makes ArcFace gating unreliable. Use it as a detector threshold, not as an identity-confidence slider.

### FD9 skip

A best face distance near `9.0` means no usable face embedding was available. The fd9 skip controls are an early-out speed path for empty/no-face stretches:

- `prescan_fd9_skip`: enable skip/probe behavior after repeated fd≈9 samples.
- `prescan_fd9_grace`: number of consecutive fd≈9 samples before skipping starts.
- `prescan_fd9_probe_period`: how often to run a real probe while skipping.

This can speed empty regions, but it can also delay re-entry if the target appears immediately after a skipped sample. Disable it when debugging pre-scan misses.

---

## Crop composition

The current cropper is not just “expand person box to ratio”. It combines identity boxes, face scale targets, person association, and frame saliency.

Key controls:

- `ratio`: comma-separated ratio list evaluated per accepted frame.
- `compose_crop_enable`: enables the current composition path.
- `compose_detect_person_for_face`: associates face hits with YOLO person boxes when needed.
- `compose_close_face_h_frac`, `compose_upper_face_h_frac`, `compose_body_face_h_frac`: target face-height fractions for close/portrait/body outputs.
- `compose_body_every_n`, `compose_person_detect_cadence`: cadence for body-shot bias and person association.
- `smart_crop_enable`, `smart_crop_steps`, `smart_crop_side_search_frac`, `smart_crop_use_grad`: local crop search / saliency behavior.
- `crop_face_side_margin_frac`, `crop_top_headroom_max_frac`, `crop_bottom_min_face_heights`, `crop_penalty_weight`: crop penalty terms.
- `crop_head_side_pad_frac`, `crop_head_top_pad_frac`, `crop_head_bottom_pad_frac`: head/hair/chin protection around detected face bounds.
- `face_max_frac_in_crop`, `face_min_frac_in_crop`, `crop_min_height_frac`: anti-overzoom / anti-underzoom guards.
- `side_guard_drop_enable`, `side_guard_drop_factor`: final post-trim side-margin safety check.
- `auto_crop_borders`, `border_threshold`, `border_scan_frac`: black-border handling.

---

## HDR still export

The HDR export path is a main feature, not just a preview helper.

The goal is to produce SDR stills from HDR video that preserve strong contrast, dark detail, highlights, and color density while avoiding the washed-out look common in simple HDR-to-SDR conversions. The current tuned path uses Windows WIC-style still rendering for primary PNG output and then repairs only impossible saturated salt pixels that can appear in some HDR AVIF/WIC render cases.

HDR preview and final crop export are separate paths:

- `hdr_passthrough`: Vulkan HDR preview path. This is preview-only and should not be treated as a prerequisite for final screencap quality.
- `hdr_screencap_fullres`: writes primary crops from original-resolution source frames.
- `hdr_archive_crops`: additionally writes source HDR crop archives.
- `hdr_crop_format`: `avif` or `mkv` for archive crops.
- `hdr_sdr_output_format`: `png` or `jpg` for primary SDR crops. PNG is preferred for maximum still quality.
- `hdr_sdr_conversion`: `windows_wic` or `ffmpeg`.
- `hdr_wic_speckle_cleanup`: repairs impossible saturated blue/red/magenta salt pixels after Windows WIC conversion.
- `hdr_avif_wic_display_compat`: writes optional AVIF archives from the WIC-compatible display conversion by default.
- `hdr_sdr_quality`, `hdr_sdr_tonemap`, `hdr_sdr_gamut_mapping`, `hdr_sdr_contrast_recovery`, `hdr_sdr_peak_detect`, `hdr_sdr_allow_inaccurate_fallback`: full-resolution SDR render quality controls.
- `hdr_export_timeout_sec`, `hdr_archive_timeout_sec`: export timeouts.

Set `PC_HDR_AVIF_SOURCE_ARCHIVE=1` only when you explicitly want raw source-HDR AVIF archives for debugging/comparison. The default archive behavior is viewer-compatible because Windows’ HDR AVIF rendering can create false saturated speckles in dark regions.

Recommended HDR still settings for the current tuned path:

- `hdr_screencap_fullres = true`
- `hdr_sdr_conversion = "windows_wic"`
- `hdr_sdr_output_format = "png"`
- `hdr_wic_speckle_cleanup = true`
- `hdr_avif_wic_display_compat = true`
- `hdr_sdr_quality = "madvr_like"`
- `hdr_sdr_tonemap = "auto"`

---

## Performance controls

- `frame_stride`: main-pass frame cadence.
- `preview_every`, `preview_max_dim`, `preview_fps_cap`, `seek_preview_peek_every`: UI preview throttling.
- `seek_fast`, `seek_max_grabs`: fast seek behavior.
- `async_save`, `async_save_wait`, `save_fsync`, `jpg_quality`: crop write behavior.
- `log_interval_sec`: progress/log update cadence.
- `ff_hwaccel`: FFmpeg hardware decode mode for HDR path (`cuda` or `off`).
- `skip_yolo_when_faceonly`: skips YOLO when a face-only decision is already available.
- `prescan_cache_mode`: use `auto` for reuse, `refresh` when changing pre-scan-affecting settings, `off` when debugging cache behavior.

---

## Curator

The curator scores already-produced crops and selects a diverse subset.

Important controls:

- `curate_enable`
- `curate_max_images`
- `curate_fd_gate`
- `curate_cos_face_dedup`
- `curate_phash_dedup`
- `curate_lambda`
- `curate_weights`
- `curate_bucket_quota`
- `curate_use_yaw_quota`

It combines identity/quality scoring with diversity selection and pHash/face-embedding deduplication.

---

## Current solid preset

`solidpreset.json` is a tuned GUI preset, not a generic default file. At the time of this README, it uses:

- `ratio = "2:3,3:2,1:1"`
- `frame_stride = 2`
- `face_thresh = 0.45`
- `use_arcface = true`
- `disable_reid = true`
- `face_visible_uses_quality = false`
- `prescan_enable = true`
- `prescan_stride = 24`
- `prescan_decode_max_w = 384`
- `prescan_cache_mode = "refresh"`
- `prescan_fd9_skip = true`
- `hdr_screencap_fullres = true`
- `hdr_sdr_conversion = "windows_wic"`
- `hdr_sdr_output_format = "png"`
- `hdr_wic_speckle_cleanup = true`
- `hdr_avif_wic_display_compat = true`
- `async_save = true`
- `skip_yolo_when_faceonly = true`

If local GUI settings disagree with the preset, the GUI settings/preset you load at runtime win.

---

## Troubleshooting

### Pre-scan does not trigger

Check these first:

- Do not set `prescan_face_conf` to `1.0`; that is a detector confidence and can suppress all candidates.
- Disable `prescan_fd9_skip` while debugging target re-entry misses.
- Use `prescan_cache_mode = "refresh"` after changing pre-scan-affecting settings.
- Verify that `prescan_fd_enter`, `prescan_fd_add`, and `prescan_fd_exit` are ArcFace distances; lower means stricter identity match.
- If `face_visible_uses_quality` is on, low-quality detected faces may not count as visible. Turn it off if you want any detected face to count for visibility gating.

### Pre-scan is too slow

- Keep `prescan_hdr_preview = false` unless diagnosing preview.
- Use `prescan_decode_max_w` and `prescan_max_width` to keep pre-scan frames small.
- Use `prescan_cache_mode = "auto"` once settings are stable.
- Keep fd9 skip enabled for long empty sections, but disable it when validating missed-entry bugs.

### Main pass is too slow

- Keep `async_save = true` and `save_fsync = false` unless debugging write failures.
- Use `skip_yolo_when_faceonly = true` when ReID is disabled and face identity is the desired path.
- Increase `frame_stride` only if you can tolerate fewer sampled frames.
- Reduce preview cost with `preview_max_dim`, `preview_every`, and `preview_fps_cap`.

### TensorRT DLLs are not found

Set the GUI TensorRT folder or `TRT_LIB_DIR` to the TensorRT `lib` directory containing `nvinfer.dll` and `nvinfer_plugin.dll`. The included launcher expects a path like:

```bat
D:\tensorrt\TensorRT-10.13.3.9\lib
```

Verify providers:

```bat
python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
```

Expected provider list includes `TensorrtExecutionProvider`, `CUDAExecutionProvider`, and `CPUExecutionProvider` when the TensorRT setup is active.

### HDR preview is black

- Enable debug logging and check the `HDR: P010 ... min/max` lines.
- If Y/UV min and max are zero, the passthrough pipe is feeding empty planes.
- Validate plane stride/shape before upload.
- Set `PC_HDR_TRACE_UPLOAD=1` to trace the Vulkan upload helper.
- Remember that preview and final crop export are separate paths; a preview issue does not automatically mean final full-resolution crop export is using the same broken path.

### HDR crops show colored speckles

For primary PNG/JPG crops, keep:

- `hdr_sdr_conversion = "windows_wic"`
- `hdr_wic_speckle_cleanup = true`
- `hdr_avif_wic_display_compat = true` for optional AVIF archives meant for normal viewing

Use `hdr_speckle_diag` and `hdr_speckle_diag_dir` only for diagnostics.

---

## Legal

You are responsible for the content you process. InsightFace, ONNX Runtime, TensorRT, CUDA, Ultralytics, OpenCLIP, and model weights are licensed by their owners. Do not redistribute NVIDIA binaries.
