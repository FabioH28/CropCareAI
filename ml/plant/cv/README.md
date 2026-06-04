# Plant CV Layer

This folder contains the plant computer-vision and digital image processing pipeline.

Use this folder for:

- preprocessing
- enhancement
- restoration
- color analysis
- leaf-first ROI isolation
- lesion segmentation
- handcrafted feature extraction
- infected-area and severity estimation
- intermediate visual outputs

Leaf isolation:

- SAM (Segment Anything) object segmentation when available — separates a green leaf from a green background, which colour cannot
- classical excess-green / colour mask as fallback (`CROPCARE_USE_SAM=0`)

Core deployed lesion segmentation (all classical, from scratch):

- thresholding
- clustering
- graph cut
- 2-of-3 consensus with a spatial-tolerance vote + a healthy-tissue colour gate (off-green or warm-hued) so healthy leaves do not read as diseased

Supplementary comparison outputs kept for analysis and presentation:

- FFT / frequency-domain views
- wavelet views
- region growing
- split-and-merge
- superpixels

Main entrypoint:

- `analyze.py`
