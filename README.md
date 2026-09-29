# Assignment 3 - Object Detection

This folder contains the code, input image, detected image, detailed results,
and PDF report required for Assignment 3.

## Files

- `your-detection.py` - Jetson-compatible Python submission.
- `input.jpg` - test image from the official `jetson-inference` repository.
- `detected_output.jpg` - image with the two selected person detections.
- `detection_results.txt` - ClassID, Confidence, bounding-box coordinates,
  Width, Height, Area, and Center for both detections.
- `Assignment_3_Report.pdf` - final report for Moodle.

## Before uploading

1. Open `Assignment_3_Report.pdf` in Adobe Acrobat Reader or another PDF
   viewer that supports form fields.
2. Fill in Student Name, Student ID, and GitHub Repository URL on page 1.
3. Save the completed PDF.
4. Upload `your-detection.py` to GitHub.
5. Upload the completed PDF to Moodle.

## Run on Jetson

Place this folder on the Jetson and run from inside it:

```bash
python3 your-detection.py \
  --input input.jpg \
  --output detected_output.jpg \
  --results detection_results.txt \
  --target-class person \
  --threshold 0.5 \
  --max-results 0
```

`--max-results 0` means that every matching detection above the confidence
threshold is printed. Use a positive value only when an explicit limit is
needed.

The program requires the `jetson_inference` and `jetson_utils` modules that
are installed when the `jetson-inference` repository is built and installed.

When the same file is run on Windows, it automatically enters a clearly
labelled demonstration mode. This prints the included verified CPU reference
results and points to `detected_output.jpg`, without claiming that Jetson
TensorRT inference ran on Windows.

## Reproducibility note

The included result was generated with the TensorFlow SSD-Mobilenet-v2 COCO
reference model on a CPU. The Jetson code loads the corresponding
`ssd-mobilenet-v2` network through `jetson-inference`. TensorRT optimization can
cause small differences in confidence scores and coordinates when the code is
run on Jetson.

Input image source:
https://github.com/dusty-nv/jetson-inference/blob/master/data/images/peds_0.jpg
