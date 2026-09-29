#!/usr/bin/env python3
"""Assignment 3 - static-image object detection on NVIDIA Jetson.

Default input/output files are kept beside this script:
  input.jpg -> detected_output.jpg + detection_results.txt
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect two objects of one class in a static image."
    )
    parser.add_argument("--input", default="input.jpg", help="Input image path")
    parser.add_argument(
        "--output", default="detected_output.jpg", help="Annotated output image path"
    )
    parser.add_argument(
        "--results", default="detection_results.txt", help="Text results path"
    )
    parser.add_argument(
        "--network", default="ssd-mobilenet-v2", help="jetson-inference network name"
    )
    parser.add_argument(
        "--target-class", default="person", help="Only report this class name"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Minimum confidence"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=0,
        help="Maximum results to report; 0 reports every matching detection",
    )
    return parser.parse_args()


def local_path(value):
    """Resolve relative assignment files beside this script."""
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def format_detection(index, detection, class_name):
    return "\n".join(
        [
            "Detection {}".format(index),
            "ClassName: {}".format(class_name),
            "ClassID: {}".format(detection.ClassID),
            "Confidence: {:.6f}".format(detection.Confidence),
            "Left: {:.3f}".format(detection.Left),
            "Top: {:.3f}".format(detection.Top),
            "Right: {:.3f}".format(detection.Right),
            "Bottom: {:.3f}".format(detection.Bottom),
            "Width: {:.3f}".format(detection.Width),
            "Height: {:.3f}".format(detection.Height),
            "Area: {:.3f}".format(detection.Area),
            "Center: ({:.3f}, {:.3f})".format(
                detection.Center[0], detection.Center[1]
            ),
        ]
    )


def run_jetson(args):
    from jetson_inference import detectNet
    from jetson_utils import videoOutput, videoSource

    input_path = local_path(args.input)
    output_path = local_path(args.output)
    results_path = local_path(args.results)
    if not input_path.exists():
        raise FileNotFoundError("Input image not found: {}".format(input_path))

    network = detectNet(args.network, threshold=args.threshold)
    source = videoSource(str(input_path))
    output = videoOutput(str(output_path))

    image = source.Capture()
    if image is None:
        raise RuntimeError("Could not read input image: {}".format(input_path))

    detections = network.Detect(image, overlay="box,labels,conf")
    selected = []
    for detection in detections:
        class_name = network.GetClassDesc(detection.ClassID)
        if class_name.lower() == args.target_class.lower():
            selected.append((detection, class_name))

    selected.sort(key=lambda item: item[0].Confidence, reverse=True)
    if args.max_results > 0:
        selected = selected[: args.max_results]

    lines = [
        "Assignment 3 - Object Detection Results",
        "Input: {}".format(input_path),
        "Network: {}".format(args.network),
        "Target class: {}".format(args.target_class),
        "Threshold: {:.2f}".format(args.threshold),
        "Selected detections: {}".format(len(selected)),
    ]

    for index, (detection, class_name) in enumerate(selected, start=1):
        lines.append("")
        lines.append(format_detection(index, detection, class_name))

    if args.max_results > 0 and len(selected) < args.max_results:
        lines.append("")
        lines.append(
            "WARNING: fewer than {} matching detections were found.".format(
                args.max_results
            )
        )

    result_text = "\n".join(lines) + "\n"
    print(result_text)
    results_path.write_text(result_text, encoding="utf-8")
    output.Render(image)

    print("Annotated image saved to: {}".format(output_path))
    print("Text results saved to: {}".format(results_path))


def run_windows_demo(args, import_error):
    """Show the verified reference result when Jetson modules are unavailable."""
    output_path = local_path(args.output)
    results_path = local_path(args.results)

    print("=" * 68)
    print("WINDOWS DEMO MODE")
    print("=" * 68)
    print("Jetson runtime is not installed on this computer:")
    print("  {}".format(import_error))
    print()
    print("Displaying the included, verified SSD-Mobilenet-v2 CPU reference result.")
    print("On a Jetson device, this same file automatically runs live inference.")
    print()

    if not results_path.exists():
        raise FileNotFoundError("Demo results not found: {}".format(results_path))
    if not output_path.exists():
        raise FileNotFoundError("Demo image not found: {}".format(output_path))

    print(results_path.read_text(encoding="utf-8"))
    print("Detected image: {}".format(output_path))
    print("Open detected_output.jpg in VS Code to view the bounding boxes.")


def main():
    args = parse_args()
    try:
        import jetson_inference  # noqa: F401
        import jetson_utils  # noqa: F401
    except ModuleNotFoundError as error:
        run_windows_demo(args, error)
        return

    run_jetson(args)


if __name__ == "__main__":
    main()
