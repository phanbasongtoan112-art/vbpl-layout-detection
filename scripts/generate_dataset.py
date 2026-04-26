#!/usr/bin/env python
"""
Render crawled HTML snapshots into a synthetic layout dataset for YOLO.

Workflow:
1. Read static HTML snapshots from `data/raw_html/`.
2. Extract the marked VBPL content root from each snapshot.
3. Re-wrap that content into a light-weight HTML page that keeps the original
   CSS links and inline styles, but removes the live app shell and scripts.
4. Render the page in headless Chrome.
5. Capture a stitched full-page screenshot.
6. Export layout annotations in YOLO format.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import tempfile
import time
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from bs4 import BeautifulSoup
from PIL import Image
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm
import yaml

CLASS_NAMES = ["title", "section", "paragraph", "table", "list"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
HTML_GLOB = "doc_*.html"
RENDER_CSS = """
html { scroll-behavior: auto !important; }
body {
    margin: 0;
    padding: 0;
    background: #ffffff !important;
    color: #111827;
}
.dataset-shell {
    width: 1200px;
    max-width: 1200px;
    min-height: 100vh;
    margin: 0 auto;
    padding: 24px 40px 72px;
    box-sizing: border-box;
    background: #ffffff;
}
.dataset-shell [data-hidden-for-dataset="true"] {
    display: none !important;
}
img, svg, canvas, video {
    max-width: 100%;
}
table {
    border-collapse: collapse;
    max-width: 100%;
}
.table-scroll-wrapper {
    overflow: visible !important;
}
"""

NORMALIZE_PAGE_SCRIPT = r"""
return (() => {
  document.documentElement.style.scrollBehavior = "auto";
  document.body.style.background = "#ffffff";
  document.querySelectorAll("*").forEach((element) => {
    const style = window.getComputedStyle(element);
    if (style.position === "sticky" || style.position === "fixed") {
      element.setAttribute("data-hidden-for-dataset", "true");
      element.style.position = "static";
      element.style.top = "auto";
      element.style.bottom = "auto";
    }
  });
  return true;
})();
"""

ANNOTATION_SCRIPT = r"""
return (() => {
  const root =
    document.querySelector(".dataset-shell [data-vbpl-content-root='true']") ||
    document.querySelector("[data-vbpl-content-root='true']") ||
    document.querySelector(".dataset-shell") ||
    document.body;

  const visible = (element) => {
    if (!element) {
      return false;
    }
    const rect = element.getBoundingClientRect();
    const style = window.getComputedStyle(element);
    return (
      rect.width >= 8 &&
      rect.height >= 8 &&
      style.display !== "none" &&
      style.visibility !== "hidden" &&
      Number(style.opacity || "1") > 0
    );
  };

  const textLength = (element) => ((element.innerText || "").trim()).length;

  const boxFor = (element, label) => {
    const rect = element.getBoundingClientRect();
    return {
      label,
      x: rect.left + window.scrollX,
      y: rect.top + window.scrollY,
      width: rect.width,
      height: rect.height,
      text: (element.innerText || "").trim().slice(0, 250),
    };
  };

  const uniqueElements = (elements) => {
    const seen = new Set();
    const output = [];
    elements.forEach((element) => {
      if (!element || seen.has(element)) {
        return;
      }
      seen.add(element);
      output.push(element);
    });
    return output;
  };

  const headingCandidates = uniqueElements(
    Array.from(
      root.querySelectorAll(
        "[data-vbpl-title='true'], h1, h2, h3, h4, [class*='prov-title'], [class*='prov-chapter'], [class*='prov-part'], [class*='prov-section'], [class*='prov-article']"
      )
    )
  ).filter((element) => visible(element) && textLength(element) >= 2);

  let titleElement = root.querySelector("[data-vbpl-title='true']");
  if (!titleElement && headingCandidates.length > 0) {
    const ranked = headingCandidates
      .map((element, index) => {
        const rect = element.getBoundingClientRect();
        const fontSize = parseFloat(window.getComputedStyle(element).fontSize || "16");
        const topBias = Math.max(0, 1200 - Math.max(rect.top, 0)) / 12;
        return {
          element,
          score: fontSize * 14 + topBias + (index === 0 ? 50 : 0),
        };
      })
      .sort((left, right) => right.score - left.score);
    titleElement = ranked[0]?.element || null;
  }

  const sectionCandidates = uniqueElements(
    Array.from(
      root.querySelectorAll(
        "h2, h3, h4, [class*='prov-chapter'], [class*='prov-part'], [class*='prov-section'], [class*='prov-article']"
      )
    )
  ).filter((element) => {
    if (!visible(element) || element === titleElement) {
      return false;
    }
    const text = (element.innerText || "").trim().toLowerCase();
    if (text.length < 2) {
      return false;
    }
    return true;
  });

  const paragraphCandidates = uniqueElements(
    Array.from(root.querySelectorAll("p, [class*='prov-content']"))
  ).filter((element) => {
    if (!visible(element)) {
      return false;
    }
    if (element.closest("table")) {
      return false;
    }
    const text = (element.innerText || "").trim();
    if (text.length < 20) {
      return false;
    }
    if (!element.matches("p") && element.querySelector("p, [class*='prov-content']")) {
      return false;
    }
    return true;
  });

  const tableCandidates = uniqueElements(Array.from(root.querySelectorAll("table"))).filter(visible);
  const listCandidates = uniqueElements(Array.from(root.querySelectorAll("ul, ol"))).filter(visible);

  const boxes = [];
  if (titleElement && visible(titleElement)) {
    boxes.push(boxFor(titleElement, "title"));
  }
  sectionCandidates.forEach((element) => boxes.push(boxFor(element, "section")));
  paragraphCandidates.forEach((element) => boxes.push(boxFor(element, "paragraph")));
  tableCandidates.forEach((element) => boxes.push(boxFor(element, "table")));
  listCandidates.forEach((element) => boxes.push(boxFor(element, "list")));

  return boxes;
})();
"""


def configure_logging(log_file: Path, verbose: bool) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("generate_dataset")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def local_http_server(root: Path) -> Iterator[str]:
    handler = partial(QuietHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    host, port = server.server_address

    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def build_driver(headless: bool, window_width: int, viewport_height: int) -> WebDriver:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--hide-scrollbars")
    options.add_argument("--lang=vi-VN")
    options.add_argument(f"--window-size={window_width},{max(viewport_height, 900)}")
    driver = webdriver.Chrome(options=options)
    return driver


def wait_for_render(driver: WebDriver, timeout: int) -> None:
    WebDriverWait(driver, timeout).until(
        lambda browser: browser.execute_script("return document.readyState") == "complete"
    )
    try:
        driver.execute_async_script(
            """
            const callback = arguments[0];
            if (document.fonts && document.fonts.ready) {
              document.fonts.ready.then(() => setTimeout(callback, 250)).catch(() => setTimeout(callback, 250));
            } else {
              setTimeout(callback, 250);
            }
            """
        )
    except Exception:
        time.sleep(0.25)


def locate_content_root(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    for selector in (
        "[data-vbpl-content-root='true']",
        ".preview-content",
        "[class*='preview-content']",
        "main",
        "body",
    ):
        node = soup.select_one(selector)
        if node is not None:
            return node
    return None


def build_render_html(raw_html_path: Path) -> str:
    soup = BeautifulSoup(raw_html_path.read_text(encoding="utf-8"), "html.parser")
    content_root = locate_content_root(soup)
    if content_root is None:
        raise ValueError(f"Could not find a content root in {raw_html_path.name}")

    head = soup.head or BeautifulSoup("<head></head>", "html.parser").head
    head_chunks: List[str] = ['<meta charset="utf-8">']
    if not head.find("base"):
        head_chunks.append(f'<base href="https://vbpl.vn/">')

    seen_stylesheets: set[str] = set()
    for node in head.find_all(["meta", "base", "link", "style"]):
        if node.name == "meta":
            if node.get("charset") or node.get("name") in {"viewport", "vbpl-original-url", "vbpl-crawled-at"}:
                head_chunks.append(str(node))
        elif node.name == "base":
            head_chunks.append(str(node))
        elif node.name == "style":
            head_chunks.append(str(node))
        elif node.name == "link":
            rel = " ".join(node.get("rel", [])).lower()
            href = node.get("href", "")
            if "stylesheet" in rel and href and href not in seen_stylesheets:
                seen_stylesheets.add(href)
                head_chunks.append(str(node))

    content_html = str(content_root)

    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"vi\">\n"
        "<head>\n"
        + "\n".join(head_chunks)
        + "\n<style id=\"dataset-render-style\">"
        + RENDER_CSS
        + "</style>\n"
        "</head>\n"
        "<body>\n"
        "<main class=\"dataset-shell\">\n"
        + content_html
        + "\n</main>\n"
        "</body>\n"
        "</html>\n"
    )


def measure_page(driver: WebDriver) -> Tuple[int, int]:
    width, height = driver.execute_script(
        """
        return [
          Math.ceil(Math.max(
            document.documentElement.scrollWidth,
            document.body ? document.body.scrollWidth : 0,
            document.documentElement.clientWidth
          )),
          Math.ceil(Math.max(
            document.documentElement.scrollHeight,
            document.body ? document.body.scrollHeight : 0,
            document.documentElement.clientHeight
          ))
        ];
        """
    )
    return int(width), int(height)


def set_viewport(driver: WebDriver, width: int, height: int) -> Tuple[int, int]:
    driver.set_window_size(width + 40, height + 120)
    time.sleep(0.15)
    inner_width, inner_height = driver.execute_script("return [window.innerWidth, window.innerHeight];")
    return int(inner_width), int(inner_height)


def scroll_positions(total_height: int, viewport_height: int) -> List[int]:
    positions: List[int] = []
    current = 0
    while True:
        positions.append(current)
        if current + viewport_height >= total_height:
            break
        next_value = min(current + viewport_height, max(total_height - viewport_height, 0))
        if next_value <= current:
            break
        current = next_value
    return positions


def capture_full_page(
    driver: WebDriver,
    output_path: Path,
    requested_width: int,
    requested_viewport_height: int,
) -> Tuple[int, int]:
    page_width, page_height = measure_page(driver)
    viewport_width, viewport_height = set_viewport(
        driver,
        max(page_width, requested_width),
        min(max(page_height, 900), requested_viewport_height),
    )
    page_width, page_height = measure_page(driver)
    positions = scroll_positions(page_height, viewport_height)
    canvas = Image.new("RGB", (page_width, page_height), "white")

    for y in positions:
        actual_y = int(
            driver.execute_script("window.scrollTo(0, arguments[0]); return Math.round(window.scrollY);", y)
        )
        time.sleep(0.15)
        tile = Image.open(BytesIO(driver.get_screenshot_as_png())).convert("RGB")
        crop_width = min(tile.width, page_width, viewport_width)
        remaining = max(page_height - actual_y, 0)
        crop_height = min(tile.height, remaining)
        if crop_height <= 0:
            continue
        if crop_height < tile.height:
            tile = tile.crop((0, tile.height - crop_height, crop_width, tile.height))
        else:
            tile = tile.crop((0, 0, crop_width, crop_height))
        canvas.paste(tile, (0, actual_y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return canvas.size


def clip_box(box: dict, page_width: int, page_height: int) -> Optional[dict]:
    x1 = max(0.0, float(box["x"]))
    y1 = max(0.0, float(box["y"]))
    x2 = min(float(page_width), x1 + float(box["width"]))
    y2 = min(float(page_height), y1 + float(box["height"]))
    width = x2 - x1
    height = y2 - y1
    if width < 8 or height < 8:
        return None
    return {
        "label": box["label"],
        "x": x1,
        "y": y1,
        "width": width,
        "height": height,
        "text": box.get("text", ""),
    }


def iou(left: dict, right: dict) -> float:
    left_x1 = left["x"]
    left_y1 = left["y"]
    left_x2 = left_x1 + left["width"]
    left_y2 = left_y1 + left["height"]
    right_x1 = right["x"]
    right_y1 = right["y"]
    right_x2 = right_x1 + right["width"]
    right_y2 = right_y1 + right["height"]

    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height
    if inter_area <= 0:
        return 0.0

    left_area = left["width"] * left["height"]
    right_area = right["width"] * right["height"]
    union = left_area + right_area - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def deduplicate_boxes(boxes: List[dict]) -> List[dict]:
    deduped: List[dict] = []
    for candidate in sorted(boxes, key=lambda item: (item["label"], item["y"], item["x"], -item["width"] * item["height"])):
        duplicate = False
        for existing in deduped:
            if existing["label"] == candidate["label"] and iou(existing, candidate) >= 0.95:
                duplicate = True
                break
        if not duplicate:
            deduped.append(candidate)
    return deduped


def to_yolo_line(box: dict, page_width: int, page_height: int) -> str:
    class_id = CLASS_TO_ID[box["label"]]
    x_center = (box["x"] + box["width"] / 2.0) / page_width
    y_center = (box["y"] + box["height"] / 2.0) / page_height
    width = box["width"] / page_width
    height = box["height"] / page_height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def write_classes_file(output_dir: Path) -> None:
    (output_dir / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")


def write_split_files(output_dir: Path, image_paths: List[Path], train_ratio: float, seed: int) -> None:
    shuffled = list(image_paths)
    random.Random(seed).shuffle(shuffled)
    split_index = max(1, int(len(shuffled) * train_ratio)) if shuffled else 0
    if len(shuffled) >= 2:
        split_index = min(split_index, len(shuffled) - 1)
    train_paths = shuffled[:split_index]
    val_paths = shuffled[split_index:] if len(shuffled) > 1 else shuffled

    train_txt = "\n".join(path.resolve().as_posix() for path in train_paths)
    val_txt = "\n".join(path.resolve().as_posix() for path in val_paths)
    (output_dir / "train.txt").write_text(train_txt + ("\n" if train_txt else ""), encoding="utf-8")
    (output_dir / "val.txt").write_text(val_txt + ("\n" if val_txt else ""), encoding="utf-8")

    dataset_yaml = {
        "path": output_dir.resolve().as_posix(),
        "train": "train.txt",
        "val": "val.txt",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    (output_dir / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate a synthetic YOLO dataset from crawled VBPL HTML.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root / "data" / "raw_html",
        help="Directory containing crawled HTML snapshots",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "synthetic_dataset",
        help="Directory where images and labels will be written",
    )
    parser.add_argument("--window-width", type=int, default=1200, help="Browser content width in pixels")
    parser.add_argument("--viewport-height", type=int, default=1600, help="Viewport height for stitched capture")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument("--wait-timeout", type=int, default=20, help="Selenium wait timeout in seconds")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for the number of HTML files to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing images and labels")
    parser.add_argument("--headless", dest="headless", action="store_true", help="Run Chrome headless (default)")
    parser.add_argument("--no-headless", dest="headless", action="store_false", help="Show Chrome window")
    parser.set_defaults(headless=True)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose console logging")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = configure_logging(Path(__file__).resolve().parents[1] / "scripts" / "generate_dataset.log", args.verbose)

    input_dir = args.input_dir
    output_dir = args.output_dir
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    html_files = sorted(input_dir.glob(HTML_GLOB))
    if args.limit and args.limit > 0:
        html_files = html_files[: args.limit]

    if not html_files:
        logger.error("No HTML files matching %s were found in %s", HTML_GLOB, input_dir)
        return 1

    logger.info("Preparing to render %d HTML file(s) from %s", len(html_files), input_dir)
    write_classes_file(output_dir)

    processed_images: List[Path] = []
    annotations_manifest_path = output_dir / "annotations.jsonl"

    driver = build_driver(args.headless, args.window_width, args.viewport_height)

    try:
        with tempfile.TemporaryDirectory(prefix="vbpl_render_") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            with local_http_server(tmp_dir) as base_url:
                for html_path in tqdm(html_files, desc="Rendering HTML", unit="file"):
                    image_path = images_dir / f"{html_path.stem}.png"
                    label_path = labels_dir / f"{html_path.stem}.txt"

                    if image_path.exists() and label_path.exists() and not args.overwrite:
                        processed_images.append(image_path)
                        continue

                    try:
                        render_html = build_render_html(html_path)
                    except Exception as exc:
                        logger.exception("Failed to build render HTML for %s: %s", html_path.name, exc)
                        continue

                    wrapper_path = tmp_dir / html_path.name
                    wrapper_path.write_text(render_html, encoding="utf-8")

                    try:
                        driver.get(f"{base_url}/{wrapper_path.name}")
                        wait_for_render(driver, args.wait_timeout)
                        driver.execute_script(NORMALIZE_PAGE_SCRIPT)
                        raw_boxes = driver.execute_script(ANNOTATION_SCRIPT) or []
                        page_width, page_height = capture_full_page(
                            driver,
                            image_path,
                            requested_width=args.window_width,
                            requested_viewport_height=args.viewport_height,
                        )
                    except TimeoutException:
                        logger.warning("Timed out while rendering %s", html_path.name)
                        continue
                    except Exception as exc:
                        logger.exception("Failed to render %s: %s", html_path.name, exc)
                        continue

                    boxes: List[dict] = []
                    for raw_box in raw_boxes:
                        label = raw_box.get("label")
                        if label not in CLASS_TO_ID:
                            continue
                        clipped = clip_box(raw_box, page_width, page_height)
                        if clipped is not None:
                            boxes.append(clipped)

                    boxes = deduplicate_boxes(boxes)
                    yolo_lines = [to_yolo_line(box, page_width, page_height) for box in boxes]
                    label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
                    processed_images.append(image_path)

                    manifest_record = {
                        "source_html": html_path.name,
                        "image": image_path.resolve().as_posix(),
                        "label": label_path.resolve().as_posix(),
                        "width": page_width,
                        "height": page_height,
                        "objects": len(boxes),
                    }
                    with annotations_manifest_path.open("a", encoding="utf-8") as manifest_handle:
                        manifest_handle.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")

    finally:
        try:
            driver.quit()
        except WebDriverException:
            pass

    all_images = sorted(images_dir.glob("*.png"))
    write_split_files(output_dir, all_images, train_ratio=args.train_ratio, seed=args.seed)
    logger.info("Dataset generation complete. Images=%d Labels=%d", len(all_images), len(list(labels_dir.glob('*.txt'))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
