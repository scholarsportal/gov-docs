import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from pdf2image import convert_from_path
import pytesseract
from concurrent.futures import ProcessPoolExecutor
from PIL import Image, ImageOps, ImageEnhance

DEBUG = False
FORCE = False
DPI = 150 # Try to match source document resolution
CONTRAST = 0.9 # lower to remove noise, higher than one to increase contrast
LANG = "eng+fra"
MAX_WORKERS = 16


def ocr_page(args):
  i, image = args
  print(f"Processing page {i} ...")
  # convert Pillow image to grayscale
  grayscale_image = ImageOps.grayscale(image)

  # blow out see-through text
  # increase brightness
  brightness_enhancer = ImageEnhance.Brightness(grayscale_image)
  brightness = 1.0
  if CONTRAST < brightness:
    brightness = 2 - CONTRAST # 1.15 to 1.7 seems to work best
  brightened_image = brightness_enhancer.enhance(brightness)
  # decrease contrast
  contrast_enhancer = ImageEnhance.Contrast(brightened_image)
  faded_image = contrast_enhancer.enhance(CONTRAST)

  # blur image slightly to smooth out noise
  image_np = np.array(faded_image)
  blurred_np = cv2.medianBlur(image_np, 1)
  processed_image = Image.fromarray(blurred_np)
  processed_image.info['dpi'] = (DPI, DPI)

  ocr_text = pytesseract.image_to_string(processed_image, lang=LANG, config=f"--dpi {DPI}")
  return (i + 1, ocr_text)  # Return page number and OCR text as a tuple


def ocr_pdf(input_path):
  input_path = Path(input_path)
  if input_path.is_file():
    files = [input_path]
  else:
    if input_path.is_dir():
      files = list(input_path.rglob("*.pdf"))
    else:
      raise FileNotFoundError(f"Could not find {input_path}")
  # get total number of files
  num_files = len(files)
  print(f"Found {num_files} pdf files in {input_path}")
  current_file = 0
  for file in files:
    current_file += 1
    filepath = file.resolve()
    output_dir = filepath.parent.parent / "text"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_txt_path = output_dir / (filepath.stem + ".txt")
    # Skip if text file already exists
    if output_txt_path.exists() and not FORCE:
      print(f"Text file {output_txt_path} already exists. Skipping.")
      continue
    print(f"Extracting images from {file.name} ({current_file} of {num_files}) ...")
    if DEBUG: # Extract only the first 30 images
      images = convert_from_path(filepath, dpi=DPI, first_page=1, last_page=30)
    else: # Extract all images
      images = convert_from_path(filepath, dpi=DPI)
    print(f"{len(images)} pages found. Starting OCR.")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
      ocr_texts = list(executor.map(ocr_page, enumerate(images)))

    with open(output_txt_path, 'w') as f:
      for page_num, text in ocr_texts:
        f.write(f"Page {page_num}:\n{text}\n\n")  # Write page number and OCR text to the file
    print(f"OCR completed and saved to {output_txt_path}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Process images from input folder and save to output folder.")
  parser.add_argument("input_path", type=str, help="Path to the input folder containing pdf files")
  parser.add_argument("--dpi", type=int, help="DPI setting for pdf OCR (optional)")
  parser.add_argument("--contrast", type=float, help="less than 1.0 to remove faint noise (optional)")
  parser.add_argument("--lang", type=str, default='eng+fra', help="Tesseract language string for OCR (optional)")
  parser.add_argument("--debug", action='store_true', help="Enable debug mode (optional)")
  parser.add_argument("--force", action='store_true', help="Process OCR even if text file exist already (optional)")
  args = parser.parse_args()
  DEBUG = args.debug
  FORCE = args.force
  if args.dpi is not None:
    DPI = args.dpi
  if args.contrast is not None:
    CONTRAST = args.contrast
  if args.lang is not None:
    LANG = args.lang
  ocr_pdf(input_path=args.input_path)
