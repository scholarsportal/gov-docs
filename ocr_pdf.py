import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from pdf2image import convert_from_path
from pdf2image.pdf2image import pdfinfo_from_path
import pytesseract
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, cpu_count
from PIL import Image, ImageEnhance

DEBUG = False
FORCE = False
DPI = 196  # 256 for high quality, 196 for medium quality, 120 for low quality
CONTRAST = 1.1  # lower than 1.0 to reduce contrast and brigtness
LANG = "eng+fra"
MAX_WORKERS = 16


def remove_bleed_through(image):
  # Convert to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # Apply slight non-local means denoising
  denoised = cv2.fastNlMeansDenoising(gray, None, 8, 7, 21)
  # Smooth the result for cleaner output
  smoothed = cv2.medianBlur(denoised, 1)
  return smoothed


def ocr_page(args):
  i, image = args
  print(f"Processing page {i} ...")
  # adjust exposure
  if CONTRAST != 1.0:
    brightness_enhancer = ImageEnhance.Brightness(image)
    brightened_image = brightness_enhancer.enhance(CONTRAST)
    contrast_enhancer = ImageEnhance.Contrast(brightened_image)
    contrasted_image = contrast_enhancer.enhance(CONTRAST)
  else:
    contrasted_image = image
  # remove noise
  denoised_image = remove_bleed_through(np.array(contrasted_image))
  processed_image = Image.fromarray(denoised_image)
  processed_image.info['dpi'] = (DPI, DPI)
  ocr_text = pytesseract.image_to_string(processed_image, lang=LANG, config=f"--dpi {DPI}")
  return (i + 1, ocr_text)  # Return page number and OCR text as a tuple


def extract_images(pages, filepath, dpi):
  return convert_from_path(filepath, dpi=dpi, first_page=pages[0], last_page=pages[-1])


def extract_images_from_pdf(filepath, dpi, first_page=1, last_page=None):
  # set total_pages to last_page if it is not None
  if last_page is not None:
    total_pages = last_page
  else:
    total_pages = pdfinfo_from_path(filepath)['Pages']
  # Split the pages into chunks for parallel processing
  page_chunks = [
      range(i, min(i + (total_pages // MAX_WORKERS) + 1, total_pages + 1))
      for i in range(1, total_pages + 1, (total_pages // MAX_WORKERS) + 1)
  ]
  # Use a multiprocessing Pool to process the chunks
  with Pool(MAX_WORKERS) as pool:
    results = pool.starmap(extract_images, [(pages, filepath, dpi) for pages in page_chunks])
  # Flatten the list of results since each chunk is processed separately
  images = [img for result in results for img in result]
  return images


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
    if DEBUG:  # Extract only the first 30 images
      images = extract_images_from_pdf(filepath, dpi=DPI, first_page=1, last_page=30)
    else:  # Extract all images
      images = extract_images_from_pdf(filepath, dpi=DPI)
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
  parser.add_argument("--contrast",
                      type=float,
                      help="less than 1.0 to remove faint noise (optional)")
  parser.add_argument("--lang",
                      type=str,
                      default='eng+fra',
                      help="Tesseract language string for OCR (optional)")
  parser.add_argument("--debug", action='store_true', help="Enable debug mode (optional)")
  parser.add_argument("--force",
                      action='store_true',
                      help="Process OCR even if text file exist already (optional)")
  args = parser.parse_args()
  DEBUG = args.debug
  FORCE = args.force
  if args.dpi is not None:
    DPI = args.dpi
  if args.contrast is not None:
    CONTRAST = args.contrast
  if args.lang is not None:
    LANG = args.lang
  MAX_WORKERS = min(cpu_count(), MAX_WORKERS)
  print(f"Using {MAX_WORKERS} workers.")
  ocr_pdf(input_path=args.input_path)
