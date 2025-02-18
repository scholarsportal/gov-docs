import os
import csv
import requests
from tqdm import tqdm


def download_pdf(url, save_path):
  response = requests.get(url, stream=True)
  total_size_in_bytes = int(response.headers.get('content-length', 0))
  block_size = 1024  #1 Kibibyte
  progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

  with open(save_path, 'wb') as file:
    for data in response.iter_content(block_size):
      progress_bar.update(len(data))
      file.write(data)
  progress_bar.close()
  if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
    print("ERROR, something went wrong")


def main(csv_file_path, number_of_files=None):
  # Determine the directory of the CSV file
  csv_dir = os.path.dirname(os.path.abspath(csv_file_path))

  with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    total_files = number_of_files if number_of_files else len(list(reader)) -1
    current_file = 0
    csvfile.seek(0)  # Reset the file pointer to the beginning
    for row in reader:
      base_url = row['Internet Archive Link']
      if base_url == "Internet Archive Link":
        continue
      if number_of_files and current_file >= number_of_files:
        break
      # get the barcode from the last part of the base_url
      barcode = base_url.split("/")[-1]
      pdf_url = f"{base_url.replace('details', 'download')}/{barcode}.pdf"
      save_path = os.path.join(csv_dir, f"{barcode}.pdf")
      current_file += 1
      if not os.path.exists(save_path):
        print(f"Downloading file {current_file} of {total_files} at {pdf_url} ...")
        download_pdf(pdf_url, save_path)
      else:
        print(f"File already exists: {save_path}, skipping.")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Download PDF files from Internet Archive.")
  parser.add_argument("csv_file_path", help="Path to the CSV file containing the Internet Archive links")
  parser.add_argument("number_of_files", help="Optional number of files to download", type=int, default=None)
  args = parser.parse_args()
  main(args.csv_file_path, args.number_of_files)
