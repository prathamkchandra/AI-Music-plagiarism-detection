import os
import json
import subprocess

# Path to your dataset audio files
DATASET_DIR = r"C:\Users\Pratham K Chandra\Desktop\Music-plaigarism\model\Audio_files"
OUTPUT_FILE = "fingerprints.json"

# Supported audio formats
SUPPORTED_FORMATS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")

fingerprints = {}

for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        if file.lower().endswith(SUPPORTED_FORMATS):
            file_path = os.path.join(root, file)

            try:
                print(f" Processing {file} ...")
                # Run fpcalc
                result = subprocess.run(
                    ["fpcalc", "-raw", file_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                if result.returncode != 0:
                    print(f" Error processing {file}: {result.stderr.strip()}")
                    continue

                # Extract fingerprint from output
                for line in result.stdout.splitlines():
                    if line.startswith("FINGERPRINT="):
                        fingerprint = line.split("=", 1)[1]
                        fingerprints[file] = fingerprint
                        print(f" Fingerprint extracted for {file}")
                        break

            except Exception as e:
                print(f" Exception for {file}: {e}")

# Save fingerprints to JSON
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(fingerprints, f, ensure_ascii=False, indent=4)

print(f" Fingerprints saved to {OUTPUT_FILE}")
