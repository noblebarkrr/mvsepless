cd /content/temp || true  # Attempt to change directory, continue if it doesn't exist
for file in *_instrumental.wav; do mv "$file" /content/output/instrumental.wav; done
cd /content/temp || true  # Attempt to change directory, continue if it doesn't exist
for file in *_vocals.wav; do cp "$file" /content/output/fullvocals.wav; done
cp /content/temp/*_vocals.wav /content/vocals.wav  # Copy all remaining vocal files