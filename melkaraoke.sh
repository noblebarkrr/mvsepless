cd /content/Mel-Band-Roformer-Vocal-Model && python inference.py --config_path /content/Mel-Band-Roformer-Vocal-Model/configs/config_vocals_mel_band_roformer.yaml --model_path /content/melmodels/MelBandRoformer.ckpt --input_folder /content/inputsep --store_dir /content/temp/

cd /content/temp && for file in *_vocals.wav; do cp "$file" /content/temp/vocals/vocals.wav; done

cd /content/Mel-Band-Roformer-Vocal-Model && python inference.py --config_path /content/Mel-Band-Roformer-Vocal-Model/configs/config_vocals_mel_band_roformer.yaml --model_path /content/melmodels/model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt --input_folder /content/temp/vocals --store_dir /content/temp/backvocals

# Organize vocal and instrumental tracks
cd /content/temp/backvocals && for file in *_vocals.wav; do cp "$file" /content/output/leadvocals.wav; done && for file in *_instrumental.wav; do cp "$file" /content/output/backvocals.wav; done

bash output.sh