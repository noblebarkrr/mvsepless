import argparse
import os
from datetime import datetime

from rvc.scripts.voice_conversion import voice_pipeline

rvc_models_dir = os.path.join(os.getcwd(), "voice_models")

parser = argparse.ArgumentParser(
    description="Замена голоса", add_help=True
)
parser.add_argument("-i", "--song_input", type=str, required=True)
parser.add_argument("-m", "--model_name", type=str, required=True)
parser.add_argument("-p", "--pitch", type=float, required=True)
parser.add_argument("-ir", "--index_rate", type=float, default=0)
parser.add_argument("-fr", "--filter_radius", type=int, default=3)
parser.add_argument("-rms", "--volume_envelope", type=float, default=0.25)
parser.add_argument("-f0", "--method", type=str, default="rmvpe+")
parser.add_argument("-hop", "--hop_length", type=int, default=128)
parser.add_argument("-pro", "--protect", type=float, default=0.33)
parser.add_argument("-f0min", "--f0_min", type=int, default="50")
parser.add_argument("-f0max", "--f0_max", type=int, default="1100")
parser.add_argument("-f", "--format", type=str, default="mp3")
parser.add_argument("-o", "--output_dir", type=str, default=None)
parser.add_argument("-c", "--custom_name", type=str, default=None)
args = parser.parse_args()

model_name = args.model_name
if not os.path.exists(os.path.join(rvc_models_dir, model_name)):
    raise Exception(
        f"\033[91mМодели {model_name} не существует. "
        "Возможно, вы неправильно ввели имя.\033[0m"
    )

# Создание имени файла на основе шаблона
input_filename = os.path.splitext(os.path.basename(args.song_input))[0]
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

if args.custom_name == None:
    output_filename = f"{model_name}_{input_filename}_{current_time}_{args.method}_{args.pitch}"
else:
    output_filename = f"{args.custom_name}"
    
# Определение пути для сохранения
output_dir = args.output_dir if args.output_dir else os.getcwd()
output_path = os.path.join(output_dir, f"{output_filename}.{args.format}")

cover_path = voice_pipeline(
    uploaded_file=args.song_input,
    voice_model=model_name,
    pitch=args.pitch,
    index_rate=args.index_rate,
    filter_radius=args.filter_radius,
    volume_envelope=args.volume_envelope,
    f0_method=args.method,
    hop_length=args.hop_length,
    protect=args.protect,
    output_format=args.format,
    f0_min=args.f0_min,
    f0_max=args.f0_max,
    output_dir=output_dir,
    output_filename=output_filename
)

print("\033[1;92m\nГолос успешно заменен!\033[0m")
print(f"Файл сохранен по пути: {cover_path}")
