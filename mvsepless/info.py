from separator import Separator, unsupported_models
import argparse

def get_list_models(limit: None | int = None, stem: None | str = None):
    
    separator = Separator()

    models = separator.get_mn()
    if stem:
        models = [model for model in models if (stem in separator.get_stems(model) or stem.lower() in separator.get_stems(model) or stem.upper() in separator.get_stems(model) or stem.capitalize() in separator.get_stems(model) or stem.title() in separator.get_stems(model))]
    if limit:
        models = models[:limit]

    f_key, s_key = "Имя модели", "Выходные стемы"

    filename_width = max(len("Имя модели"), max(len(model) for model in models))
    stems_width = max(len("Выходные стемы"), max(len(", ".join(separator.get_stems(model))) for model in models))

    print("|-", "-" * filename_width, "-+-", "-" * stems_width, "-|", sep="")
    print(f"| {'Имя модели':<{filename_width}} | {'Выходные стемы':<{stems_width}} |")
    print("|-", "-" * filename_width, "-+-", "-" * stems_width, "-|", sep="")
    for model in models:
        stems = ", ".join(separator.get_stems(model))
        print(f"| {model:<{filename_width}} | {stems:<{stems_width}} |")
        print("|-", "-" * filename_width, "-+-", "-" * stems_width, "-|", sep="")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Список моделей вместе со всеми стемами"
    )
    parser.add_argument("--stem", default=None, type=str, help="Фильтрация по выбранному стему")
    parser.add_argument("--limit", default=None, type=int, help="Лимит отображаемых моделей")
    args = parser.parse_args()
    get_list_models(limit=args.limit, stem=args.stem)
