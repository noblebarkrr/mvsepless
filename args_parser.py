import argparse
from pathlib import Path
from i18n import _i18n
BASE_DIR = Path(__file__).resolve().parent
from audio import output_formats

def tobool(val: str | bool | int):
    if isinstance(val, int):
        return True if val >= 1 else False
    elif isinstance(val, str):
        if val in ["y", "yes", "Yes", "true", "True", "1"]:
            return True
        else:
            return False
    elif isinstance(val, bool):
        return val

class NestedAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # Разбиваем dest по точке, например 'database.host'
        group, dest = self.dest.split('.', 1)
        # Получаем или создаем вложенный Namespace
        groupspace = getattr(namespace, group, argparse.Namespace())
        # Устанавливаем значение во вложенный объект
        setattr(groupspace, dest, values)
        # Сохраняем вложенный объект в основной
        setattr(namespace, group, groupspace)

class NestedStoreTrue(argparse.Action):
    def __init__(self, option_strings, dest, default=False, help=None, **kwargs):
        # 1. Сразу при создании парсера готовим структуру во вложенном Namespace
        super().__init__(option_strings=option_strings, dest=dest, nargs=0, default=default, help=help, **kwargs)
        
    def __call__(self, parser, namespace, values, option_string=None):
        # 2. Если флаг передан, меняем False на True
        group, attr = self.dest.split('.', 1)
        groupspace = getattr(namespace, group, argparse.Namespace())
        setattr(groupspace, attr, True)
        setattr(namespace, group, groupspace)

def add_custom_sources_args(parser):
    parser.add_argument(
        "--add_custom_model_info_path", "--add-custom-model-info-path",
        "--model_info", "--model-info", "-minfo",
        type=str, dest="custom_model_info_path", default=None,
        help=_i18n("arg_custom_model_info_help")
    )
    parser.add_argument(
        "--add_custom_models_dir", "--add-custom-models-dir",
        "--models_dir", "--models-dir", "-mdir",
        type=str, dest="custom_models_dir", default=None,
        help=_i18n("arg_custom_models_dir_help")
    )
    parser.add_argument(
        "--model_source", "--model-source", "-msrc",
        type=str, dest="model_source", default=None,
        choices=["hface", "hf", "huggingface", "mscope", "ms", "modelscope", "github", "gh"],
        help=_i18n("arg_model_source_help")
    )

def parse_separator_args(add_params_args: dict = {}):
    parser = argparse.ArgumentParser(
        description=_i18n("arg_main_description"),
        epilog=_i18n("arg_main_epilog")
    )
    subparsers = parser.add_subparsers(
        title=_i18n("arg_subcommands_title"),
        dest="mode",
        description=_i18n("arg_subcommands_description"),
        help=_i18n("arg_subcommands_help")
    )
    
    # separate
    separate_parser = subparsers.add_parser(
        "separate",
        help=_i18n("arg_separate_help"),
        description=_i18n("arg_separate_description"),
        epilog=_i18n("arg_separate_epilog")
    )
    add_custom_sources_args(separate_parser)
    # custom_separate
    custom_separate_parser = subparsers.add_parser(
        "custom_separate",
        help=_i18n("arg_custom_separate_help"),
        description=_i18n("arg_custom_separate_description"),
        epilog=_i18n("arg_custom_separate_epilog")
    )
    
    # info
    info_parser = subparsers.add_parser(
        "info",
        help=_i18n("arg_info_help"),
        description=_i18n("arg_info_description"),
        epilog=_i18n("arg_info_epilog")
    )
    add_custom_sources_args(info_parser)
    # auto_ensemble
    auto_ensemble_parser = subparsers.add_parser(
        "auto_ensemble",
        help=_i18n("arg_auto_ensemble_help"),
        description=_i18n("arg_auto_ensemble_description"),
        epilog=_i18n("arg_auto_ensemble_epilog")
    )
    add_custom_sources_args(auto_ensemble_parser)
    # manual_ensemble
    manual_ensemble_parser = subparsers.add_parser(
        "manual_ensemble",
        help=_i18n("arg_manual_ensemble_help"),
        description=_i18n("arg_manual_ensemble_description"),
        epilog=_i18n("arg_manual_ensemble_epilog")
    )
    
    # subtract
    subtract_parser = subparsers.add_parser(
        "subtract",
        help=_i18n("arg_subtract_help"),
        description=_i18n("arg_subtract_description"),
        epilog=_i18n("arg_subtract_epilog")
    )

    preset_parser = subparsers.add_parser(
        "preset",
        help=_i18n("arg_preset_help"),
        description=_i18n("arg_preset_description"),
        epilog=_i18n("arg_preset_epilog")
    )
    add_custom_sources_args(preset_parser)

    # separate
    separate_parser.add_argument(
        "-i", "--i", "-input", "--input", "--input_files", "--input-files", 
        nargs="+", dest="input",
        help=_i18n("arg_input_help")
    )
    separate_parser.add_argument(
        "-o", "-out", "-output", "--output", "--output_dir", "--output-dir", 
        type=str, default=".", dest="output_dir",
        help=_i18n("arg_output_dir_help")
    )
    separate_parser.add_argument(
        "-of", "-output_fmt", "--output_format", "--output-format", 
        type=str, choices=output_formats, default=output_formats[0], dest="output_format",
        help=_i18n("arg_output_format_help", formats=", ".join(output_formats), default=output_formats[0])
    )
    separate_parser.add_argument(
        "-tm", "-tmplt", "--template", type=str, default="NAME_STEM_MODEL", dest="template",
        help=_i18n("arg_template_help", keys=_i18n("template_keys_separate"), example="NAME_STEM_MODEL")
    )
    separate_parser.add_argument(
        "-mn", "-model", "--model_name", "--model-name", 
        type=str, default="bs_6stem", dest="model_name",
        help=_i18n("arg_model_name_help")
    )
    separate_parser.add_argument(
        "-inst", "-ext_inst", "-ext-inst", "--extract_instrumental", "--extract-instrumental", 
        action="store_true", dest="extract_instrumental",
        help=_i18n("arg_extract_instrumental_help")
    )
    separate_parser.add_argument(
        "-ispec", "-spec_invert", "-spec-invert", "--use_spec_invert", "--use-spec-invert",  
        action="store_true", dest="use_spec_invert",
        help=_i18n("arg_use_spec_invert_help")
    )
    separate_parser.add_argument(
        "-iplus", "-invert_plus", "-invert-plus", "--invert_plus", "--invert-plus",  
        action="store_true", dest="invert_plus",
        help=_i18n("invert_plus")
    )
    separate_parser.add_argument(
        "-st", "--st", "-stems", "--stems", "--selected_stems", "--selected-stems", 
        nargs="*", metavar="STEM", dest="selected_stems",
        help=_i18n("arg_selected_stems_help")
    )
    separate_parser.add_argument(
        "-hi-prec", "-hi_prec", "-hi-precision", "--hi_precision", "-pref-flt", "-pref_flt", "--prefer_float", "--prefer_float",  
        action="store_true", dest="prefer_float",
        help=_i18n("prefer_float")
    )
    for param_name, param_value in add_params_args.items():
        param_type = param_value.get("type")
        default = param_value.get("default")
        separate_parser.add_argument(
            f"--{param_name}", 
            action=NestedStoreTrue if param_type == "bool" else NestedAction,
            type=None if param_type == "bool" else (int if param_type == "int" else (float if param_type == "float" else str)),
            default=default,
            dest=f"add_params.{param_name}",
            help=_i18n("arg_add_param_help")
        )

    # custom_separate
    custom_separate_parser.add_argument(
        "-i", "--i", "-input", "--input", "--input_files", "--input-files", 
        nargs="+", dest="input",
        help=_i18n("arg_input_help")
    )
    custom_separate_parser.add_argument(
        "-o", "-out", "-output", "--output", "--output_dir", "--output-dir", 
        type=str, default=".", dest="output_dir",
        help=_i18n("arg_output_dir_help")
    )
    custom_separate_parser.add_argument(
        "-of", "-output_fmt", "--output_format", "--output-format", 
        type=str, choices=output_formats, default=output_formats[0], dest="output_format",
        help=_i18n("arg_output_format_help", formats=", ".join(output_formats), default=output_formats[0])
    )
    custom_separate_parser.add_argument(
        "-tm", "-tmplt", "--template", type=str, default="NAME_STEM_MODEL", dest="template",
        help=_i18n("arg_template_help", keys=_i18n("template_keys_separate"), example="NAME_STEM_MODEL")
    )
    custom_separate_parser.add_argument(
        "-mt", "-mtype", "--model_type", "--model-type", 
        type=str, default="bs_roformer", dest="model_type",
        help=_i18n("arg_model_type_help")
    )
    custom_separate_parser.add_argument(
        "-ckpt", "--ckpt", "-checkpoint", "--checkpoint", "--checkpoint_path", "--checkpoint-path", 
        type=str, required=True, dest="checkpoint_path",
        help=_i18n("arg_checkpoint_path_help")
    )
    custom_separate_parser.add_argument(
        "-conf", "--conf", "-config", "--config", "--config_path", "--config-path", 
        type=str, required=True, dest="config_path",
        help=_i18n("arg_config_path_help")
    )
    custom_separate_parser.add_argument(
        "-inst", "-ext_inst", "-ext-inst", "--extract_instrumental", "--extract-instrumental", 
        action="store_true", dest="extract_instrumental",
        help=_i18n("arg_extract_instrumental_help")
    )
    custom_separate_parser.add_argument(
        "-ispec", "-spec_invert", "-spec-invert", "--use_spec_invert", "--use-spec-invert",  
        action="store_true", dest="use_spec_invert",
        help=_i18n("arg_use_spec_invert_help")
    )
    custom_separate_parser.add_argument(
        "-iplus", "-invert_plus", "-invert-plus", "--invert_plus", "--invert-plus",  
        action="store_true", dest="invert_plus",
        help=_i18n("invert_plus")
    )
    custom_separate_parser.add_argument(
        "-st", "--st", "-stems", "--stems", "--selected_stems", "--selected-stems", 
        nargs="*", metavar="STEM", dest="selected_stems",
        help=_i18n("arg_selected_stems_help")
    )
    custom_separate_parser.add_argument(
        "-hi-prec", "-hi_prec", "-hi-precision", "--hi_precision", "-pref-flt", "-pref_flt", "--prefer_float", "--prefer_float",  
        action="store_true", dest="prefer_float",
        help=_i18n("prefer_float")
    )
    for param_name, param_value in add_params_args.items():
        param_type = param_value.get("type")
        default = param_value.get("default")
        custom_separate_parser.add_argument(
            f"--{param_name}", 
            action=NestedStoreTrue if param_type == "bool" else NestedAction,
            type=None if param_type == "bool" else (int if param_type == "int" else (float if param_type == "float" else str)),
            default=default,
            dest=f"add_params.{param_name}",
            help=_i18n("arg_add_param_help")
        )

    iterative_ensemble_parser = subparsers.add_parser(
        "iterative_ensemble",
        help=_i18n("arg_iterative_ensemble_help"),
        description=_i18n("arg_iterative_ensemble_description"),
        epilog=_i18n("arg_iterative_ensemble_epilog")
    )
    add_custom_sources_args(iterative_ensemble_parser)
    
    iterative_ensemble_parser.add_argument(
        "-i", "--i", "-input", "--input", "--input_file", "--input-file", 
        type=str, required=True, dest="input",
        help=_i18n("arg_input_single_help")
    )
    iterative_ensemble_parser.add_argument(
        "-o", "-out", "-output", "--output", "--output_dir", "--output-dir", 
        type=str, default=".", dest="output_dir",
        help=_i18n("arg_output_dir_help")
    )
    iterative_ensemble_parser.add_argument(
        "-of", "-output_fmt", "--output_format", "--output-format", 
        type=str, choices=output_formats, default=output_formats[0], dest="output_format",
        help=_i18n("arg_output_format_help", formats=", ".join(output_formats), default=output_formats[0])
    )
    iterative_ensemble_parser.add_argument(
        "-tm", "-tmplt", "--template", type=str, default="NAME_ITER", dest="template",
        help=_i18n("arg_template_help", keys=_i18n("template_keys_iterative_ensemble"), example="NAME_ITER_ITER")
    )
    iterative_ensemble_parser.add_argument(
        "-n", "-iters", "--num_iters", "--num-iters", 
        type=int, default=4, dest="num_iters",
        help=_i18n("arg_num_iters_help")
    )
    iterative_ensemble_parser.add_argument(
        "-save_intermediate", "-save-intermediate", "--save_intermediate", "--save-intermediate", 
        action="store_true", dest="save_intermediate",
        help=_i18n("arg_save_intermediate_help")
    )
    iterative_ensemble_parser.add_argument(
        "-hi-prec", "-hi_prec", "-hi-precision", "--hi_precision", "-pref-flt", "-pref_flt", "--prefer_float", "--prefer_float",  
        action="store_true", dest="prefer_float",
        help=_i18n("prefer_float")
    )
    iterative_ensemble_flow_group = iterative_ensemble_parser.add_mutually_exclusive_group(required=True)
    iterative_ensemble_flow_group.add_argument(
        "-flow", "--flow", nargs="+", metavar="MODEL:PRIMARY_STEM:INVERT", 
        dest="flow",
        help=_i18n("arg_iterative_flow_help")
    )
    iterative_ensemble_flow_group.add_argument(
        "-json", "-preset", "-preset_json", "-preset-json", "--preset_json", "--preset-json", 
        type=str, dest="preset",
        help=_i18n("arg_preset_json_help")
    )

    # auto_ensemble
    auto_ensemble_parser.add_argument(
        "-i", "--i", "-input", "--input", "--input_file", "--input-file", 
        type=str, required=True, dest="input",
        help=_i18n("arg_input_single_help")
    )
    auto_ensemble_parser.add_argument(
        "-o", "-out", "-output", "--output", "--output_dir", "--output-dir", 
        type=str, default=".", dest="output_dir",
        help=_i18n("arg_output_dir_help")
    )
    auto_ensemble_parser.add_argument(
        "-of", "-output_fmt", "--output_format", "--output-format", 
        type=str, choices=output_formats, default=output_formats[0], dest="output_format",
        help=_i18n("arg_output_format_help", formats=", ".join(output_formats), default=output_formats[0])
    )
    auto_ensemble_parser.add_argument(
        "-tm", "-tmplt", "--template", type=str, default="NAME_TYPE_COUNT", dest="template",
        help=_i18n("arg_template_help", keys=_i18n("template_keys_auto_ensemble"), example="NAME_COUNT_TYPE")
    )
    auto_ensemble_parser.add_argument(
        "-t", "-type", "-etype", "--ensemble_type", "--ensemble-type", 
        type=str, default="avg_fft", dest="ensemble_type",
        help=_i18n("arg_ensemble_type_help")
    )
    auto_ensemble_parser.add_argument(
        "-ispec", "-spec_invert", "-spec-invert", "--use_spec_invert", "--use-spec-invert",  
        action="store_true", dest="use_spec_invert",
        help=_i18n("arg_use_spec_invert_help")
    )
    auto_ensemble_parser.add_argument(
        "-save_stems", "-save-stems", "-save_primary_stems", "--save-primary-stems", 
        action="store_true", dest="save_primary_stems",
        help=_i18n("arg_save_primary_stems_help")
    )
    auto_ensemble_parser.add_argument(
        "-hi-prec", "-hi_prec", "-hi-precision", "--hi_precision", "-pref-flt", "-pref_flt", "--prefer_float", "--prefer_float",  
        action="store_true", dest="prefer_float",
        help=_i18n("prefer_float")
    )
    auto_ensemble_flow_group = auto_ensemble_parser.add_mutually_exclusive_group(required=True)
    auto_ensemble_flow_group.add_argument(
        "-flow", "--flow", nargs="+", metavar="MODEL:PRIMARY_STEM:INVERT:WEIGHTS", 
        dest="flow",
        help=_i18n("arg_flow_help")
    )
    auto_ensemble_flow_group.add_argument(
        "-json", "-preset", "-preset_json", "-preset-json", "--preset_json", "--preset-json", 
        type=str, dest="preset",
        help=_i18n("arg_preset_json_help")
    )

    # manual_ensemble
    manual_ensemble_parser.add_argument(
        "-i", "--i", "-input", "--input", "--input_files", "--input-files", 
        nargs="+", dest="input",
        help=_i18n("arg_input_help")
    )
    manual_ensemble_parser.add_argument(
        "-o", "-out", "-output", "--output", "--output_dir", "--output-dir", 
        type=str, default=".", dest="output_dir",
        help=_i18n("arg_output_dir_help")
    )
    manual_ensemble_parser.add_argument(
        "-of", "-output_fmt", "--output_format", "--output-format", 
        type=str, choices=output_formats, default=output_formats[0], dest="output_format",
        help=_i18n("arg_output_format_help", formats=", ".join(output_formats), default=output_formats[0])
    )
    manual_ensemble_parser.add_argument(
        "-tm", "-tmplt", "--template", type=str, default="NAME_TYPE", dest="template",
        help=_i18n("arg_template_help", keys=_i18n("template_keys_manual_ensemble"), example="NAME_TYPE")
    )
    manual_ensemble_parser.add_argument(
        "-t", "-type", "-etype", "--ensemble_type", "--ensemble-type", 
        type=str, default="avg_fft", dest="ensemble_type",
        help=_i18n("arg_ensemble_type_help")
    )
    manual_ensemble_parser.add_argument(
        "-w", "-weights", "--weights", type=float, nargs="*", dest="weights",
        help=_i18n("arg_weights_help")
    )
    manual_ensemble_parser.add_argument(
        "-hi-prec", "-hi_prec", "-hi-precision", "--hi_precision", "-pref-flt", "-pref_flt", "--prefer_float", "--prefer_float",  
        action="store_true", dest="prefer_float",
        help=_i18n("prefer_float")
    )

    # subtract
    subtract_parser.add_argument(
        "-i1", "--i1", "-input1", "--input1", "--input_file1", "--input-file1", 
        type=str, required=True, dest="input_1",
        help=_i18n("arg_input1_help")
    )
    subtract_parser.add_argument(
        "-i2", "--i2", "-input2", "--input2", "--input_file2", "--input-file2", 
        type=str, required=True, dest="input_2",
        help=_i18n("arg_input2_help")
    )
    subtract_parser.add_argument(
        "-o", "-out", "-output", "--output", "--output_dir", "--output-dir", 
        type=str, default=".", dest="output_dir",
        help=_i18n("arg_output_dir_help")
    )
    subtract_parser.add_argument(
        "-of", "-output_fmt", "--output_format", "--output-format", 
        type=str, choices=output_formats, default=output_formats[0], dest="output_format",
        help=_i18n("arg_output_format_help", formats=", ".join(output_formats), default=output_formats[0])
    )
    subtract_parser.add_argument(
        "-tm", "-tmplt", "--template", type=str, default="NAME_TYPE", dest="template",
        help=_i18n("arg_template_help", keys=_i18n("template_keys_subtract"), example="NAME_TYPE")
    )
    subtract_parser.add_argument(
        "-ispec", "-spec_invert", "-spec-invert", "--use_spec_invert", "--use-spec-invert",  
        action="store_true", dest="use_spec_invert",
        help=_i18n("arg_use_spec_invert_help")
    )
    subtract_parser.add_argument(
        "-hi-prec", "-hi_prec", "-hi-precision", "--hi_precision", "-pref-flt", "-pref_flt", "--prefer_float", "--prefer_float",  
        action="store_true", dest="prefer_float",
        help=_i18n("prefer_float")
    )

    # info
    info_parser.add_argument(
        "-u", "-update", "--update", action="store_true", dest="update",
        help=_i18n("arg_update_help")
    )
    info_parser.add_argument(
        "-get", "--get", dest="get", action="store_true"
    )
    info_parser.add_argument(
        "-clear", "-clear_cache", "-clear-cache", "--clear_cache", "--clear-cache", 
        action="store_true", dest="clear_cache",
        help=_i18n("arg_clear_cache_help")
    )
    info_parser.add_argument(
        "-mn", "-model", "--model_name", "--model-name", 
        type=str, default="bs_6stem", dest="model_name",
        help=_i18n("arg_model_name_help")
    )
    info_parser.add_argument(
        "-dw", "-download", "--download", action="store_true", dest="download",
        help=_i18n("arg_download_help")
    )
    info_parser.add_argument(
        "-l", "-limit", "--limit", type=int, default=None, dest="limit",
        help=_i18n("arg_limit_help")
    )
    info_parser.add_argument(
        "-s", "-stem", "--stem", type=str, default=None, dest="stem",
        help=_i18n("arg_stem_filter_help")
    )
    info_parser.add_argument(
        "-oi", "-installed", "--only_installed", "--only-installed", 
        action="store_true", dest="only_installed",
        help=_i18n("arg_only_installed_help")
    )

    # phase_fixer
    phase_fixer_parser = subparsers.add_parser(
        "phase_fixer",
        help=_i18n("arg_phase_fixer_help"),
        description=_i18n("arg_phase_fixer_description"),
        epilog=_i18n("arg_phase_fixer_epilog")
    )

    phase_fixer_parser.add_argument(
        "-i1", "--i1", "-target", "--target", "--target_file", "--target-file",
        type=str, required=True, dest="target",
        help=_i18n("arg_phase_fixer_target_help")
    )

    phase_fixer_parser.add_argument(
        "-i2", "--i2", "-source", "--source", "--source_file", "--source-file",
        type=str, required=True, dest="source",
        help=_i18n("arg_phase_fixer_source_help")
    )

    phase_fixer_parser.add_argument(
        "-o", "-out", "-output", "--output", "--output_dir", "--output-dir",
        type=str, default=".", dest="output_dir",
        help=_i18n("arg_output_dir_help")
    )

    phase_fixer_parser.add_argument(
        "-of", "-output_fmt", "--output_format", "--output-format",
        type=str, choices=output_formats, default=output_formats[0], dest="output_format",
        help=_i18n("arg_output_format_help", formats=", ".join(output_formats), default=output_formats[0])
    )

    phase_fixer_parser.add_argument(
        "-tm", "-tmplt", "--template",
        type=str, default="NAME_TYPE", dest="template",
        help=_i18n("arg_template_help", keys=_i18n("template_keys_phase_fixer"), example="NAME_TYPE")
    )

    phase_fixer_parser.add_argument(
        "-tmag", "--transfer_magnitude", "--transfer-magnitude",
        action="store_true", dest="transfer_magnitude",
        help=_i18n("arg_transfer_magnitude_help")
    )

    phase_fixer_parser.add_argument(
        "-nphase", "--transfer_phase", "--transfer-phase",
        action="store_true", dest="transfer_phase",
        help=_i18n("arg_transfer_phase_help")
    )

    phase_fixer_parser.add_argument(
        "-nblend", "--freq_blend", "--freq-blend",
        action="store_true", dest="freq_blend",
        help=_i18n("arg_freq_blend_help")
    )

    phase_fixer_parser.add_argument(
        "-lc", "--low_cutoff", "--low-cutoff",
        type=int, default=500, dest="low_cutoff",
        help=_i18n("arg_low_cutoff_help")
    )

    phase_fixer_parser.add_argument(
        "-hc", "--high_cutoff", "--high-cutoff",
        type=int, default=5000, dest="high_cutoff",
        help=_i18n("arg_high_cutoff_help")
    )

    phase_fixer_parser.add_argument(
        "-hi-prec", "-hi_prec", "-hi-precision", "--hi_precision",
        "-pref-flt", "-pref_flt", "--prefer_float", "--prefer-float",
        action="store_true", dest="prefer_float",
        help=_i18n("prefer_float")
    )

    preset_parser.add_argument(
        "-i", "--i", "-input", "--input", "--input_files", "--input-files", 
        nargs="+", dest="input",
        help=_i18n("arg_input_help")
    )
    preset_parser.add_argument(
        "-o", "-out", "-output", "--output", "--output_dir", "--output-dir", 
        type=str, default=".", dest="output_dir",
        help=_i18n("arg_output_dir_help")
    )
    preset_parser.add_argument(
        "-of", "-output_fmt", "--output_format", "--output-format", 
        type=str, choices=output_formats, default=output_formats[0], dest="output_format",
        help=_i18n("arg_output_format_help", formats=", ".join(output_formats), default=output_formats[0])
    )
    preset_parser.add_argument(
        "-tm", "-tmplt", "--template", type=str, default="NAME_STEM_MODEL", dest="template",
        help=_i18n("arg_template_help", keys=_i18n("template_keys_separate"), example="NAME_STEM_MODEL")
    )
    preset_parser.add_argument(
        "-p", "-preset", "--preset", "--preset_path", "--preset-path", 
        type=str, required=True, dest="preset",
        help=_i18n("arg_preset_path_help")
    )
    preset_parser.add_argument(
        "-st", "--st", "-stems", "--stems", "--selected_stems", "--selected-stems", 
        nargs="*", metavar="STEM", dest="selected_stems",
        help=_i18n("arg_selected_stems_help")
    )
    preset_parser.add_argument(
        "-hi-prec", "-hi_prec", "-hi-precision", "--hi_precision", "-pref-flt", "-pref_flt", "--prefer_float", "--prefer_float",  
        action="store_true", dest="prefer_float",
        help=_i18n("prefer_float")
    )
    
    # Добавляем поддержку add_params для пресетов (так как ноды separate внутри пресета могут использовать их)
    for param_name, param_value in add_params_args.items():
        param_type = param_value.get("type")
        default = param_value.get("default")
        preset_parser.add_argument(
            f"--{param_name}", 
            action=NestedStoreTrue if param_type == "bool" else NestedAction,
            type=None if param_type == "bool" else (int if param_type == "int" else (float if param_type == "float" else str)),
            default=default,
            dest=f"add_params.{param_name}",
            help=_i18n("arg_add_param_help")
        )

    return parser.parse_args()

    return parser.parse_args()


def parse_vbach_args():
    parser = argparse.ArgumentParser(
        description=_i18n("vbach_main_description"),
        epilog=_i18n("vbach_main_epilog")
    )
    subparsers = parser.add_subparsers(
        title=_i18n("arg_subcommands_title"),
        dest="mode",
        description=_i18n("arg_subcommands_description"),
        help=_i18n("arg_subcommands_help")
    )
    
    # infer
    infer_parser = subparsers.add_parser(
        "infer",
        help=_i18n("vbach_infer_help"),
        description=_i18n("vbach_infer_description"),
        epilog=_i18n("vbach_infer_epilog")
    )
    
    # infer_custom_f0
    infer_custom_f0_parser = subparsers.add_parser(
        "infer_custom_f0",
        help=_i18n("vbach_infer_custom_f0_help"),
        description=_i18n("vbach_infer_custom_f0_description"),
        epilog=_i18n("vbach_infer_custom_f0_epilog")
    )
    
    # download_hubert
    download_hubert_parser = subparsers.add_parser(
        "download_hubert",
        help=_i18n("vbach_download_hubert_help"),
        description=_i18n("vbach_download_hubert_description"),
        epilog=_i18n("vbach_download_hubert_epilog")
    )

    # infer
    infer_parser.add_argument(
        "-i", "--i", "-input", "--input", "--input_files", "--input-files", 
        nargs="+", dest="input",
        help=_i18n("arg_input_help")
    )
    infer_parser.add_argument(
        "-o", "-out", "-output", "--output", "--output_dir", "--output-dir", 
        type=str, default=".", dest="output_dir",
        help=_i18n("arg_output_dir_help")
    )
    infer_parser.add_argument(
        "-m", "-model", "--model_path", "--model-path", 
        type=str, required=True, dest="checkpoint_path",
        help=_i18n("vbach_model_path_help")
    )
    infer_parser.add_argument(
        "-idx", "-index", "--index_path", "--index-path", 
        type=str, default="", dest="index_path",
        help=_i18n("vbach_index_path_help")
    )
    infer_parser.add_argument(
        "-p", "-pitch", "--pitch", type=int, default=0, dest="pitch",
        help=_i18n("vbach_pitch_help")
    )
    infer_parser.add_argument(
        "-f0m", "-f0_method", "--f0_method", "--f0-method", 
        type=str, default="rmvpe+", dest="f0_method",
        help=_i18n("vbach_f0_method_help")
    )
    infer_parser.add_argument(
        "-idxr", "-index_rate", "--index_rate", "--index-rate", 
        type=float, default=0.75, dest="index_rate",
        help=_i18n("vbach_index_rate_help")
    )
    infer_parser.add_argument(
        "-ve", "-volume_envelope", "--volume_envelope", "--volume-envelope", 
        type=float, default=0.25, dest="volume_envelope",
        help=_i18n("vbach_volume_envelope_help")
    )
    infer_parser.add_argument(
        "-pr", "-protect", "--protect", type=float, default=0.33, dest="protect",
        help=_i18n("vbach_protect_help")
    )
    infer_parser.add_argument(
        "-hl", "-hop_length", "--hop_length", "--hop-length", 
        type=int, default=128, dest="hop_length",
        help=_i18n("vbach_hop_length_help")
    )
    infer_parser.add_argument(
        "-emb", "-embedder", "--embedder_model", "--embedder-model", 
        type=str, default="hubert_base", dest="embedder",
        help=_i18n("vbach_embedder_help")
    )
    infer_parser.add_argument(
        "-tf", "-use_transformers", "--use_transformers", "--use-transformers", 
        action="store_true", dest="use_transformers",
        help=_i18n("vbach_use_transformers_help")
    )
    infer_parser.add_argument(
        "-of", "-output_fmt", "--output_format", "--output-format", 
        type=str, choices=output_formats, default=output_formats[0], dest="output_format",
        help=_i18n("arg_output_format_help", formats=", ".join(output_formats), default=output_formats[0])
    )
    infer_parser.add_argument(
        "-stm", "-stereo_mode", "--stereo_mode", "--stereo-mode", 
        type=str, choices=("mono", "left/right", "sim/dif"), default="mono", dest="stereo_mode",
        help=_i18n("vbach_stereo_mode_help")
    )
    infer_parser.add_argument(
        "-f0min", "--f0_min", "--f0-min", type=int, default=50, dest="f0_min",
        help=_i18n("vbach_f0_min_help")
    )
    infer_parser.add_argument(
        "-f0max", "--f0_max", "--f0-max", type=int, default=1100, dest="f0_max",
        help=_i18n("vbach_f0_max_help")
    )
    infer_parser.add_argument(
        "-chd", "-chunk_duration", "--chunk_duration", "--chunk-duration", 
        type=int, default=7, dest="chunk_duration",
        help=_i18n("vbach_chunk_duration_help")
    )
    infer_parser.add_argument(
        "-tm", "-tmplt", "--template", type=str, default="NAME_F0METHOD_PITCH", dest="template",
        help=_i18n("arg_template_help", keys=_i18n("template_keys_vbach"), example="NAME_F0METHOD_PITCH")
    )

    # infer_custom_f0
    infer_custom_f0_parser.add_argument(
        "-i", "--i", "-input", "--input", type=str, required=True, dest="input",
        help=_i18n("arg_input_single_help")
    )
    infer_custom_f0_parser.add_argument(
        "-o", "-out", "-output", "--output", "--output_dir", "--output-dir", 
        type=str, default=".", dest="output_dir",
        help=_i18n("arg_output_dir_help")
    )
    infer_custom_f0_parser.add_argument(
        "-m", "-model", "--model_path", "--model-path", 
        type=str, required=True, dest="checkpoint_path",
        help=_i18n("vbach_model_path_help")
    )
    infer_custom_f0_parser.add_argument(
        "-idx", "-index", "--index_path", "--index-path", 
        type=str, default="", dest="index_path",
        help=_i18n("vbach_index_path_help")
    )
    infer_custom_f0_parser.add_argument(
        "-p", "-pitch", "--pitch", type=int, default=0, dest="pitch",
        help=_i18n("vbach_pitch_help")
    )
    infer_custom_f0_parser.add_argument(
        "-f0f", "-f0_file", "--f0_file", "--f0-file", 
        type=str, dest="f0_file",
        help=_i18n("vbach_f0_file_help")
    )
    infer_custom_f0_parser.add_argument(
        "-idxr", "-index_rate", "--index_rate", "--index-rate", 
        type=float, default=0.75, dest="index_rate",
        help=_i18n("vbach_index_rate_help")
    )
    infer_custom_f0_parser.add_argument(
        "-ve", "-volume_envelope", "--volume_envelope", "--volume-envelope", 
        type=float, default=0.25, dest="volume_envelope",
        help=_i18n("vbach_volume_envelope_help")
    )
    infer_custom_f0_parser.add_argument(
        "-pr", "-protect", "--protect", type=float, default=0.33, dest="protect",
        help=_i18n("vbach_protect_help")
    )
    infer_custom_f0_parser.add_argument(
        "-emb", "-embedder", "--embedder_model", "--embedder-model", 
        type=str, default="hubert_base", dest="embedder",
        help=_i18n("vbach_embedder_help")
    )
    infer_custom_f0_parser.add_argument(
        "-tf", "-use_transformers", "--use_transformers", "--use-transformers", 
        action="store_true", dest="use_transformers",
        help=_i18n("vbach_use_transformers_help")
    )
    infer_custom_f0_parser.add_argument(
        "-of", "-output_fmt", "--output_format", "--output-format", 
        type=str, choices=output_formats, default=output_formats[0], dest="output_format",
        help=_i18n("arg_output_format_help", formats=", ".join(output_formats), default=output_formats[0])
    )
    infer_custom_f0_parser.add_argument(
        "-stm", "-stereo_mode", "--stereo_mode", "--stereo-mode", 
        type=str, choices=("mono", "left/right", "sim/dif"), default="mono", dest="stereo_mode",
        help=_i18n("vbach_stereo_mode_help")
    )
    infer_custom_f0_parser.add_argument(
        "-f0min", "--f0_min", "--f0-min", type=int, default=50, dest="f0_min",
        help=_i18n("vbach_f0_min_help")
    )
    infer_custom_f0_parser.add_argument(
        "-f0max", "--f0_max", "--f0-max", type=int, default=1100, dest="f0_max",
        help=_i18n("vbach_f0_max_help")
    )
    infer_custom_f0_parser.add_argument(
        "-chd", "-chunk_duration", "--chunk_duration", "--chunk-duration", 
        type=int, default=7, dest="chunk_duration",
        help=_i18n("vbach_chunk_duration_help")
    )
    infer_custom_f0_parser.add_argument(
        "-tm", "-tmplt", "--template", type=str, default="NAME_F0METHOD_PITCH", dest="template",
        help=_i18n("arg_template_help", keys=_i18n("template_keys_vbach"), example="NAME_F0METHOD_PITCH")
    )

    # download_hubert
    download_hubert_parser.add_argument(
        "-emb", "-embedder", "--embedder_model", "--embedder-model", 
        type=str, default="hubert_base", dest="embedder",
        help=_i18n("vbach_embedder_help")
    )
    download_hubert_parser.add_argument(
        "-tf", "-use_transformers", "--use_transformers", "--use-transformers", 
        action="store_true", dest="use_transformers",
        help=_i18n("vbach_use_transformers_help")
    )

    return parser.parse_args()


def parse_f0_extract():
    parser = argparse.ArgumentParser(
        description=_i18n("f0_extract_description"),
        epilog=_i18n("f0_extract_epilog")
    )
    parser.add_argument(
        "-i", "--i", "-input", "--input", 
        type=str, required=True, dest="input",
        help=_i18n("arg_input_single_help")
    )
    parser.add_argument(
        "-f0m", "-f0_method", "--f0_method", "--f0-method", 
        type=str, default="rmvpe+", dest="f0_method",
        help=_i18n("vbach_f0_method_help")
    )
    parser.add_argument(
        "-f0min", "--f0_min", "--f0-min", 
        type=int, default=50, dest="f0_min",
        help=_i18n("vbach_f0_min_help")
    )
    parser.add_argument(
        "-f0max", "--f0_max", "--f0-max", 
        type=int, default=1100, dest="f0_max",
        help=_i18n("vbach_f0_max_help")
    )
    parser.add_argument(
        "-o", "-out", "-output", "--output", "--output_path", "--output-path", 
        type=str, default=None, dest="output_path",
        help=_i18n("f0_extract_output_help")
    )
    return parser.parse_args()


def parse_app_args():
    parser = argparse.ArgumentParser(
        description=_i18n("app_description"),
        epilog=_i18n("app_epilog")
    )
    parser.add_argument(
        "--add_custom_model_info_path", "--add-custom-model-info-path",
        "--model_info", "--model-info", "-minfo",
        type=str, dest="custom_model_info_path", default=None,
        help=_i18n("arg_custom_model_info_help")
    )
    parser.add_argument(
        "--add_custom_models_dir", "--add-custom-models-dir",
        "--models_dir", "--models-dir", "-mdir",
        type=str, dest="custom_models_dir", default=None,
        help=_i18n("arg_custom_models_dir_help")
    )
    parser.add_argument(
        "--model_source", "--model-source", "-msrc",
        type=str, dest="model_source", default=None,
        choices=["hface", "hf", "huggingface", "mscope", "ms", "modelscope", "github", "gh"],
        help=_i18n("arg_model_source_help")
    )
    parser.add_argument(
        "-s", "-share", "--share", "--public", "--gradio_share", "--gradio-share", 
        action="store_true", dest="share",
        help=_i18n("app_share_help")
    )
    parser.add_argument(
        "-p", "-port", "--port", "--server_port", "--server-port", 
        type=int, default=None, dest="port",
        help=_i18n("app_port_help")
    )
    return parser.parse_args()


def parse_mvsep_api_args():
    """
    Парсер аргументов командной строки для MVSEP API.
    Покрывает все методы класса MVSEP_Client.
    """
    parser = argparse.ArgumentParser(
        description=_i18n("mvsep_api_cli_description"),  
        epilog=_i18n("mvsep_api_cli_epilog"),            
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Глобальные аргументы
    parser.add_argument(
        "--token", "-t",
        type=str, default=None,
        help=_i18n("mvsep_api_arg_token_help")           
    )
    parser.add_argument(
        "--region", "-r",
        type=str, default="main",
        choices=["main", "de", "de2", "sg", "hk"],
        help=_i18n("mvsep_api_arg_region_help")          
    )

    subparsers = parser.add_subparsers(
        title=_i18n("arg_subcommands_title"),
        dest="command",
        description=_i18n("mvsep_api_arg_commands_desc"), 
        help=_i18n("arg_subcommands_help")
    )

    # ======================================================================
    # register
    # ======================================================================
    reg_parser = subparsers.add_parser(
        "register",
        help=_i18n("mvsep_api_cmd_register_help"),       
        description=_i18n("mvsep_api_cmd_register_desc") 
    )
    reg_parser.add_argument("--name", type=str, required=True,
                            help=_i18n("mvsep_api_arg_name_help"))      
    reg_parser.add_argument("--email", type=str, required=True,
                            help=_i18n("mvsep_api_arg_email_help"))     
    reg_parser.add_argument("--password", type=str, required=True,
                            help=_i18n("mvsep_api_arg_password_help"))  

    # ======================================================================
    # login
    # ======================================================================
    login_parser = subparsers.add_parser(
        "login",
        help=_i18n("mvsep_api_cmd_login_help"),          
        description=_i18n("mvsep_api_cmd_login_desc")    
    )
    login_parser.add_argument("--email", type=str, required=True,
                              help=_i18n("mvsep_api_arg_email_help"))
    login_parser.add_argument("--password", type=str, required=True,
                              help=_i18n("mvsep_api_arg_password_help"))
    # ======================================================================
    # set_token
    # ======================================================================
    set_token_parser = subparsers.add_parser(
        "set_token",
        help=_i18n("mvsep_api_cmd_set_token_help"),
        description=_i18n("mvsep_api_cmd_set_token_desc")
    )
    set_token_parser.add_argument(
        "--token", "-t", 
        type=str, 
        required=True,
        help=_i18n("mvsep_api_arg_token_help")
    )
    # ======================================================================
    # user_info
    # ======================================================================
    subparsers.add_parser(
        "user_info",
        help=_i18n("mvsep_api_cmd_user_info_help"),      
        description=_i18n("mvsep_api_cmd_user_info_desc") 
    )

    # ======================================================================
    # check_token
    # ======================================================================
    subparsers.add_parser(
        "check_token",
        help=_i18n("mvsep_api_cmd_check_token_help"),    
        description=_i18n("mvsep_api_cmd_check_token_desc") 
    )

    # ======================================================================
    # separation_history
    # ======================================================================
    hist_parser = subparsers.add_parser(
        "separation_history",
        help=_i18n("mvsep_api_cmd_history_help"),        
        description=_i18n("mvsep_api_cmd_history_desc")  
    )
    hist_parser.add_argument("--start", type=int, default=0,
                             help=_i18n("mvsep_api_arg_start_help"))    
    hist_parser.add_argument("--limit", type=int, default=10,
                             help=_i18n("arg_limit_help"))
    hist_parser.add_argument("--show-all", action="store_true",
                             help=_i18n("mvsep_api_arg_show_all_help")) 

    # ======================================================================
    # purchases
    # ======================================================================
    purch_parser = subparsers.add_parser(
        "purchases",
        help=_i18n("mvsep_api_cmd_purchases_help"),      
        description=_i18n("mvsep_api_cmd_purchases_desc") 
    )
    purch_parser.add_argument("--limit", type=int, default=20,
                              help=_i18n("arg_limit_help"))
    purch_parser.add_argument("--offset", type=int, default=0,
                              help=_i18n("mvsep_api_arg_offset_help"))   
    purch_parser.add_argument("--status", type=str, default="all",
                              choices=["all", "completed", "pending", "failed"],
                              help=_i18n("mvsep_api_arg_status_help"))   

    # ======================================================================
    # credit_alert
    # ======================================================================
    credit_parser = subparsers.add_parser(
        "credit_alert",
        help=_i18n("mvsep_api_cmd_credit_alert_help"),   
        description=_i18n("mvsep_api_cmd_credit_alert_desc") 
    )
    credit_parser.add_argument("--set-threshold", type=int, default=None,
                               help=_i18n("mvsep_api_arg_set_threshold_help")) 

    # ======================================================================
    # premium
    # ======================================================================
    prem_parser = subparsers.add_parser(
        "premium",
        help=_i18n("mvsep_api_cmd_premium_help"),        
        description=_i18n("mvsep_api_cmd_premium_desc")  
    )
    prem_action = prem_parser.add_mutually_exclusive_group(required=True)
    prem_action.add_argument("--enable", action="store_true",
                             help=_i18n("mvsep_api_arg_enable_help"))   
    prem_action.add_argument("--disable", action="store_true",
                             help=_i18n("mvsep_api_arg_disable_help"))  

    # ======================================================================
    # long_filenames
    # ======================================================================
    lf_parser = subparsers.add_parser(
        "long_filenames",
        help=_i18n("mvsep_api_cmd_long_fn_help"),        
        description=_i18n("mvsep_api_cmd_long_fn_desc")  
    )
    lf_action = lf_parser.add_mutually_exclusive_group(required=True)
    lf_action.add_argument("--enable", action="store_true",
                           help=_i18n("mvsep_api_arg_enable_help"))
    lf_action.add_argument("--disable", action="store_true",
                           help=_i18n("mvsep_api_arg_disable_help"))

    # ======================================================================
    # news
    # ======================================================================
    news_parser = subparsers.add_parser(
        "news",
        help=_i18n("mvsep_api_cmd_news_help"),           
        description=_i18n("mvsep_api_cmd_news_desc")     
    )
    news_parser.add_argument("--lang", type=str, default="en",
                             help=_i18n("mvsep_api_arg_lang_help"))     
    news_parser.add_argument("--start", type=int, default=0,
                             help=_i18n("mvsep_api_arg_start_help"))
    news_parser.add_argument("--limit", type=int, default=10,
                             help=_i18n("arg_limit_help"))

    # ======================================================================
    # queue
    # ======================================================================
    queue_parser = subparsers.add_parser(
        "queue",
        help=_i18n("mvsep_api_cmd_queue_help"),          
        description=_i18n("mvsep_api_cmd_queue_desc")    
    )
    queue_parser.add_argument("--summary", action="store_true",
                              help=_i18n("mvsep_api_arg_summary_help")) 

    # ======================================================================
    # demo
    # ======================================================================
    demo_parser = subparsers.add_parser(
        "demo",
        help=_i18n("mvsep_api_cmd_demo_help"),           
        description=_i18n("mvsep_api_cmd_demo_desc")     
    )
    demo_parser.add_argument("--start", type=int, default=0,
                             help=_i18n("mvsep_api_arg_start_help"))
    demo_parser.add_argument("--limit", type=int, default=10,
                             help=_i18n("arg_limit_help"))
    demo_parser.add_argument("--algorithm-id", type=int, default=None,
                             help=_i18n("mvsep_api_arg_algo_id_help"))  

    # ======================================================================
    # algorithms
    # ======================================================================
    algos_parser = subparsers.add_parser(
        "algorithms",
        help=_i18n("mvsep_api_cmd_algos_help"),          
        description=_i18n("mvsep_api_cmd_algos_desc")    
    )
    algos_parser.add_argument("--raw", action="store_true",
                              help=_i18n("mvsep_api_arg_raw_help"))      
    algos_parser.add_argument("--scopes", type=str, default="single_upload",
                              help=_i18n("mvsep_api_arg_scopes_help"))   

    # ======================================================================
    # separate (create + wait + download)
    # ======================================================================
    sep_parser = subparsers.add_parser(
        "separate",
        help=_i18n("arg_separate_help"),
        description=_i18n("mvsep_api_cmd_separate_desc") 
    )
    sep_parser.add_argument("-i", "--input", type=str, required=True,
                            help=_i18n("arg_input_single_help"))
    sep_parser.add_argument("-o", "--output-dir", type=str, default=".",
                            help=_i18n("arg_output_dir_help"))
    sep_parser.add_argument("--sep-type", type=int, default=20,
                            help=_i18n("mvsep_api_arg_sep_type_help"))   
    sep_parser.add_argument("--add-opt1", type=str, default=None,
                            help=_i18n("mvsep_api_arg_add_opt_help").format(n=1)) 
    sep_parser.add_argument("--add-opt2", type=str, default=None,
                            help=_i18n("mvsep_api_arg_add_opt_help").format(n=2))
    sep_parser.add_argument("--add-opt3", type=str, default=None,
                            help=_i18n("mvsep_api_arg_add_opt_help").format(n=3))
    sep_parser.add_argument("--output-format", type=int, default=0,
                            choices=[0, 1, 2, 3, 4, 5],
                            help=_i18n("mvsep_api_arg_out_fmt_help"))    
    sep_parser.add_argument("--url", type=str, default=None,
                            help=_i18n("mvsep_api_arg_url_help"))        
    sep_parser.add_argument("--remote-type", type=str, default="direct",
                            help=_i18n("mvsep_api_arg_remote_type_help")) 
    sep_parser.add_argument("--preset-id", type=int, default=None,
                            help=_i18n("mvsep_api_arg_preset_id_help"))  
    sep_parser.add_argument("--webhook-url", type=str, default=None,
                            help=_i18n("mvsep_api_arg_webhook_help"))    
    sep_parser.add_argument("--is-demo", action="store_true",
                            help=_i18n("mvsep_api_arg_is_demo_help"))    

    # ======================================================================
    # status
    # ======================================================================
    status_parser = subparsers.add_parser(
        "status",
        help=_i18n("mvsep_api_cmd_status_help"),         
        description=_i18n("mvsep_api_cmd_status_desc")   
    )
    status_parser.add_argument("--hash", type=str, required=True,
                               help=_i18n("mvsep_api_arg_hash_help"))    

    # ======================================================================
    # download
    # ======================================================================
    dl_parser = subparsers.add_parser(
        "download",
        help=_i18n("mvsep_api_cmd_download_help"),       
        description=_i18n("mvsep_api_cmd_download_desc") 
    )
    dl_parser.add_argument("--hash", type=str, required=True,
                           help=_i18n("mvsep_api_arg_hash_help"))
    dl_parser.add_argument("-o", "--output-dir", type=str, default=".",
                           help=_i18n("arg_output_dir_help"))
    dl_parser.add_argument("--mirror", type=int, default=0,
                           help=_i18n("mvsep_api_arg_mirror_help"))      

    # ======================================================================
    # cancel
    # ======================================================================
    cancel_parser = subparsers.add_parser(
        "cancel",
        help=_i18n("mvsep_api_cmd_cancel_help"),         
        description=_i18n("mvsep_api_cmd_cancel_desc")   
    )
    cancel_parser.add_argument("--hash", type=str, required=True,
                               help=_i18n("mvsep_api_arg_hash_help"))

    # ======================================================================
    # delete
    # ======================================================================
    delete_parser = subparsers.add_parser(
        "delete",
        help=_i18n("mvsep_api_cmd_delete_help"),         
        description=_i18n("mvsep_api_cmd_delete_desc")   
    )
    delete_parser.add_argument("--hash", type=str, required=True,
                               help=_i18n("mvsep_api_arg_hash_help"))

    # ======================================================================
    # qc_add (Quality Checker)
    # ======================================================================
    qc_add_parser = subparsers.add_parser(
        "qc_add",
        help=_i18n("mvsep_api_cmd_qc_add_help"),         
        description=_i18n("mvsep_api_cmd_qc_add_desc")   
    )
    qc_add_parser.add_argument("--zipfile", type=str, required=True,
                               help=_i18n("mvsep_api_arg_zipfile_help")) 
    qc_add_parser.add_argument("--algo-name", type=str, required=True,
                               help=_i18n("mvsep_api_arg_algo_name_help")) 
    qc_add_parser.add_argument("--main-text", type=str, required=True,
                               help=_i18n("mvsep_api_arg_main_text_help")) 
    qc_add_parser.add_argument("--password", type=str, required=True,
                               help=_i18n("mvsep_api_arg_password_help"))
    qc_add_parser.add_argument("--dataset-type", type=str, default="0",
                               help=_i18n("mvsep_api_arg_dataset_type_help")) 
    qc_add_parser.add_argument("--ensemble", type=int, default=0,
                               help=_i18n("mvsep_api_arg_ensemble_help")) 

    # ======================================================================
    # qc_get
    # ======================================================================
    qc_get_parser = subparsers.add_parser(
        "qc_get",
        help=_i18n("mvsep_api_cmd_qc_get_help"),         
        description=_i18n("mvsep_api_cmd_qc_get_desc")   
    )
    qc_get_parser.add_argument("--entry-id", type=int, required=True,
                               help=_i18n("mvsep_api_arg_entry_id_help")) 

    # ======================================================================
    # qc_delete
    # ======================================================================
    qc_del_parser = subparsers.add_parser(
        "qc_delete",
        help=_i18n("mvsep_api_cmd_qc_delete_help"),      
        description=_i18n("mvsep_api_cmd_qc_delete_desc") 
    )
    qc_del_parser.add_argument("--entry-id", type=int, required=True,
                               help=_i18n("mvsep_api_arg_entry_id_help"))
    qc_del_parser.add_argument("--password", type=str, required=True,
                               help=_i18n("mvsep_api_arg_password_help"))

    # ======================================================================
    # qc_queue
    # ======================================================================
    qc_queue_parser = subparsers.add_parser(
        "qc_queue",
        help=_i18n("mvsep_api_cmd_qc_queue_help"),       
        description=_i18n("mvsep_api_cmd_qc_queue_desc") 
    )
    qc_queue_parser.add_argument("--start", type=int, default=0,
                                 help=_i18n("mvsep_api_arg_start_help"))
    qc_queue_parser.add_argument("--limit", type=int, default=10,
                                 help=_i18n("arg_limit_help"))

    # ======================================================================
    # qc_leaderboard
    # ======================================================================
    qc_lb_parser = subparsers.add_parser(
        "qc_leaderboard",
        help=_i18n("mvsep_api_cmd_qc_lb_help"),          
        description=_i18n("mvsep_api_cmd_qc_lb_desc")    
    )
    qc_lb_parser.add_argument("--dataset-type", type=str, default="0",
                              help=_i18n("mvsep_api_arg_dataset_type_help"))
    qc_lb_parser.add_argument("--start", type=int, default=0,
                              help=_i18n("mvsep_api_arg_start_help"))
    qc_lb_parser.add_argument("--limit", type=int, default=10,
                              help=_i18n("arg_limit_help"))
    qc_lb_parser.add_argument("--algo-filter", type=str, default=None,
                              help=_i18n("mvsep_api_arg_algo_filter_help")) 
    qc_lb_parser.add_argument("--sort", type=str, default=None,
                              help=_i18n("mvsep_api_arg_sort_help"))        

    # ======================================================================
    # batch (пакетная обработка через batch_inference)
    # ======================================================================
    batch_parser = subparsers.add_parser(
        "batch",
        help=_i18n("mvsep_api_cmd_batch_help"),          
        description=_i18n("mvsep_api_cmd_batch_desc")    
    )
    batch_parser.add_argument("-i", "--input", nargs="+", required=True,
                              help=_i18n("arg_input_help"))
    batch_parser.add_argument("-o", "--output-dir", type=str, default=".",
                              help=_i18n("arg_output_dir_help"))
    batch_parser.add_argument("--sep-type", type=str, required=True,
                              help=_i18n("mvsep_api_arg_sep_type_name_help")) 
    batch_parser.add_argument("--output-format", type=int, default=0,
                              choices=[0, 1, 2, 3, 4, 5],
                              help=_i18n("mvsep_api_arg_out_fmt_help"))
    batch_parser.add_argument("--add-opt1", type=str, default=None,
                              help=_i18n("mvsep_api_arg_add_opt_help").format(n=1))
    batch_parser.add_argument("--add-opt2", type=str, default=None,
                              help=_i18n("mvsep_api_arg_add_opt_help").format(n=2))
    batch_parser.add_argument("--add-opt3", type=str, default=None,
                              help=_i18n("mvsep_api_arg_add_opt_help").format(n=3))

    return parser.parse_args()