def str2bool(value):
    true_values = ['true', '1', 'yes', 'y', 't', 'on']
    false_values = ['false', '0', 'no', 'n', 'f', 'off']
    
    if isinstance(value, str):
        value_lower = value.lower().strip()
        if value_lower in true_values:
            return True
        elif value_lower in false_values:
            return False
        else:
            raise ValueError(f"Не удалось преобразовать '{value}' в булево значение")
    elif isinstance(value, bool):
        return value
    else:
        return bool(value)