import gradio as gr
import os

THEMES = {
    "gamma": gr.themes.Citrus(
    primary_hue="teal",
    secondary_hue="blue",
    neutral_hue="blue",
    spacing_size="sm",
    font=[gr.themes.GoogleFont('Montserrat'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
    ),

    "beta": gr.themes.Default(
    primary_hue="violet",
    secondary_hue="cyan",
    neutral_hue="blue",
    spacing_size="sm",
    text_size="sm",
    font=[gr.themes.GoogleFont("Tektur"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
    body_text_color="*neutral_950",
    body_text_color_subdued="*neutral_500",
    background_fill_primary="*neutral_200",
    background_fill_primary_dark="*neutral_800",
    border_color_accent="*primary_950",
    border_color_accent_dark="*neutral_700",
    border_color_accent_subdued="*primary_500",
    border_color_primary="*primary_800",
    border_color_primary_dark="*neutral_400",
    color_accent_soft="*primary_100",
    color_accent_soft_dark="*neutral_800",
    link_text_color="*secondary_700",
    link_text_color_active="*secondary_700",
    link_text_color_hover="*secondary_800",
    link_text_color_visited="*secondary_600",
    link_text_color_visited_dark="*secondary_700",
    block_background_fill="*background_fill_secondary",
    block_background_fill_dark="*neutral_950",
    block_label_background_fill="*secondary_400",
    block_label_text_color="*neutral_800",
    panel_background_fill="*background_fill_primary",
    checkbox_background_color="*background_fill_secondary",
    checkbox_label_background_fill_dark="*neutral_900",
    input_background_fill_dark="*neutral_900",
    input_background_fill_focus="*neutral_100",
    input_background_fill_focus_dark="*neutral_950",
    button_small_radius="*radius_sm",
    button_secondary_background_fill="*neutral_400",
    button_secondary_background_fill_dark="*neutral_500",
    button_secondary_background_fill_hover_dark="*neutral_950",
    ),

    "alpha": gr.themes.Base(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Montserrat"), "Arial", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Roboto Mono"), "Courier New", "monospace"]
    ).set(
    button_primary_background_fill="#3a7bd5",
    button_primary_background_fill_hover="#2c65c0",
    button_primary_text_color="#ffffff",
    input_background_fill="#ffffff",
    input_border_color="#d0d0d6",
    block_background_fill="#ffffff",
    border_color_primary="#d0d0d6"
    ),

    "vbachgen": gr.themes.Base(
    primary_hue="rose",
    spacing_size="sm",
    font=[gr.themes.GoogleFont('Tektur')],
    ),

    "mvsep": gr.themes.Base( # Тема соответствующая цветовой стилистике MVSep.com
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Montserrat"), "Arial", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Roboto Mono"), "Courier New", "monospace"]
    ).set(
    button_primary_background_fill="#3a7bd5",
    button_primary_background_fill_hover="#2c65c0",
    button_primary_text_color="#ffffff",
    input_background_fill="#ffffff",
    input_border_color="#d0d0d6",
    block_background_fill="#ffffff",
    border_color_primary="#d0d0d6"
    )
}