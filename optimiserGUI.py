import ipywidgets as widgets
from ipywidgets import Layout, HBox, VBox
from IPython.display import display, clear_output
if 'google.colab' in str(get_ipython()):
    from AlloyML_Public.optimiser import *
else:
    from optimiser import *


def extractSettingsFromGUI(GUI_inputs, mode):
    settings = scanSettings(mode)

    for key in settings.range_based_inputs:
        settings.range_based_inputs[key] = [GUI_inputs['range_based_inputs'][key][0].value,
                                            GUI_inputs['range_based_inputs'][key][1].value]

    if bool(settings.constant_inputs):
        for key in GUI_inputs['constant_inputs']:
            settings.constant_inputs[key] = GUI_inputs['constant_inputs'][key].value

    for key in settings.categorical_inputs:
        settings.categorical_inputs[key] = []
        for index, value in enumerate(settings.categorical_inputs_info[key]['span']):
            if GUI_inputs['categorical_inputs'][key][index].value:
                settings.categorical_inputs[key].append(settings.categorical_inputs_info[key]['span'][index])

    settings.max_steps = int(GUI_inputs["scan_settings"]["Max Steps"].value)

    for key in settings.targets:
        settings.targets[key] = GUI_inputs['scan_settings'][key].value
    return settings

def generateModeSelectionGUI(mode = 'DoS'):
    mode_dropdown = widgets.Dropdown(
        options=['DoS', 'Mechanical'],
        value=mode,
        description='<b>Select Mode:</b>',
        style={'description_width': 'initial'},
        disabled=False
    )
    display(mode_dropdown)
    generateMainGUI(mode)
    def on_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            clear_output(wait=True)
            generateModeSelectionGUI(change['new'])
    mode_dropdown.observe(on_change)

def generateMainGUI(mode):
    settings = scanSettings(mode)
    KEY_LABEL_WIDTH = "40px"
    TO_LABEL_WIDTH = "15px"
    INPUT_BOX_WIDTH = "60px"
    INPUT_BOX_HEIGHT = "20px"

    LEFT_RIGHT_PADDING = Layout(margin="0px 30px 0px 30px")
    BOTTOM_PADDING = Layout(margin="0px 0px 5px 0px")

    default_input_box_layout = Layout(width=INPUT_BOX_WIDTH, height=INPUT_BOX_HEIGHT)

    GUI_inputs = {"range_based_inputs": {},
                  "constant_inputs": {},
                  "categorical_inputs": {},
                  "scan_settings": {}
                  }

    range_based_inputs_VBox = [widgets.HTML("<b>Ranged-based Inputs</b>")]
    for key in settings.range_based_inputs:
        key_label = widgets.Label(f"{key}:", layout=Layout(width=KEY_LABEL_WIDTH))
        lower_bound_box = widgets.FloatText(value=settings.range_based_inputs[key][0], layout=default_input_box_layout)
        to_label = widgets.Label("to", layout=Layout(width=TO_LABEL_WIDTH))
        upper_bound_box = widgets.FloatText(value=settings.range_based_inputs[key][1], layout=default_input_box_layout)
        range_based_inputs_VBox.append(HBox([key_label, lower_bound_box, to_label, upper_bound_box]))
        GUI_inputs["range_based_inputs"][key] = [lower_bound_box, upper_bound_box]

    if bool(settings.constant_inputs):
        constant_inputs_VBox = [widgets.HTML("<b>Constant Inputs</b>")]
        for key in settings.constant_inputs:
            key_label = widgets.Label(f"{key}:", layout=Layout(width=KEY_LABEL_WIDTH * 10))
            value_box = widgets.FloatText(value=settings.constant_inputs[key], layout=default_input_box_layout)
            constant_inputs_VBox.append(HBox([key_label, value_box]))
            GUI_inputs["constant_inputs"][key] = value_box

    categorical_inputs_VBox = [widgets.HTML("<b>Categorical Inputs</b>")]
    for key in settings.categorical_inputs:
        categorical_inputs_VBox.append(widgets.HTML(f'{key}:'))
        GUI_inputs["categorical_inputs"][key] = []
        for i, value in enumerate(settings.categorical_inputs_info[key]['span']):
            value_checkbox = widgets.Checkbox(description=settings.categorical_inputs_info[key]['tag'][i],
                                              disabled=False,
                                              indent=False)
            if value in settings.categorical_inputs[key]:
                value_checkbox.value = True
            categorical_inputs_VBox.append(value_checkbox)
            GUI_inputs["categorical_inputs"][key].append(value_checkbox)

    scan_settings_VBox = [widgets.HTML("<b>Scan Settings</b>")]
    label = widgets.Label("Max Steps: ")
    input_box = widgets.FloatText(value=settings.max_steps, layout=default_input_box_layout)
    scan_settings_VBox.append(HBox([label, input_box]))
    GUI_inputs["scan_settings"]["Max Steps"] = input_box
    for key in settings.targets:
        input_box = widgets.FloatText(value=settings.targets[key], layout=default_input_box_layout)
        row = HBox([widgets.HTML(f'Target {key}:'), input_box])
        scan_settings_VBox.append(row)
        GUI_inputs["scan_settings"][key] = input_box

    first_column = VBox(range_based_inputs_VBox)

    if settings.constant_inputs:
        second_column = VBox([VBox(constant_inputs_VBox, layout=BOTTOM_PADDING),
                              VBox(categorical_inputs_VBox, layout=BOTTOM_PADDING),
                              VBox(scan_settings_VBox)], layout=LEFT_RIGHT_PADDING)
    else:
        second_column = VBox([VBox(categorical_inputs_VBox, layout=BOTTOM_PADDING),
                              VBox(scan_settings_VBox)], layout=LEFT_RIGHT_PADDING)
    display(HBox([first_column, second_column]))

    run_scan_button = widgets.Button(description="Run Optimiser")
    display(run_scan_button)

    def on_button_clicked(b):
        optimiser(extractSettingsFromGUI(GUI_inputs, mode))

    run_scan_button.on_click(on_button_clicked)
