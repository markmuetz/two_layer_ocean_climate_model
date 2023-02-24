import ipywidgets as widgets


def box1():
    cb1 = widgets.Checkbox(
        value=False,
        description='Check me',
        disabled=False,
        indent=False
    )
    cb2 = widgets.Checkbox(
        value=False,
        description='Check me 2',
        disabled=False,
        indent=False
    )
    return widgets.Box(children=[cb1, cb2])

