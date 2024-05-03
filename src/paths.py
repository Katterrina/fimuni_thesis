import os

data_root = os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.dirname(__file__)
            ),
            'data'
        )
)

figures_root = os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.dirname(__file__)
            ),
            'figures'
        )
)

def path(relp):
    return os.path.join(data_root, os.path.normpath(relp))

def path_figures(relp):
    return os.path.join(figures_root, os.path.normpath(relp))