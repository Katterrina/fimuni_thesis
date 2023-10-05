import os 

data_root = os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.dirname(__file__)
            ),
            'data'
        )
)

def path(relp):
    return os.path.join(data_root, os.path.normpath(relp))
