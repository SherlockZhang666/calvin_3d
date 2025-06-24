"""'VR Data Collection and Rendering
:copyright: 2019 by Oier Mees, Lukas Hermann, Wolfram Burgard
:license: GPLv3, see LICENSE for more details.
"""

__version__ = "0.0.1"
__project__ = "calvin_env"
__author__ = "Oier Mees, Lukas Hermann"
__license__ = "GPLv3"
__email__ = "meeso@informatik.uni-freiburg.de, hermannl@informatik.uni-freiburg.de,"

# First Version (6.23)
# API for Calvin evaluation environment
try:
    from .evaluation_env import CalvinEvaluationEnv
    
    def make_calvin_env(dataset_path, show_gui=False, num_sequences=1000):
        """ Can be called to create a Calvin evaluation environment instance."""
        return CalvinEvaluationEnv(dataset_path, show_gui, num_sequences)
    
    __all__ = ["CalvinEvaluationEnv", "make_calvin_env"]
    
except ImportError as e:
    # Error handling for import failure
    import warnings
    warnings.warn(f"Calvin environment import failure: {e}")
    __all__ = []