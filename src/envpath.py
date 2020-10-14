from dataclasses import dataclass
from .utils.PULog import logger
# This special script handles ALL environmental variable allow SINGLE import
# to get all necessary path and IDE friendly tracing.


path_keyword_raw = "path_raw"
path_keyword_interim = "path_interim"
path_keyword_processed = "path_processed"
path_keyword_output = "path_output"
path_keyword_eval = "path_eval"
path_keyword_model = "path_model"
path_keyword_test = "path_test"
path_keyword_trained_weights = "path_trained_weights"


@dataclass
class GetPath:
    """
    This is a centralized class variable that stores all path possibly used.
    """
    from pathlib import Path
    from environs import Env
    import os

    env = Env()
    env.read_env()
    logger.info("Loading paths from environmental file.")

    # Get training dataframe
    # This MUST be present at least, or else it will throw error.
    raw: Path = Path(env(path_keyword_raw))

    # Two subsequently inferred folder path.
    train_data: Path = raw / "train/"
    test_data: Path = raw / "test/"

    if path_keyword_interim not in os.environ:
        logger.warning(f"Enviromental variable {path_keyword_interim} not set.")
    else:
        interim: Path = Path(env(path_keyword_interim))
    if path_keyword_processed not in os.environ:
        logger.warning(f"Enviromental variable {path_keyword_processed} not set.")
    else:
        processed: Path = Path(env(path_keyword_processed))
    if path_keyword_output not in os.environ:
        logger.warning(f"Enviromental variable {path_keyword_output} not set.")
    else:
        output: Path = Path(env(path_keyword_output))
    if path_keyword_eval not in os.environ:
        logger.warning(f"Enviromental variable {path_keyword_eval} not set.")
    else:
        eval: Path = Path(env(path_keyword_eval))
    if path_keyword_model not in os.environ:
        logger.warning(f"Enviromental variable {path_keyword_model} not set.")
    else:
        model: Path = Path(env(path_keyword_model))
    if path_keyword_test not in os.environ:
        logger.warning(f"Enviromental variable {path_keyword_test} not set.")
    else:
        test: Path = Path(env(path_keyword_test))
    if path_keyword_trained_weights not in os.environ:
        logger.warning(f"Enviromental variable {path_keyword_trained_weights} not set.")
    else:
        trained_weights: Path = Path(env(path_keyword_trained_weights))


# This is THE Singleton that store all path information.

# from src.envpath import AllPaths
AllPaths = GetPath()
logger.info(AllPaths)


def test_AllPaths():
    print(AllPaths)
