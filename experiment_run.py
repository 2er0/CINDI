import gc
import time
from datetime import datetime
from functools import partial
from typing import Iterator
from torch.multiprocessing import Process

from loguru import logger

from models.real_nvp import DEVICE
# from models.real_nvp import update_device
# disable GPU use only for local debugging
# update_device('cpu')

from global_utils import generator_seek, flow_parse_args, format_exception
from dataset.mtads_loading import load_all_stored_datasets

# START MAIN EXPERIMENTS FOR ANEO
from execute_fixing_selective_loop_v1_aneo import run as run_fixing_selective_loop_v1_aneo
from execute_train_only_v1_aneo import run as run_train_only_v1_aneo
from execute_simple_skip_v1_aneo import run as run_simple_skip_v1_aneo
from execute_simple_impute_v1_aneo import run as run_simple_impute_v1_aneo
# END MAIN EXPERIMENTS FOR ANEO
# START MAIN EXPERIMENTS FOR FSB
from execute_simple_skip_v1_fsb import run as run_simple_skip_v1_fsb
from execute_simple_impute_v1_fsb import run as run_simple_impute_v1_fsb
from execute_train_only_v1_fsb import run as run_train_only_v1_fsb
from execute_fixing_selective_loop_v1_fsb import run as run_fixing_selective_loop_v1_fsb
# END MAIN EXPERIMENTS FOR FSB
# START DIFFERENT METHOD TEST FOR ANEO
from execute_simple_with_models_v1_aneo import run as run_simple_with_method_v1_aneo

# END DIFFERENT METHOD TEST FOR ANEO

executor_map = {
    # START ANEO specific experiments
    "fixing_selective_loop_v1_aneo": run_fixing_selective_loop_v1_aneo,
    "train_only_v1_aneo": run_train_only_v1_aneo,
    "simple_skip_v1_aneo": run_simple_skip_v1_aneo,
    "simple_impute_v1_aneo": run_simple_impute_v1_aneo,
    # END ANEO specific experiments
    # START FSB specific experiments
    "simple_skip_v1_fsb": run_simple_skip_v1_fsb,
    "simple_impute_v1_fsb": run_simple_impute_v1_fsb,
    "train_only_v1_fsb": run_train_only_v1_fsb,
    "fixing_selective_loop_v1_fsb": run_fixing_selective_loop_v1_fsb,
    # END FSB specific experiments
    # START ANEO different method test
    "simple_dynamix_v1_aneo": partial(run_simple_with_method_v1_aneo, interpolation_method="dynamix"),
    "simple_knowimp_v1_aneo": partial(run_simple_with_method_v1_aneo, interpolation_method="knowimp"),
    # END ANEO different method test
}


def create_generator_for_runs(generators: Iterator, run_args: dict):
    for generator in generators:
        # make n runs for each type
        for i in range(run_args["run"]["nruns"]):
            yield generator, i


def run_one_exp(run_args, generator, i_run):
    try:
        # run one experiment combination
        device = DEVICE
        logger.info(" | ".join(["START", generator[0], str(i_run + 1),
                                "| ModelType:", run_args["model"]["model_type"],
                                "max.Past:", str(run_args["dataset"]["max_past_range"])]))

        run = executor_map[run_args["project"]["experiment"]]
        run(run_args=run_args, generator=generator, device=device)

        time.sleep(2)

        # catch all exceptions and continue with the next run
    except ValueError as e:
        logger.warning('ValueError: An exception occurred: {}'.format(e))
        print(format_exception(e))
    except BrokenPipeError as e:
        logger.warning('BrokenPipeError: An exception occurred: {}'.format(e))
        print(format_exception(e))
        time.sleep(60)
    except FileExistsError as e:
        logger.warning('FileExistsError: An exception occurred: {}'.format(e))
        print(format_exception(e))
    except OSError as e:
        logger.warning('OSError: An exception occurred: {}'.format(e))
        print(format_exception(e))
    except BaseException as e:
        logger.warning('BaseException: An exception occurred: {}'.format(e))
        print(format_exception(e))
    except Exception as e:
        logger.warning('Exception: An exception occurred: {}'.format(e))
        print(format_exception(e))
    finally:
        logger.info(" | ".join(["END", generator[0], str(i_run + 1),
                                "| ModelType:", run_args["model"]["model_type"],
                                "Past:", str(run_args["dataset"]["max_past_range"])]))
        # time.sleep(5)


def run_exp(run_args: dict, generators: Iterator):
    for generator, i_run in create_generator_for_runs(generators,
                                                      run_args):
        # run one experiment provided by the generator and the run arguments
        logger.info("Start: Process to run one experiment")
        run_wrapper = partial(run_one_exp, run_args=run_args, generator=generator, i_run=i_run)
        p = Process(target=run_wrapper)
        start_time = datetime.now()
        p.start()
        while p.is_alive():
            time.sleep(5)
            # kill after 24 hours
            if (datetime.now() - start_time).seconds > 60 * 60 * 24:
                logger.warning("Killing process due to timeout of 24 hours")
                p.kill()
                break
        gc.collect()
        logger.info("End: Process finished")


if __name__ == "__main__":
    # parse the arguments and run the experiment
    args = flow_parse_args()
    logger.info(args)
    drop = False
    # if args["dataset"]["name"] in ["fsb"]:  # run srb with noisy data in training as well
    #     drop = True

    # run the experiment
    run_exp(args, generator_seek(load_all_stored_datasets(args["dataset"]["name"]),
                                 args["dataset"]["generator_seek"],
                                 args["dataset"]["generator_stop"],
                                 drop))
