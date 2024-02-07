# from fastconv.experiment_configs.utils import Argument
from pds.utils.experiment_utils import Argument

arguments_list_of_lists = []

output_folder = "outputs-launch"

# dataset_lists = [
#     Argument(
#         name="lego",
#         arg_string=f"--data ../../data/nerf_synthetic/lego/ --output-dir {output_folder}/lego",
#     ),        
# ]
# arguments_list_of_lists.append(dataset_lists)

experiments_list = [
    Argument(
        name="baseline",
        arg_string="--config.prompt \'A DSLR photo of a giraffe\'",
    ),
    Argument(
        name="baseline_low_CFG",
        arg_string="--config.prompt \'A DSLR photo of a giraffe\' --config.pds_cfg 30",
    ),
    Argument(
        name="baseline_higher_projection_t",
        arg_string="--config.prompt \'A DSLR photo of a giraffe\' --config.project_t_schedule.lower_bound_final 0.03 --config.project_t_schedule.lower_bound_final 0.06",
    ),
    Argument(
        name="baseline_highest_projection_t",
        arg_string="--config.prompt \'A DSLR photo of a giraffe\' --config.project_t_schedule.lower_bound_final 0.03 --config.project_t_schedule.lower_bound_final 0.1",
    ),
    Argument(
        name="baseline_no_thresholding",
        arg_string="--config.prompt \'A DSLR photo of a giraffe\' --config.thresholding None",
    ),
    Argument(
        name="baseline_pds_t_annealing",
        arg_string="--config.prompt \'A DSLR photo of a giraffe\' --config.pds_t_schedule.mode schedule --config.pds_t_schedule.schedule linear --config.pds_t_schedule.lower_bound_final 0.03 --config.project_t_schedule.lower_bound_final 0.1",
    ),
    Argument(
        name="baseline_dark_negative_prompt",
        arg_string="--config.prompt \'A DSLR photo of a giraffe\' --config.extra_src_prompts \', oversaturated, smooth, pixelated, dark, cartoon, foggy, hazy, blurry, bad structure, noisy, malformed\'",
    ),
]
arguments_list_of_lists.append(experiments_list)


