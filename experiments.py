# from fastconv.experiment_configs.utils import Argument
from pds.utils.experiment_utils import Argument

arguments_list_of_lists = []

output_folder = "outputs-launch"

# experiments_list = [
#     Argument(
#         name="baseline_high_lr_50_cfg",
#         arg_string="--config.prompt \'a photo of the inside of a smart home, indoor photography\'",
#     ),
#     Argument(
#         name="baseline_high_lr_50_cfg_seed",
#         arg_string="--config.prompt \'a photo of the inside of a smart home, indoor photography\' --config.seed 23",
#     ),
#     Argument(
#         name="baseline_low_CFG_no_thresholding --config.thresholding None",
#         arg_string="--config.prompt \'a photo of the inside of a smart home, indoor photography\' --config.pds_cfg 30",
#     ),
#     Argument(
#         name="baseline_higher_projection_t_high_lr_50_cfg",
#         arg_string="--config.prompt \'a photo of the inside of a smart home, indoor photography\' --config.project_t_schedule.lower_bound_final 0.03 --config.project_t_schedule.lower_bound_final 0.06",
#     ),
#     Argument(
#         name="baseline_highest_projection_t_high_lr_50_cfg",
#         arg_string="--config.prompt \'a photo of the inside of a smart home, indoor photography\' --config.project_t_schedule.lower_bound_final 0.03 --config.project_t_schedule.lower_bound_final 0.1",
#     ),
#     Argument(
#         name="baseline_no_thresholding",
#         arg_string="--config.prompt \'a photo of the inside of a smart home, indoor photography\' --config.thresholding None",
#     ),
#     Argument(
#         name="baseline_pds_t_annealing_high_lr_50_cfg",
#         arg_string="--config.prompt \'a photo of the inside of a smart home, indoor photography\' --config.pds_t_schedule.mode schedule --config.pds_t_schedule.schedule  linear --config.pds_t_schedule.lower_bound_final 0.03 --config.pds_t_schedule.upper_bound_final 0.1 --config.pds_t_schedule.warmup_steps 0 --config.pds_t_schedule.num_steps 3000",
#     ),
#     Argument(
#         name="baseline_dark_negative_prompt_high_lr_50_cfg",
#         arg_string="--config.prompt \'a photo of the inside of a smart home, indoor photography\' --config.extra_src_prompts \', oversaturated, smooth, pixelated, dark, cartoon, foggy, hazy, blurry, bad structure, noisy, malformed\'",
#     ),
# ]

experiments_list = [
    Argument(
        name="dessert_tgt_equals_src",
        arg_string="--config.prompt \'a DSLR, 4k photo of a table with desserts on it.\' --config.loss_coefficients 1 1.5 --config.no_eps_loss_use_proj_x0",
    ),
    Argument(
        name="fire_tgt_equals_src",
        arg_string="--config.prompt \'A fire with smoke\' --config.loss_coefficients 1 1.5 --config.no_eps_loss_use_proj_x0",
    ),
    Argument(
        name="astronaut_tgt_equals_src",
        arg_string="--config.prompt \'A photo of an astronaut riding a horse.\' --config.loss_coefficients 1 1.5 --config.no_eps_loss_use_proj_x0",
    ),
    Argument(
        name="dessert_tgt",
        arg_string="--config.prompt \'a DSLR, 4k photo of a table with desserts on it.\' --config.loss_coefficients 1 1.5",
    ),
    Argument(
        name="fire",
        arg_string="--config.prompt \'A fire with smoke\' --config.loss_coefficients 1 1.5",
    ),
    Argument(
        name="astronaut",
        arg_string="--config.prompt \'A photo of an astronaut riding a horse.\' --config.loss_coefficients 1 1.5",
    ),
    # Argument(
    #     name="more_eps_tgt_equals_src_50_cfg",
    #     arg_string="--config.prompt \'a photo of the inside of a smart home, indoor photography\' --config.loss_coefficients 1.2 1 --config.no_eps_loss_use_proj_x0  --config.pds_cfg 50",
    # ),
    # Argument(
    #     name="more_eps_50_cfg",
    #     arg_string="--config.prompt \'a photo of the inside of a smart home, indoor photography\' --config.loss_coefficients 1.2 1 --config.pds_cfg 50",
    # ),
    # Argument(
    #     name="more_eps_tgt_equals_src_seed_one_50_cfg",
    #     arg_string="--config.prompt \'a photo of the inside of a smart home, indoor photography\' --config.loss_coefficients 1.2 1 --config.no_eps_loss_use_proj_x0 --config.seed 12  --config.pds_cfg 50",
    # ),
    # Argument(
    #     name="more_eps_seed_one_50_cfg",
    #     arg_string="--config.prompt \'a photo of the inside of a smart home, indoor photography\' --config.loss_coefficients 1.2 1  --config.seed 12  --config.pds_cfg 50",
    # ),
    # Argument(
    #     name="more_x_tgt_equals_src",
    #     arg_string="--config.prompt \'a photo of the inside of a smart home, indoor photography\' --config.loss_coefficients 1.5 1 --config.no_eps_loss_use_proj_x0",
    # ),
    # Argument(
    #     name="more_x",
    #     arg_string="--config.prompt \'a photo of the inside of a smart home, indoor photography\' --config.loss_coefficients 1.5 1",
    # ),
    # Argument(
    #     name="more_x_tgt_equals_src_seed_one",
    #     arg_string="--config.prompt \'a photo of the inside of a smart home, indoor photography\' --config.loss_coefficients 1.5 1 --config.no_eps_loss_use_proj_x0 --config.seed 12",
    # ),
    # Argument(
    #     name="more_x_seed_one",
    #     arg_string="--config.prompt \'a photo of the inside of a smart home, indoor photography\' --config.loss_coefficients 1.5 1  --config.seed 12",
    # ),
    # Argument(
    #     name="x_only",
    #     arg_string="--config.prompt \'a photo of the inside of a smart home, indoor photography\' --config.loss_coefficients 1 0",
    # ),
    # Argument(
    #     name="eps_only",
    #     arg_string="--config.prompt \'a photo of the inside of a smart home, indoor photography\' --config.loss_coefficients 0 1",
    # ),
    # Argument(
    #     name="z_weighting",
    #     arg_string="--config.prompt \'a photo of the inside of a smart home, indoor photography\'",
    # ),
    # Argument(
    #     name="x_only_seed",
    #     arg_string="--config.seed 96 --config.prompt \'a photo of the inside of a smart home, indoor photography\' --config.loss_coefficients 1 0",
    # ),
    # Argument(
    #     name="eps_only_seed",
    #     arg_string="--config.seed 96 --config.prompt \'a photo of the inside of a smart home, indoor photography\' --config.loss_coefficients 0 1",
    # ),
    # Argument(
    #     name="z_weighting_seed",
    #     arg_string="--config.seed 96 --config.prompt \'a photo of the inside of a smart home, indoor photography\'",
    # ),
]
arguments_list_of_lists.append(experiments_list)


