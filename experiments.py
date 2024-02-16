# from fastconv.experiment_configs.utils import Argument
from pds.utils.experiment_utils import Argument

arguments_list_of_lists = []

output_folder = "outputs-launch"

experiments_list = [
    # Argument(
    #     name="dessert_tgt_equals_src",
    #     arg_string="--config.prompt \'a DSLR, 4k photo of a table with desserts on it.\' --config.loss_coefficients 1 1.5 --config.no_eps_loss_use_proj_x0",
    # ),
    # Argument(
    #     name="fire_tgt_equals_src",
    #     arg_string="--config.prompt \'A fire with smoke\' --config.loss_coefficients 1 1.5 --config.no_eps_loss_use_proj_x0",
    # ),
    # Argument(
    #     name="astronaut_tgt_equals_src",
    #     arg_string="--config.prompt \'A photo of an astronaut riding a horse.\' --config.loss_coefficients 1 1.5 --config.no_eps_loss_use_proj_x0",
    # ),
    # Argument(
    #     name="dessert_tgt",
    #     arg_string="--config.prompt \'a DSLR, 4k photo of a table with desserts on it.\' --config.loss_coefficients 1 1.5",
    # ),
    # Argument(
    #     name="fire",
    #     arg_string="--config.prompt \'A fire with smoke\' --config.loss_coefficients 1 1.5",
    # ),
    # Argument(
    #     name="astronaut",
    #     arg_string="--config.prompt \'A photo of an astronaut riding a horse.\' --config.loss_coefficients 1 1.5",
    # ),
    # Argument(
    #     name="dessert_no_x_reverse_anneal_pds",
    #     arg_string="--config.prompt \'a DSLR, 4k photo of a table with desserts on it.\' " +
    #     "--config.loss_coefficients 0 1 " +
    #     "--config.pds_t_schedule.mode \'schedule\' " +
    #     "--config.pds_t_schedule.upper_bound_final 0.9 " +
    #     "--config.pds_t_schedule.lower_bound_final 0.5 " +
    #     "--config.pds_t_schedule.num_steps 1600 " +
    #     "--config.pds_t_schedule.warmup_steps 200 " +
    #     "--config.pds_t_schedule.schedule \'linear\' "
    # ),
    # Argument(
    #     name="fire_no_x_reverse_anneal_pds",
    #     arg_string="--config.prompt \'A fire with smoke.\' " +
    #     "--config.loss_coefficients 0 1 " +
    #     "--config.pds_t_schedule.mode \'schedule\' " +
    #     "--config.pds_t_schedule.upper_bound_final 0.9 " +
    #     "--config.pds_t_schedule.lower_bound_final 0.5 " +
    #     "--config.pds_t_schedule.num_steps 1600 " +
    #     "--config.pds_t_schedule.warmup_steps 200 " +
    #     "--config.pds_t_schedule.schedule \'linear\' "
    # ),
    # Argument(
    #     name="astronaut_no_x_reverse_anneal_pds",
    #     arg_string="--config.prompt \'A photo of an astronaut riding a horse.\' " +
    #     "--config.loss_coefficients 0 1 " +
    #     "--config.pds_t_schedule.mode \'schedule\' " +
    #     "--config.pds_t_schedule.upper_bound_final 0.9 " +
    #     "--config.pds_t_schedule.lower_bound_final 0.5 " +
    #     "--config.pds_t_schedule.num_steps 1600 " +
    #     "--config.pds_t_schedule.warmup_steps 200 " +
    #     "--config.pds_t_schedule.schedule \'linear\' "
    # ),
    Argument(
        name="dessert_no_x_anneal_pds",
        arg_string="--config.prompt \'a DSLR, 4k photo of a table with desserts on it.\' " +
        "--config.loss_coefficients 0 1 " +
        "--config.pds_t_schedule.mode \'schedule\' " +
        "--config.pds_t_schedule.upper_bound_final 0.4 " +
        "--config.pds_t_schedule.lower_bound_final 0.02 " +
        "--config.pds_t_schedule.num_steps 1600 " +
        "--config.pds_t_schedule.warmup_steps 200 " +
        "--config.pds_t_schedule.schedule \'linear\' "
    ),
    Argument(
        name="fire_no_x_anneal_pds",
        arg_string="--config.prompt \'A fire with smoke.\' " +
        "--config.loss_coefficients 0 1 " +
        "--config.pds_t_schedule.mode \'schedule\' " +
        "--config.pds_t_schedule.upper_bound_final 0.4 " +
        "--config.pds_t_schedule.lower_bound_final 0.02 " +
        "--config.pds_t_schedule.num_steps 1600 " +
        "--config.pds_t_schedule.warmup_steps 200 " +
        "--config.pds_t_schedule.schedule \'linear\' "
    ),
    Argument(
        name="astronaut_no_x_anneal_pds",
        arg_string="--config.prompt \'A photo of an astronaut riding a horse.\' " +
        "--config.loss_coefficients 0 1 " +
        "--config.pds_t_schedule.mode \'schedule\' " +
        "--config.pds_t_schedule.upper_bound_final 0.4 " +
        "--config.pds_t_schedule.lower_bound_final 0.02 " +
        "--config.pds_t_schedule.num_steps 1600 " +
        "--config.pds_t_schedule.warmup_steps 200 " +
        "--config.pds_t_schedule.schedule \'linear\' "
    ),
    Argument(
        name="dessert_no_x",
        arg_string="--config.prompt \'a DSLR, 4k photo of a table with desserts on it.\' " +
        "--config.loss_coefficients 0 1"
    ),
    Argument(
        name="fire_no_x",
        arg_string="--config.prompt \'A fire with smoke\' " +
        "--config.loss_coefficients 0 1"
    ),
    Argument(
        name="astronaut_no_x",
        arg_string="--config.prompt \'A photo of an astronaut riding a horse.\' " +
        "--config.loss_coefficients 0 1"
    ),
]
arguments_list_of_lists.append(experiments_list)


