from src.utils.file_utils import *
from config.path import *

makedir('./shared/')
makedir('./shared/task/')
makedir('./shared/manual/')
makedir(model_input_folder)
makedir(model_output_folder)
makedir(filtered_model_output_folder)
makedir(task_output_excel_folder)

makedir(manual_input_folder)
makedir(manual_model_input_folder)
makedir(manual_model_output_folder)
makedir(manual_filtered_model_output_folder)
makedir(manual_task_output_excel_folder)

makedir(LOG_PATH)