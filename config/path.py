AGENCY_LIST_PATH = './config/data/agency.json'
AGENCY_INV_LIST_PATH = './config/data/agency_inv.json'
AD_LIST_PATH = './config/data/ad.json'

LOG_PATH = './logs/'

model_input_folder = './shared/task/model_input/'
model_output_folder = './shared/task/model_output/'
filtered_model_output_folder = './shared/task/filtered_model_output/'
task_output_excel_folder = '/disk2/zyy/medOpinionSystem/shared/task/task_output_excel/'


manual_input_folder = './shared/manual/manual_input/'
manual_model_input_folder = './shared/manual/model_input/'
manual_model_output_folder = './shared/manual/model_output/'
manual_filtered_model_output_folder = './shared/manual/filtered_model_output/'
manual_task_output_excel_folder = '/disk2/zyy/medOpinionSystem/shared/manual/task_output_excel/'

crawler_path = './scripts/crawler.sh'
bispn_path = './scripts/bispn.sh'
bispn_m_path = './scripts/bispn_m.sh'

platform_map = {'小红书':'xhs', '知乎':'zhihu', '哔哩哔哩':'bili', '微博':'wb', '抖音':'dy'}
platform_map_inv = {'xhs':'小红书', 'zhihu':'知乎', 'bili':'哔哩哔哩', 'wb':'微博', 'dy':'抖音'}
head_labels = ["医疗机构", "部门/科室", "医生", "护士", "管理人员"]
tail_labels = [ "诊疗水平", "医学道德", "服务管理", "收费水平", "医疗监管", "社会口碑", 
               "人员待遇", "资源/团队", "环境/设施"]