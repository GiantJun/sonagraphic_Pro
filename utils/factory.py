from methods.finetune import Finetune
from methods.test_model import TestModel
from methods.emsemble_test import Multi_Avg_Test, Multi_Vote_Test
from methods.gen_grad_cam import Gen_Grad_CAM
from methods.contrastive import Contrastive_Methods

def get_trainer(trainer_id, config, seed):
    name = config.method.lower()
    if name == 'finetune' or name == 'retrain':
        return Finetune(trainer_id, config, seed)
    elif name == 'test':
        return TestModel(trainer_id, config, seed)
    elif name == 'emsemble_test_vote':
        return Multi_Vote_Test(trainer_id, config, seed)
    elif name == 'emsemble_test_avg':
        return Multi_Avg_Test(trainer_id, config, seed)
    elif name == 'gen_grad_cam':
        return Gen_Grad_CAM(trainer_id, config, seed)
    elif name in ['mocov2', 'sup_simclr', 'simclr', 'bal_sup_mocov2']:
        return Contrastive_Methods(trainer_id, config, seed)
    else:
        raise ValueError('Unknown method {}'.format(name))

