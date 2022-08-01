from methods.finetune import Finetune
from methods.test_model import TestModel
from methods.emsemble_test_vote import Multi_Vote_Test
from methods.emsemble_test_avg import Multi_Avg_Test
from methods.gen_grad_cam import Gen_Grad_CAM
from methods.contrastive import Contrastive_Methods

def get_trainer(trainer_id, config, seed):
    name = config.method.lower()
    if name == 'finetune':
        return Finetune(trainer_id, config, seed)
    elif name == 'test':
        return TestModel(trainer_id, config, seed)
    elif name == 'emsemble_test_vote':
        return Multi_Vote_Test(trainer_id, config, seed)
    elif name == 'emsemble_test_avg':
        return Multi_Avg_Test(trainer_id, config, seed)
    elif name == 'gen_grad_cam':
        return Gen_Grad_CAM(trainer_id, config, seed)
    elif name in ['mocov2', 'sup_simclr', 'simclr']:
        return Contrastive_Methods(trainer_id, config, seed)
    else:
        raise NotImplementedError('Unknown method {}'.format(name))

