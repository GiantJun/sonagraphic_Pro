from methods.sonagraph_finetune import Sonagraph_Finetune
from methods.test_model import TestModel
from methods.emsemble_test_vote import Multi_Vote_Test
from methods.emsemble_test_avg import Multi_Avg_Test

def get_trainer(trainer_id, args, seed):
    name = args['method'].lower()
    if name == 'finetune':
        return Sonagraph_Finetune(trainer_id, args, seed)
    elif name == 'test':
        return TestModel(trainer_id, args, seed)
    elif name == 'emsemble_test_vote':
        return Multi_Vote_Test(trainer_id, args, seed)
    elif name == 'emsemble_test_avg':
        return Multi_Avg_Test(trainer_id, args, seed)
    else:
        raise NotImplementedError('Unknown method {}'.format(name))

