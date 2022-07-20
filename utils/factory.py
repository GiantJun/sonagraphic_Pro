from methods.sonagraph_finetune import Sonagraph_Finetune
from methods.test_model import TestModel

def get_trainer(trainer_id, args, seed):
    name = args['method'].lower()
    if name == 'finetune':
        return Sonagraph_Finetune(trainer_id, args, seed)
    elif name == 'test':
        return TestModel(trainer_id, args, seed)
    else:
        raise NotImplementedError('Unknown method {}'.format(name))

