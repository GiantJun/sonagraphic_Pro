from methods.finetune import Finetune
from methods.test_model import TestModel

def get_trainer(trainer_id, args, seed):
    name = args['method'].lower()
    if name == 'finetune':
        return Finetune(trainer_id, args, seed)
    elif name == 'test':
        return TestModel(trainer_id, args, seed)
    else:
        raise NotImplementedError('Unknown method {}'.format(name))

