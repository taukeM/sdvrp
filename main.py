from sdvrp.agent import A2CAgent
from sdvrp.attention_model import AttentionModel
from sdvrp.env import DataGenerator, Env
from sdvrp.baseline import RolloutBaseline as Baseline

args = {
    'n_epochs': 2,
    'n_batch': 1,
    'batch_size': 4,
    'n_nodes': 6,
    'initial_demand_size': 2,
    'max_load': 9,
    'speed': 0.5,
    'lambda': 10,
    'data_dir': '',
    'log_dir': '',
    'save_path': '',
    'decode_len': 13,
    'actor_net_lr': 0.001,
    'lr_decay': 1.0,
    'max_grad_norm': 1.0,
    'save_interval': 1,
    'bl_alpha': 0.05,
    'embedding_dim': 128,

}
data_generator = DataGenerator(args)
data = data_generator.get_test_all()
env = Env(args)
model = AttentionModel(args['embedding_dim'], args['embedding_dim'], args['n_nodes'])
agent = A2CAgent(model, args, env, data_generator)
baseline = Baseline(agent, agent.model, args, data_generator)
agent.train_epochs(baseline)