from morl_baselines.multi_policy.pcn.pcn import PCN


class MOCustomRewardVector:
    def __init__(self):
        self.reward_vector = None
        
    def set_reward_vector(self, reward_vector):
        self.reward_vector = reward_vector

    
    
class PCN_CUSTOM_REWARD(PCN, MOCustomRewardVector):
    def __init__(self, env, scaling_factor, learning_rate = 0.001, gamma = 1, batch_size = 256, hidden_dim = 64, noise = 0.1, project_name = "MORL-Baselines", experiment_name = "PCN", wandb_entity = None, log = True, seed = None, device = "auto", model_class = None, relabel_buffer=True):
        super().__init__(env, scaling_factor, learning_rate, gamma, batch_size, hidden_dim, noise, project_name, experiment_name, wandb_entity, log, seed, device, model_class)
        self.relabel_buffer = relabel_buffer

    def train(self, **kwargs):
        self.max_buffer_size = kwargs.get("max_buffer_size", 100)
        super().train(**kwargs)

    def set_reward_vector(self, reward_vector):
        super().set_reward_vector(reward_vector)
        self.reward_vector = reward_vector
        
        if self.env.has_wrapper_attr("set_reward_vector_function"):
            self.env.get_wrapper_attr("set_reward_vector_function")(reward_vector)
        else:
            self.env = RewardVectorFunctionWrapper(self.env, reward_vector)
        if self.relabel_buffer:
            global_step = 0
            old_replay = deepcopy(self.experience_replay)
            self.experience_replay = []
            if len(self.experience_replay) > 0:
                new_experience_replay = []
                for transitions in old_replay:
                    acc_r = 0
                    new_transitions = []
                    for t, transition in enumerate(transitions):
                        reward = self.reward_vector(
                            transition.observation, transition.action, transition.next_observation, transition.terminal
                        ).detach().cpu().numpy()
                        new_transitions.append(Transition(transition.observation, transition.action, np.float32(reward).copy(), transition.next_observation, transition.terminated))
                        global_step += 1
                    # add episode in-place
                    self._add_episode(transitions, max_size=self.max_buffer_size, step=global_step)
            

    