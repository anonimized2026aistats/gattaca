from gymnasium import register

register(id="gym-PBN/PBNEnv-v0", entry_point="gym_PBN.envs:PBNEnv")

register(id="gym-PBN/PBNBangEnv-v0", entry_point="gym_PBN.envs:PBNBangEnv")

register(id="gym-PBN/ControlPBNEnv-v0", entry_point="gym_PBN.envs:ControlPBNEnv")


register(id="gym-PBN/PBN-target-v0", entry_point="gym_PBN.envs:PBNTargetEnv")

register(
    id="gym-PBN/Bittner-28-v0",
    entry_point="gym_PBN.envs:Bittner28",
    nondeterministic=True,
    max_episode_steps=100,
)

register(
    id="gym-PBN/Bittner-30-v0",
    entry_point="gym_PBN.envs:Bittner30",
    nondeterministic=True,
    max_episode_steps=100,
)

register(
    id="gym-PBN/Bittner-70-v0",
    entry_point="gym_PBN.envs:Bittner70",
    nondeterministic=True,
    max_episode_steps=100,
)

register(
    id="gym-PBN/Bittner-7-v0",
    entry_point="gym_PBN.envs:Bittner7",
    nondeterministic=True,
    max_episode_steps=100,
)

register(
    id="gym-PBN/Bittner-10-v0",
    entry_point="gym_PBN.envs:Bittner10",
    nondeterministic=True,
    max_episode_steps=100,
)

register(
    id="gym-PBN/Bittner-50-v0",
    entry_point="gym_PBN.envs:Bittner50",
    nondeterministic=True,
    max_episode_steps=100,
)

register(
    id="gym-PBN/Bittner-100-v0",
    entry_point="gym_PBN.envs:Bittner100",
    nondeterministic=True,
    max_episode_steps=100,
)

register(
    id="gym-PBN/Bittner-200-v0",
    entry_point="gym_PBN.envs:Bittner200",
    nondeterministic=True,
    max_episode_steps=100,
)

register(id="gym-PBN/PBN-sampled-data-v0", entry_point="gym_PBN.envs:PBNSampledDataEnv")

register(
    id="gym-PBN/PBN-self-triggering-v0", entry_point="gym_PBN.envs:PBNSelfTriggeringEnv"
)

register(id="gym-PBN/PBCN-v0", entry_point="gym_PBN.envs:PBCNEnv")

register(
    id="gym-PBN/PBCN-sampled-data-v0", entry_point="gym_PBN.envs:PBCNSampledDataEnv"
)

register(
    id="gym-PBN/PBCN-self-triggering-v0",
    entry_point="gym_PBN.envs:PBCNSelfTriggeringEnv",
)

register(
    id="gym-PBN/BittnerMulti-7-v0",
    entry_point="gym_PBN.envs:BittnerMulti7",
    nondeterministic=True,
    max_episode_steps=100,
)

register(
    id="gym-PBN/BittnerMulti-10-v0",
    entry_point="gym_PBN.envs:BittnerMulti10",
    nondeterministic=True,
    max_episode_steps=100,
)

register(
    id="gym-PBN/BittnerMulti-20-v0",
    entry_point="gym_PBN.envs:BittnerMulti20",
    nondeterministic=True,
    max_episode_steps=100,
)

register(
    id="gym-PBN/BittnerMulti-25-v0",
    entry_point="gym_PBN.envs:BittnerMulti25",
    nondeterministic=True,
    max_episode_steps=100,
)

register(
    id="gym-PBN/BittnerMulti-28-v0",
    entry_point="gym_PBN.envs:BittnerMulti28",
    nondeterministic=True,
    max_episode_steps=100,
)

register(
    id="gym-PBN/BittnerMulti-30-v0",
    entry_point="gym_PBN.envs:BittnerMulti28",
    nondeterministic=True,
    max_episode_steps=100,
)

register(
    id="gym-PBN/BittnerMulti-50-v0",
    entry_point="gym_PBN.envs:BittnerMulti50",
    nondeterministic=True,
    max_episode_steps=100,
)

register(
    id="gym-PBN/BittnerMultiGeneral-v0",
    entry_point="gym_PBN.envs:BittnerMultiGeneral",
    nondeterministic=True,
    max_episode_steps=100,
)

register(
    id="gym-PBN/BittnerControlGeneral-v0",
    entry_point="gym_PBN.envs:BittnerControlGeneral",
    nondeterministic=True,
    max_episode_steps=100,
)

register(
    id="gym-PBN/GTEx-v0",
    entry_point="gym_PBN.envs:PBNGTEx",
    nondeterministic=True,
    max_episode_steps=100,
)
