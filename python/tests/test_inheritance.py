import pytest

import ecole.configuring
import ecole.base

# class MyObservation(ecole.base.Observation):
#     def __init__(self):
#         super().__init__()
# 
# 
# def test_observation(model):
# #     conf = ecole.configuring.Configure("conflict/lpiterations")
# # presolving/maxrounds
#     obs = MyObservation()
#     env = ecole.configuring.Env.make_dummy("conflict/lpiterations")
#     for _ in range(2):
#         count = 0
#         obs, done = env.reset(model.clone())
#         while not done:
#             obs, reward, done, info = env.step(0)
#             count += 1
#         assert count == 1
