from kaggle_environments import make
#from vivek_agent import agent1
from from_lux.game import Game
from from_isaiah.lux_gym.act_spaces import  BaseActSpace
from from_isaiah.lux_gym.act_spaces import  BaseActSpace
def agent_dummmy(observation, configuration):
        global game_state

        ### Do not edit ###
        if observation["step"] == 0:
            game_state = Game()
            game_state._initialize(observation["updates"])
            game_state._update(observation["updates"][2:])
            game_state.id = observation.player
            #model = NeuralNet(**get_dict_from_class(model_param))
        else:
            game_state._update(observation["updates"])
        actions = ['m u_1 n', 'm u_2 e']
        return actions
episodes=1
size=12
#my_agent=agent1(board_dim=(12,12))
for eps in range(episodes):
    print("=== Episode {} ===".format(eps))
    env = make("lux_ai_2021", configuration={"seed": 562124210, "loglevel": 2, "annotations": True,"width":size, "height":size}, debug=True)
    steps = env.run([lambda x: ['m u_1 n', 'm u_2 e'],lambda x: ['m u_1 n', 'm u_2 e']])
    env.render(mode="html", width=1200, height=800)