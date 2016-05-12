

############################# Parameters ###############    
#Define input params
batch_size = 25                # batch_size i.e. #(experiences) sampled from exp dataset
im_height = im_width = 80
HISTORY = 3
input_shape = (HISTORY,im_height,im_width)

#replay memory in #(experiences) 
EXPERIENCE_SIZE = 80001       #total replay memory dataset
POPULATE = 10000         


TARGET_UPDATE_FREQ = 3000     ## When to update target weights i.e. "fixed targets"
SAVE_NETWORK_FREQ = 20000   ## When to save network weights
LEARNING_RATE = 1e-6

#Greedy Approach (Deterministic Network))
INITIAL_EPSILON = .1     #initial value
FINAL_EPSILON = 0.90       #final value
EPSILON_ANNEAL = 1000000
EPSILON_CHANGE = (FINAL_EPSILON - INITIAL_EPSILON) / EPSILON_ANNEAL

#Rewards defined in wrapped_flappy_bird.py
DISCOUNT = 0.95

## TO LOAD QUEUE FROM DATABASE?
LOAD_POPULATED_QUEUE = True

PRETRAINED = False
PRETRAINED_PATH = 'saved_DDQN/DDQN_weights_iter_320000'

TRAIN_PRETRAINED = True
TRAIN_PRETRAINED_PATH = 'saved_DDQN/DDQN_weights_iter_440000'

### Epochs when to check average scoreDigitsepochs = [10000,30000,60000,1e5]
SAVE_QUEUE_FREQ = 30000
