## TO DO 
##TSNE visualization using last conv layer!

import sys,os
from subprocess import call
import numpy as np

sys.path.append('/home/athreyav/Desktop/545/545_Project/Pygame/game/')
import random
from collections import deque
from scipy.misc import imresize,imrotate

from six.moves import cPickle
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam,RMSprop
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
import all_models
import wrapped_flappy_bird as game
from termcolor import colored
from copy import deepcopy
import params_double_dqn as p
import matplotlib.pyplot as plt
import theano.tensor as T

sys.setrecursionlimit(10000000)

input_shape = p.input_shape


#############################################################################################################################################################################
## Custom loss
def clipped_max_objective(y_true, y_pred):

    coords = y_true.nonzero()[0],y_true.nonzero()[1]
    
    y_pred_ = y_pred[coords]
    y_true_ = y_true[coords]
   
    #print y_pred_.eval({y_true : np.array([[3,0],[0,4],[2,0],[10,0],[0,11]]).astype('float32'), y_pred: np.arange(100,110).reshape((5,2)).astype('float32') })
    ## NO loss clipping
    return T.mean((y_pred_ - y_true_)**2)


def save_queue(EXPERIENCE_MEMORY):
    with open('saved_DDQN/double_dqn_queue_new.pkl','wb') as f:
        cPickle.dump(EXPERIENCE_MEMORY,f,protocol=cPickle.HIGHEST_PROTOCOL)
        
    call(['rm','saved_DDQN/double_dqn_queue.pkl'])
    call(['mv','saved_DDQN/double_dqn_queue_new.pkl','saved_DDQN/double_dqn_queue.pkl'])



def select_action(action_states,prob):
    state = np.random.choice([0,1,2],p=prob)
    return action_states[state], state
    
'''Double DQN implementation : Get argmax over actions of Online Net and use Q-value for selected action with Target Net. That is choose which action with online model and choose value of that action with target model.'''
def get_targets(mini_batch,target_model,model):
    # mini_batch format : (input_state,action,reward,output_state,Terminal,epsilon)
    
    actions= np.argmax(np.asarray([item[1] for item in mini_batch]),axis=1).astype(int)
    next_state = np.concatenate(tuple([exp[3] for exp in mini_batch]),axis=0)
    train_inputs = np.concatenate(tuple([exp[0] for exp in mini_batch]),axis=0)
    online_action_select = model.predict_on_batch(next_state)
    Q_target = target_model.predict_on_batch(next_state)
    Q_target = Q_target[np.arange(Q_target.shape[0]),online_action_select.argmax(axis=1)]
    target = np.zeros(shape=(len(mini_batch),2))
    for item in range(len(mini_batch)):
        target[item,actions[item]] = mini_batch[item][2] + p.DISCOUNT*Q_target[item]*int(not mini_batch[item][-2])

    assert(target.shape[0] == p.batch_size)
    return target, train_inputs

   
    
def main(input_shape):
    optim = RMSprop(lr=p.LEARNING_RATE, rho = 0.9, epsilon=1e-06)    
    model = all_models.model_default(input_shape)
    action_states = [[0,1],[1,0]]
    gameState = game.GameState()
    highestScore = 0
    totScore = 0

    if p.TRAIN_PRETRAINED and p.LOAD_POPULATED_QUEUE:
        if not os.path.isfile(p.TRAIN_PRETRAINED_PATH):
            print 'Pretrained Weights not found. Check the path provided.'
            sys.exit(1)
        model.load_weights(p.TRAIN_PRETRAINED_PATH)
        model.compile(loss = clipped_max_objective, optimizer=optim)
        print 'Loading expereince queue from disk..should take 1-2 mins...'
        with open('saved_DDQN/double_dqn_queue.pkl','r') as f:
            EXPERIENCE_MEMORY  = cPickle.load(f)
        epsilon = EXPERIENCE_MEMORY[-1][-1]
        exp_num=int(p.TRAIN_PRETRAINED_PATH.split('_')[-1])
        updateCount = exp_num/3000
        p.POPULATE = 0

    elif p.LOAD_POPULATED_QUEUE and not p.PRETRAINED:
        model.compile(loss = clipped_max_objective, optimizer=optim)
        print 'Loading expereince queue from disk..should take 1-2 mins...'
        with open('saved_DDQN/double_dqn_queue.pkl','r') as f:
            EXPERIENCE_MEMORY  = cPickle.load(f)
        epsilon = p.INITIAL_EPSILON
        exp_num=0
        highestScore = 0
        updateCount = 0
        p.POPULATE = 0
        totScore=0
    
    elif p.PRETRAINED:
        model.compile(loss = clipped_max_objective, optimizer=optim)
        if not os.path.isfile(p.PRETRAINED_PATH):
            print 'Pretrained Weights not found. Check the path provided.'
            sys.exit(1)
        rgbDisplay, reward, tState = gameState.frame_step(np.array([0,1]))
        grayDisplay = np.dot(imresize(rgbDisplay, (80,80), interp='bilinear')[:,:,:3], [0.299, 0.587, 0.114])
        grayDisplay = [grayDisplay for _ in range(p.HISTORY)]
        input_state = np.stack(grayDisplay, axis=2).reshape((1,p.HISTORY)+grayDisplay[0].shape)
        run_pretrained(input_state,model,action_states,gameState)

    else:
        EXPERIENCE_MEMORY = deque(maxlen = p.EXPERIENCE_SIZE)
        model.compile(loss = clipped_max_objective, optimizer=optim)
        exp_num = 0
        epsilon = p.INITIAL_EPSILON           ## epsilon is probability with which we will choose network output
        totScore = 0
        updateCount = 0


    
    ## Sanity Check
    print colored('Sanity Check...','green')
    rgbDisplay, reward, tState = gameState.frame_step(np.array([0,1]))
    
    grayDisplay = np.dot(imresize(rgbDisplay, (80,80), interp='bilinear')[:,:,:3], [0.299, 0.587, 0.114])
    grayDisplay = [grayDisplay for _ in range(p.HISTORY)]
    input_state = np.stack(grayDisplay, axis=2).reshape((1,p.HISTORY)+grayDisplay[0].shape)
    model.predict(input_state,batch_size=1,verbose=1)

    if p.PRETRAINED and os.path.isfile(p.PRETRAINED_PATH):
        run_pretrained(input_state,model,action_states,gameState)
 
    
    print 'Saving Model Architecture to file...model_arch.json\n'
    with open('saved_DDQN/model_arch.json','w') as f:
        f.write(model.to_json())
    
    #Create target network            
    target_model = deepcopy(model)

        
######################################## Populate experience dataset #############################################################################################################
    
    
    #Save time by loading a populated queue       
    try:    
        while '545 grade' != 'A+':
            while p.POPULATE:
                print 'Take a coffee break while the network populates the replay database. %d experiences to go...\n\n' %(p.POPULATE)
                nn_out = model.predict(input_state,batch_size=1,verbose=0)
                nn_action = [[0,0]]
                nn_action[0][np.argmax(nn_out)] =1
                assert(len(nn_action+action_states)==3)
                action,rand_flag = select_action(nn_action+action_states,prob=[epsilon,(1-epsilon)/7,(1-epsilon)*6/7])
                rgbDisplay, reward, tState = gameState.frame_step(action)
                grayDisplay = (np.dot(np.fliplr(imrotate(imresize(rgbDisplay, (80,80), interp='bilinear'), -90))[:,:,:3], [0.299, 0.587, 0.114])).reshape((1,1,80,80))

                output_state = np.append(grayDisplay,input_state[:,:p.HISTORY-1,:,:], axis=1)

                EXPERIENCE_MEMORY.append(tuple((input_state,action,reward,output_state,tState,epsilon)))
                print 'MODE : ' +colored('POPULATE\n','blue',attrs=['bold'])+ 'EXPERIENCE # : %d\t EPSILON:  %f (fixed)\t'%(exp_num,epsilon) + 'REWARD : ' + colored('NA. Let birdy flap around for a while','magenta',attrs=['bold']) + '\t Max Q : %f'%nn_out.max()

                p.POPULATE-=1
                input_state = output_state
                if not p.POPULATE:
                    with open('saved_DDQN/double_dqn_queue.pkl','wb') as f:
                        cPickle.dump(EXPERIENCE_MEMORY,f,protocol=cPickle.HIGHEST_PROTOCOL)
                            
                            
#############################################################################################################################################################################

######################################################### TRAIN #############################################################################################################

            

            ## Get new state!
            print input_state.shape
            nn_out = model.predict(input_state,batch_size=1,verbose=0)
            nn_action = [[0,0]]
            nn_action[0][np.argmax(nn_out)] =1
            action,rand_flag = select_action(nn_action+action_states,prob=[epsilon,(1-epsilon)*1/7,(1-epsilon)*6/7])
            rgbDisplay, reward, tState = gameState.frame_step(action)
            grayDisplay = (np.dot(np.fliplr(imrotate(imresize(rgbDisplay, (80,80), interp='bilinear'), -90))[:,:,:3], [0.299, 0.587, 0.114])).reshape((1,1,80,80))
#            plt.imshow(grayDisplay[0,0,:])
#            plt.show()
            output_state = np.append(grayDisplay,input_state[:,:p.HISTORY-1,:,:], axis=1)
            
            if reward==1:
                rewString = colored(str(reward),'green',attrs=['bold'])
                totScore +=1
            elif 0<reward<1:
                rewString = colored(str(reward),'yellow',attrs=['bold'])
            else:
                totScore = 0
                rewString = colored(str(reward),'red',attrs=['bold'])
            
            if action[1]==1:
                _act = '(flap)'
            elif action[0]==1:
                _act = '(nothing)'
            else:
                print 'Invalid Action'
                sys.exit()
            
            #No need to check if experience memory is full since duh, its a deque. EDIT : For memory reasons, im limiting the size
            if len(EXPERIENCE_MEMORY) > 50000:
                EXPERIENCE_MEMORY.popleft()
                EXPERIENCE_MEMORY.append(tuple((input_state,action,reward,output_state,tState,epsilon)))
            else:
                EXPERIENCE_MEMORY.append(tuple((input_state,action,reward,output_state,tState,epsilon)))
            
            print '\n\nMODE : ' +colored('TRAINING HARD *eye of the tiger theme playing in the distance*\n','blue',attrs=['bold'])+ '(%d **) EXPERIENCE # : %d\t EPSILON : %f\t ACTION:  %s\t REWARD : %s\t Q-Value : %f\t\t Game Score : %s \t Highest Score : %d\t\t Updated Target %d times...' %(len(EXPERIENCE_MEMORY),exp_num, epsilon,colored('predicted '+_act,'green',attrs=['bold']) if rand_flag==0 else 'random '+_act,rewString,nn_out.max(),colored(str(totScore),'green',attrs=['dark']) if totScore >0 else colored(str(totScore),'grey'),highestScore,updateCount) 
            
            #Get mini-batch & GRAD DESCENT!
            mini_batch = random.sample(EXPERIENCE_MEMORY,p.batch_size)

            #Get target values for each experience in minibatch

            targets,mini_batch_inputs  = get_targets(mini_batch,target_model,model)
            
            #predict on batch i.e. test_on_batch    
            loss = model.train_on_batch(mini_batch_inputs,targets)
            print 'Loss : ' ,colored(str(loss),'cyan',attrs=['bold'])
            
            #Tread lightly and increase thine greediness gently, for there is enough in this game for birdy's need but not for birdy's greed!
            if epsilon <= p.FINAL_EPSILON:
                 epsilon += p.EPSILON_CHANGE


            #update target networks and write score to file
            if exp_num % p.TARGET_UPDATE_FREQ ==0:
                target_model = deepcopy(model)
                with open('score_details.log','a+') as f:
                    toWrite =  '(%d **) EXPERIENCE # : %d\t EPSILON : %f\t ACTION:  %s REWARD : %s\t Q-Value : %f\tGame Score : %s \t Highest Score : %d\t Updated Target %d times...\n\n\n' %(len(EXPERIENCE_MEMORY),exp_num, epsilon,'predicted '+_act if rand_flag==0 else 'random '+_act, rewString, nn_out.max(),str(totScore) if totScore >0 else str(totScore), highestScore,updateCount)  
                    f.write(toWrite)
                updateCount +=1
            

            # Save network periodically        
            if exp_num % p.SAVE_NETWORK_FREQ ==0:
                model.save_weights('saved_DDQN/DDQN_weights_iter_%d'%exp_num, overwrite= True)
                
            if exp_num % p.SAVE_QUEUE_FREQ == 0:
               save_queue(EXPERIENCE_MEMORY)
               pass
            
            #Update highest Score
            if totScore > highestScore:
                highestScore = totScore
                
                
            input_state = output_state
            exp_num+=1
            
    except KeyboardInterrupt:

        if raw_input('\nSave queue to file (Y/N) : ') != 'n':
            print '\n\n\nSaving queue to file before quitting...takes 1-2 mins'
            save_queue(EXPERIENCE_MEMORY)
            print 'Saved queue. Quitting...'

        else:
            print 'Queue NOT saved. Quitting...'
             
    except Exception as e:
        model.save_weights('saved_DDQN/DDQN_weights_iter_%d'%exp_num, overwrite= True)
        print e.message
        
#############################################################################################################################################################################


#################################################################################################################   RUN PRE-TRAINED MODEL ##################################

def run_pretrained(input_state,model,action_states,gameState):
    print '\n\nLoading pretrained weights onto model...'
   
    model.load_weights(p.PRETRAINED_PATH)
    epsilon=1
    while True:
        print 'Running pretrained model (no exploration) with weights at ', p.PRETRAINED_PATH 
               
        nn_out = model.predict(input_state,batch_size=1,verbose=0)
        nn_action = [[0,0]]
        nn_action[0][np.argmax(nn_out)] =1
        action,rand_flag = select_action(nn_action+action_states,prob=[epsilon,(1-epsilon)*1/7,(1-epsilon)*6/7])
        rgbDisplay, reward, tState = gameState.frame_step(action)
        grayDisplay = (np.dot(np.fliplr(imrotate(imresize(rgbDisplay, (80,80), interp='bilinear'), -90))[:,:,:3], [0.299, 0.587, 0.114])).reshape((1,1,80,80))
        output_state = np.append(grayDisplay,input_state[:,:p.HISTORY-1,:,:], axis=1)
        

#############################################################################################################################################################################

if __name__ == "__main__":


    #Run Main

    main(input_shape)

        
    
