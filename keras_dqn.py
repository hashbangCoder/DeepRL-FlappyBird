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
import params as p
import matplotlib.pyplot as plt
import theano.tensor as T
from theano import shared

sys.setrecursionlimit(10000000)

input_shape = p.input_shape


#############################################################################################################################################################################
## Custom loss
def clipped_max_objective(y_true, y_pred):

    #samp_true = np.random.randint(0,2,(4,2)).astype('float32')
    #pred_samp = np.random.rand(4,2).astype('float32')
    coords = y_true.nonzero()[0],y_true.nonzero()[1]
    
    y_pred_ = y_pred[coords]
    y_true_ = y_true[coords]
    ### Implement loss clipping    
    #return T.clip(T.mean((y_pred - y_true)**2),-1,1)

    #print y_pred_.eval({y_pred:pred_samp, y_true:samp_true}), '\n',pred_samp, '\n', samp_true
    #### NO LOSS CLIPPING
    return T.mean((y_pred_ - y_true_)**2)


def save_queue(EXPERIENCE_MEMORY):
    with open('saved nets/saved_queue_new.pkl','wb') as f:
        cPickle.dump(EXPERIENCE_MEMORY,f,protocol=cPickle.HIGHEST_PROTOCOL)
        
    call(['rm','saved nets/saved_queue.pkl'])
    call(['mv','saved nets/saved_queue_new.pkl','saved nets/saved_queue.pkl'])



def select_action(action_states,prob):
    state = np.random.choice([0,1,2],p=prob)
    return action_states[state], state
    
def get_targets(mini_batch,target_model):
    
    # mini_batch format : (input_state,action,reward,output_state,tState,epsilon)
    actions= np.argmax(np.asarray([item[1] for item in mini_batch]),axis=1).astype(int)
    state_inputs = np.concatenate(tuple([exp[3] for exp in mini_batch]),axis=0)
    train_inputs = np.concatenate(tuple([exp[0] for exp in mini_batch]),axis=0)
    est_values = (target_model.predict_on_batch(state_inputs)).max(axis=1)
    target = np.zeros(shape=(len(mini_batch),2))
    for item in range(len(mini_batch)):
        target[item,actions[item]] = mini_batch[item][2] + p.DISCOUNT*est_values[item]*int(not mini_batch[item][-2])
    #target = np.asarray([mini_batch[item][2] + p.DISCOUNT*est_values[item]  if not mini_batch[item][-2] else mini_batch[item][2] for item in range(len(mini_batch))])
    #assert(target.shape[0] == p.batch_size)

    return target, train_inputs
    
   
    
def main(input_shape):
    optim = Adam(lr=p.LEARNING_RATE, rho = 0.9, epsilon=1e-06)    
    model = all_models.model_default(input_shape)
    action_states = [[0,1],[1,0]]
    gameState = game.GameState()
    epsilon = 0.1           ## epsilon is probability with which we will choose network output
    
    if p.TRAIN_PRETRAINED and os.path.isfile(p.TRAIN_PRETRAINED_PATH):
        model.load_weights(p.TRAIN_PRETRAINED_PATH)
        model.compile(loss = clipped_max_objective, optimizer=optim)
        exp_num=int(p.TRAIN_PRETRAINED_PATH.split('_')[-1])
    else:
        model.compile(loss = clipped_max_objective, optimizer=optim)
        exp_num = 0
        
    
         ## Add new experiences to the right and pop from left
    totScore = 0
    highestScore = 0
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
    with open('saved nets/model_arch.json','w') as f:
        f.write(model.to_json())


    
    #Create target network            
    target_model = deepcopy(model)

    if not os.path.isfile('saved nets/weights_for_target_only'):
        model.save_weights('saved nets/weights_for_target_only')
    target_model.load_weights('saved nets/weights_for_target_only')

        
######################################## Populate experience dataset #############################################################################################################
    
    
    #Save time by loading a populated queue
    if p.LOAD_POPULATED_QUEUE:
        with open('saved nets/saved_queue.pkl','rb') as f:
        #with open('saved nets/saved_queue.pkl','rb') as f:
            print 'Loading expereince queue from disk..should take 1-2 mins...'
            EXPERIENCE_MEMORY = cPickle.load(f)
        if not p.TRAIN_PRETRAINED:
            epsilon = p.INITIAL_EPSILON
        else:
            epsilon = EXPERIENCE_MEMORY[-1][-1]

        p.POPULATE = 0
            
    else:
        EXPERIENCE_MEMORY = deque(maxlen = p.EXPERIENCE_SIZE)
        
        
    try:    
        while '545 grade' != 'A+':
            while p.POPULATE:
                print 'Take a coffee break while the network populates the replay database. %d experiences to go...\n\n' %(p.POPULATE)
                nn_out = model.predict(input_state,batch_size=1,verbose=0)
                nn_action = [[0,1]] if np.argmax(nn_out) else [[1,0]]
                assert(len(nn_action+action_states)==3)
                action,rand_flag = select_action(nn_action+action_states,prob=[epsilon,(1-epsilon)/7,(1-epsilon)*6/7])
                rgbDisplay, reward, tState = gameState.frame_step(action)
                grayDisplay = (np.dot(np.fliplr(imrotate(imresize(rgbDisplay, (80,80), interp='bilinear'), -90))[:,:,:3], [0.299, 0.587, 0.114])).reshape((1,1,80,80))

                output_state = np.append(grayDisplay,input_state[:,:2,:,:], axis=1)

                EXPERIENCE_MEMORY.append(tuple((input_state,action,reward,output_state,tState,epsilon)))
                print 'MODE : ' +colored('POPULATE\n','blue',attrs=['bold'])+ 'EXPERIENCE # : %d\t EPSILON:  %f (fixed)\t'%(exp_num,epsilon) + 'REWARD : ' + colored('NA. Let birdy flap around for a while','magenta',attrs=['bold']) + '\t Max Q : %f'%nn_out.max()

                p.POPULATE-=1
                input_state = output_state
                if not p.POPULATE:
                        totScore = 0
                        
                        with open('saved nets/saved_queue.pkl','wb') as f:
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
            #grayDisplay = (np.dot(np.fliplr(imrotate(imresize(rgbDisplay, (80,80), interp='bilinear'), -90))[:,:,:3], [0.299, 0.587, 0.114])).reshape((1,1,80,80))
            grayDisplay = (np.dot(np.fliplr(imrotate(imresize(rgbDisplay, (80,80), interp='bilinear'), -90))[:,:,:3], [0.299, 0.587, 0.114])).reshape((1,1,80,80))
#            plt.imshow(grayDisplay[0,0,:])
#            plt.show()
            output_state = np.append(grayDisplay,input_state[:,:2,:,:], axis=1)
            
            if reward==1:
                rewString = colored(str(reward),'green',attrs=['bold'])
                totScore +=1
            elif 0<reward<1:
                rewString = colored(str(reward),'yellow',attrs=['bold'])
            else:
                totScore = 0
                rewString = colored(str(reward),'red',attrs=['bold'])
            
            if action == action_states[0]:
                _act = 'flap'
            else:
                _act = 'nothing'
            
            #No need to check if experience memory is full since duh, its a deque. EDIT : For memory reasons, im limiting the size
            EXPERIENCE_MEMORY.append(tuple((input_state,action,reward,output_state,tState,epsilon)))
            print '\n\nMODE : ' +colored('TRAINING HARD *eye of the tiger theme playing in the distance*\n','blue',attrs=['bold'])+ '(%d **) EXPERIENCE # : %d\t EPSILON : %f\t ACTION:  %s\t REWARD : %s\t Q-Value : %f\t\t Game Score : %s \t Highest Score : %d\t\t Updated Target %d times...' %(len(EXPERIENCE_MEMORY),exp_num, epsilon,colored('predicted '+_act,'cyan') if rand_flag==0 else colored('random '+_act,'red'),rewString,nn_out.max(),colored(str(totScore),'green',attrs=['dark']) if totScore >0 else colored(str(totScore),'grey'),highestScore,updateCount) 
            
            #Get mini-batch & GRAD DESCENT!
            mini_batch = random.sample(EXPERIENCE_MEMORY,p.batch_size)

            #Get target values for each experience in minibatch
            targets,mini_batch_inputs  = get_targets(mini_batch,target_model)
            
            
            #predict on batch i.e. test_on_batch    
            loss = model.train_on_batch(mini_batch_inputs,targets)
            print 'Loss : ' ,colored(str(loss),'cyan')
            #Tread lightly and increase thine greediness gently, for there is enough in this game for birdy's need but not for birdy's greed!
            if epsilon <= p.FINAL_EPSILON:
                 epsilon += p.EPSILON_CHANGE


            #update target networks
            if exp_num % p.TARGET_UPDATE_FREQ ==0:
                
                model.save_weights('saved nets/weights_for_target_only',overwrite = True)
                target_model.load_weights('saved nets/weights_for_target_only')  
            #target weight integrity check
                for layer in range(len(target_model.layers)):
                    if target_model.layers[layer].get_weights():
                        assert(target_model.layers[layer].get_weights()[0] == model.layers[layer].get_weights()[0]).all()
                        
                updateCount +=1
            

            # Save network periodically        
            if exp_num % p.SAVE_NETWORK_FREQ ==0:
                model.save_weights('saved nets/weights_iter_%d'%exp_num, overwrite= True)
                
            if exp_num % p.SAVE_QUEUE_FREQ == 0:
               save_queue(EXPERIENCE_MEMORY)
               pass
            
            #Update highest Score
            if totScore > highestScore:
                highestScore = totScore
                
                
            input_state = output_state
            exp_num+=1
            
    except KeyboardInterrupt:

        if raw_input('\nPress "y" to save queue to file : ') == 'y':
            print '\n\n\nSaving queue to file before quitting...takes 1-2 mins'
            save_queue(EXPERIENCE_MEMORY)
            print 'Saved queue. Quitting...'

        else:
            print 'Queue NOT saved. Quitting...'
             
                
        
        


#############################################################################################################################################################################


#################################################################################################################   RUN PRE-TRAINED MODEL ##################################

def run_pretrained(input_state,model,action_states,gameState):
    print '\n\nLoading pretrained weights onto model...'
    model.load_weights(p.PRETRAINED_PATH)
    epsilon=1
    while True:
        print 'Running pretrained model (no exploration) with weights at ', p.PRETRAINED_PATH 
               
        nn_out = model.predict(input_state,batch_size=1,verbose=0)
        nn_action = [[0,1]] if np.argmax(nn_out) else [[1,0]]
        action,rand_flag = select_action(nn_action+action_states,prob=[epsilon,(1-epsilon)/2,(1-epsilon)/2])
        rgbDisplay, reward, tState = gameState.frame_step(action)
        #grayDisplay = (np.dot(imresize(rgbDisplay, (80,80), interp='bilinear')[:,:,:3], [0.299, 0.587, 0.114])).reshape((1,1,80,80))
        grayDisplay = (np.dot(np.fliplr(imrotate(imresize(rgbDisplay, (80,80), interp='bilinear'), -90))[:,:,:3], [0.299, 0.587, 0.114])).reshape((1,1,80,80))
        output_state = np.append(input_state[:,1:,:,:], grayDisplay,axis=1)
        


#############################################################################################################################################################################

if __name__ == "__main__":


    #Run Main

    main(input_shape)

        
    
