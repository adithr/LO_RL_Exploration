"""

Original Authors: R Hahnloser, A Zai (2019)
Current Author : Adithyan Radhakrishnan

Python version of SARSA learning code written in Matlab 
Original code repository:


"""

import numpy as np
from scipy.linalg import toeplitz as tp

###Parameter Declarations
"""
_h : for hearing birds
_d : for deaf birds
lo : Light-Off
W_matrix : Transition probability matrix for a markovian process
"""
# TO DO : Include plot settings 

no_of_notes = 3
lo_note = 2
no_of_actions = 6 
alpha = 0.001/no_of_notes

tau = 0.99
verbose = 0
gamma = 1
epsilon = 1e-40


#Simulation Parameters
no_of_trials = 1000 # per note
do_control=2 # add control condition without LO (0 = only LO, 1 = add control, 2 = remove entropy gain)
no_of_birds = 3

#Rewards
include_positive_rewards = False
no_of_rewards = 15

if include_positive_rewards:
    reward_space = np.array([-np.logspace(0.1,50,no_of_rewards),\
                    -np.logspace(0.1,5,no_of_rewards)]).reshape(-1)
    
    no_of_rewards = reward_space.size
    reward_space = np.sort(reward_space) # %%
else:
    reward_space = -np.logspace(0.1,50,no_of_rewards)
                       
#States
no_of_states_h = no_of_actions + 2 #hearing
no_of_states_d = 2 #deaf
total_no_of_states = no_of_states_h * no_of_notes + 1



###Sensorimotor model : motor action -> sensory states

h1 = np.zeros(no_of_states_h)
h1[:3] = np.array([1/4,1/2,1/4])
h2 = np.zeros(no_of_actions)
h2[0]  = 1/4

w_matrix_h = tp(h1,h2) # ???

w_matrix = np.zeros((total_no_of_states,no_of_notes*no_of_actions))
for i in range(0,no_of_notes):
    w_matrix[(i)*no_of_states_h:(i+1)*no_of_states_h,\
            (i)*no_of_actions:(i+1)*no_of_actions] = w_matrix_h

#Transition probabilities (?)
transition_matrix = np.cumsum(w_matrix,0)


###Variable Quantifying Model Simulation

#continencies
contingencies_h = np.zeros((no_of_birds,no_of_rewards))
contingencies_d = np.zeros((no_of_birds,no_of_rewards)) #same shape for deaf and hearing 


value_states_h = np.zeros((no_of_birds,no_of_rewards))
value_states_d = np.zeros((no_of_birds,no_of_rewards))

mean_value_states_h = np.zeros((no_of_birds,no_of_rewards,no_of_actions*no_of_notes))
mean_value_states_d = np.zeros((no_of_birds,no_of_rewards,no_of_actions*no_of_notes))


vta_plus_h = np.zeros((no_of_notes,no_of_birds))
vta_minus_h= np.zeros((no_of_notes,no_of_birds))
#actions_old_h = np.zeros(3) # %% Why 3
#actions_old_d = actions_old_h ## commented from previous version


### Learning Process
for control in range(do_control):
  if control==0:
    # % What is N_on?
    N_on = 1
  elif control == 1:
    N_on = no_of_trials
    print("Control (no LO)\n")
  else:
    N_on = 1
    print("Control (LO but no entropy)\n")
  
  for bird in range(no_of_birds):
    #VTA responses for different notes for specific reward Jplot
    vta_j_h = np.zeros((no_of_notes,no_of_trials))
    vta_j_d = np.zeros((no_of_notes,no_of_trials))
    
    # keeps track of LO (=1) or no LO (=0)
    lo_h = np.zeros(no_of_trials)
    lo_d = np.zeros(no_of_trials)
    print(f'Bird{bird}\n')

    for reward_no in range(no_of_rewards):
      reward_per_LO = reward_space[reward_no]
      if verbose:
        print(f'j={reward_no}, {no_of_rewards}')

      # Initialize
      # hearing
      v_sarsa_h = np.zeros(no_of_actions*no_of_notes)
      a_s_counter_h =  np.ones((total_no_of_states,no_of_actions*no_of_notes))
      a_s_counter_h_all = np.ones((total_no_of_states,no_of_actions*no_of_notes))
      no_of_lo_events_h = np.zeros((total_no_of_states, no_of_actions*no_of_notes))
      state_history_h = np.zeros(no_of_trials*no_of_notes) #state history
      action_history_h = np.zeros(no_of_trials*no_of_notes) #state history

      #Deaf
      v_sarsa_d = np.zeros(no_of_actions*no_of_notes)
      a_s_counter_d =  np.ones((total_no_of_states,no_of_actions*no_of_notes))
      a_s_counter_d_all = np.ones((total_no_of_states,no_of_actions*no_of_notes))
      no_of_lo_events_d = np.zeros((total_no_of_states, no_of_actions*no_of_notes))
      state_history_h = np.zeros(no_of_trials*no_of_notes) #state history
      action_history_h = state_history_h

      c_i = 0 # % What is this for?
      for trial in range(no_of_trials):
        for note in range(no_of_notes):
          #TODO : Add c_i at the end
          #c_i += 1 # % ??
          
        #states and actions corr to current note
          state_list = np.arange(note,(note+1)) * no_of_states_h 
          action_list = np.arange(note,(note+1)) * no_of_actions
          
          # Take action based on value fn
          #TO DO : 1. Ask Anja about the 0*rand 
          action_current_h = np.argmax(v_sarsa_h[action_list]) # max value from sarsa
          action_current_h = action_current_h + note*no_of_actions # scale it to the corr note

          action_history_h[c_i] = action_current_h

          # sample next state 
          # % Why is it sampled that way?
          epsilon    = np.random.rand(1)[0]
          
          #return the firststate where transition_probability > epsilon
          state_next_h = np.where(transition_matrix[:,action_current_h] > epsilon)[0][0] 
          state_history_h[c_i] = state_next_h
          
          # %possible error due to 0/1 start index
          is_lo_H = (trial > N_on) and (note + 1) == lo_note and \
                     (state_next_h+1) >((lo_note-1)*no_of_states_h+no_of_states_h/2)
          breakpoint()
          



